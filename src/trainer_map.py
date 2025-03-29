import copy
import time
from src.utils.util_pytorch import UtilPytorch
import numpy
import ray
import torch

@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, model,initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = model
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            with open("/dev/tty", "w") as terminal:    
                print("You are not training on GPU.\n",file=terminal)


        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                policy_loss,
            ) = self.update_weights(batch)
            if self.config.PER:
                replay_buffer.update_priorities.remote(priorities, index_batch)

            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            UtilPytorch.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss,
                }
            )

            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                norm = self.config.adjust_ratio(self.training_step, played_games)
                while (
                   norm*self.training_step
                    / max(
                        1, played_games
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                    norm = self.config.adjust_ratio(self.training_step, played_games)


    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch
      
        self.optimizer.zero_grad()

        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
 
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)

        policy_logits,value, reward, = self.model.initial_inference(
            observation_batch
        )
        curr_state = observation_batch
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, curr_state = self.model.recurrent_inference(
                curr_state, action_batch[:, i]
            )

            predictions.append((value, reward, policy_logits))

        ## Compute losses
        value_loss, policy_loss = (0, 0)
        value, reward, policy_logits = predictions[0]
        current_value_loss, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss

        priorities[:, 0] = (
            numpy.abs(value[:,0].detach().cpu().numpy() - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )
        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_policy[:, i],
            )

            value_loss += current_value_loss

            policy_loss += current_policy_loss

            priorities[:, i] = (
                numpy.abs(value[:,0].detach().cpu().numpy() - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        policy_loss = policy_loss.mean() / self.config.max_moves
        value_loss = value_loss.mean() * self.config.value_loss_weight / self.config.max_moves
        loss = value_loss + policy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=1)
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            loss.item(),
            value_loss.item(),
            policy_loss.item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


    @staticmethod
    def loss_function(
        value,
        policy_logits,
        target_value,
        target_policy,
    ):
        value = value.unsqueeze(-1)
        target_value = target_value.unsqueeze(-1)
        device = target_value.device

        policy_logits = policy_logits.to(device)
        value = value.to(device)
        
        mask_value = target_value == float('-inf')
        masked_value = torch.where(mask_value,0.,value)
        masked_target_value = torch.where(mask_value,0.,target_value)

        value_loss = ((masked_value-masked_target_value)**2)


        mask_policy =( policy_logits == -torch.inf).to(device)  | (target_policy == float('-inf')).to(device)
        masked_target_policy = torch.where(mask_policy,0.,target_policy)
        
        policy_logits = torch.nn.functional.log_softmax(policy_logits,dim=-1)
        masked_policy = torch.where(mask_policy,0.,policy_logits)
        policy_loss = (-masked_target_policy * masked_policy).sum(-1,keepdim=True)

        return value_loss, policy_loss

