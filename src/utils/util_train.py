from pathlib import Path
import numpy
from src.utils.util_replay_buffer import UtilReplayBuffer
from src.entities.training_results import TrainingResults
import sys
import numpy as np
from pympler import asizeof
from src.enums.enum_model_name import EnumModelName
class UtilTrain:

    @staticmethod
    def compute_target_value(config, game_history, index):
        bootstrap_index = index + config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
            )

            value = last_step_value * config.discount**config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            value += (
                reward
            ) * config.discount**i

        return value
    @staticmethod
    def compute_target_value_pre_train(mapping_history, index,last_gae):
        value = 0
        for i, (__,_,reward,v,sum_r) in enumerate(
            mapping_history[index:]
        ):
            value += (
                reward
            ) * 0.997**i
        return value

    @staticmethod
    def make_target_pre_train(mapping_history,gamma=0.99,lamb=0.95):
        """
        Generate targets for every unroll steps.
        """
        states,rewards,values = mapping_history
        last_gae = 0
        len_rollout = len(states)
        adv = [0 for i in range(len_rollout)]
        for t in reversed(range(len_rollout)):
            if t == len_rollout - 1:
                nextvalues = 0
            else:
                nextvalues = values[t+1]

            delta = rewards[t] + gamma * nextvalues - values[t]
            adv[t] = last_gae = delta + gamma * lamb * last_gae
        returns = (numpy.array(adv) + numpy.array(values)).tolist()

        return returns
    @staticmethod
    def make_target(config,game_history,):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_policies, actions = [], [], []
        for current_index in range(config.num_unroll_steps + 1):
            value = UtilTrain.compute_target_value(config,game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0.)
                # Uniform policy
                target_policies.append(
                    [
                        float('-inf')
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(float('-inf'))
                # Uniform policy
                target_policies.append(
                    [
                        float('-inf')
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(config.action_space))
        # return target_values, target_rewards, target_policies, actions
        return target_values, [], target_policies, actions

    @staticmethod
    def collate_fn_decorator(config):
        def collate_fn(samples):
            batch_target_values = []
            batch_target_policies = []
            batch_actions = []
            observation_batch = []
            for     pping in samples:
                target_values, _, target_policies, actions = UtilTrain.make_target(config,mapping)
                batch_actions.append(actions)
                batch_target_policies.append(target_policies)
                batch_target_values.append(target_values)
                observation_batch.append(mapping.observation_history[0])
            return batch_target_values,batch_target_policies,batch_actions,observation_batch
        return collate_fn
    
    @staticmethod
    def collate_fn_decorator_pre_train(config):
        def collate_fn_pre_train(samples):
            batch_states,batch_actions,batch_action_indices,batch_rewards = [],[],[],[]
            batch_values,batch_policy_probs,batch_returns,batch_old_action_probs,batch_old_values = [],[],[],[],[]
            for sample in samples:
                state,action,action_indice,reward,value,policy_prob,ret,old_action_prob,old_value = sample
                batch_states.append(state)
                batch_actions.append(action)
                batch_action_indices.append(action_indice)
                batch_rewards.append(reward)
                batch_values.append(value)
                batch_policy_probs.append(policy_prob)
                batch_returns.append(ret)
                batch_old_action_probs.append(old_action_prob)
                batch_old_values.append(old_value)

            return batch_states,batch_actions,batch_action_indices,batch_rewards,batch_values,batch_policy_probs,batch_returns,batch_old_action_probs,batch_old_values
        return collate_fn_pre_train
    
    @staticmethod
    def loss_pre_train():
        ...
    
    @staticmethod
    def read_replay_buffers(path,config,type_interconnections):
        training_results = TrainingResults(config.model_name.value,config.arch_dims,type_interconnections.value)
        
        dirr = Path(path)
        buffers = []
        for file in (dirr.rglob('*replay_buffer.pkl')):
            if file.is_file():
                replay_buffer = UtilReplayBuffer.get_replay_buffer(file)
                buffers += list(replay_buffer['buffer'].values())
        print("Total mappings:",len(buffers))
        training_results.total_mapping_samples = len(buffers)
        training_results.total_states = numpy.sum([len(buffer.observation_history) for buffer in buffers])
        total_bytes_size = asizeof.asizeof(buffers)
        total_gb_memory = total_bytes_size / (1024**3)
        training_results.gb_memory = total_gb_memory
        training_results.save_csv()
        return buffers