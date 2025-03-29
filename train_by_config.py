from pathlib import Path 
from src.train_mapping_dataset import TrainMappingDataset
from torch.utils.data.dataloader import DataLoader
from src.utils.util_configs import UtilConfigs
import numpy
from src.enums.enum_interconnections import EnumInterconnections
from src.enums.enum_mode import EnumMode
import torch
import math
from src.graphs.graphs.networkx_graph import NetworkXGraph
from torch.utils.tensorboard import SummaryWriter
from src.utils.util_dfg import UtilDFG
from src.utils.util_initialize import UtilInitialize
from src.utils.util_train import UtilTrain
import sys
from src.utils.util_module import UtilModule
import time
from src.entities.training_results import TrainingResults
import os
from torch.utils.data import Dataset, DataLoader, random_split
from src.enums.enum_model_name import EnumModelName
from torch.distributions import Categorical
import copy
import random
def update_lr(optimizer,initial_lr,decay_rate,cur_training_step,decay_steps):
    lr = initial_lr * decay_rate ** (
            cur_training_step  / decay_steps
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_action_and_action_indice(cur_state,policy_probs,is_mapzero,eval):
    if is_mapzero:
        if not eval:
            dist = Categorical(torch.tensor(policy_probs))
            action = dist.sample()
        else:
            action = numpy.argmax(policy_probs)
        action_indice = action
    else:
        legal_actions = numpy.array(cur_state.get_legal_actions())
        if not eval:
            dist = Categorical(torch.tensor(policy_probs))
            action = legal_actions[dist.sample()]
        else:
            action =  legal_actions[numpy.argmax(policy_probs)]
        index = numpy.argwhere(numpy.array(legal_actions) == action)
        action_indice = index
    return action.item(),action_indice.item()

@torch.no_grad
def generate_mappings(initial_states,environment,model,old_model,is_mapzero,eval = False):
    model.eval()
    mappings = []
    states = []
    actions = []
    rewards = []
    policy_probs = []
    values = []
    all_returns = []
    action_indices = []
    old_policy_probs = []
    old_values = []
    mappings = []
    count = 0
    for i,initial_state in enumerate(initial_states):
        cur_state = initial_state
        mappings.append([])
        is_end_state = False
        reward_sum = 0
        temp_states = []
        temp_rewards = []
        temp_values = []
        while not is_end_state:
            p,v,_= model.initial_inference([cur_state])
            states.append(cur_state)
            temp_states.append(cur_state)
            values.append(v.item())
            temp_values.append(v.item())
            p = torch.nn.functional.softmax(p.squeeze(),dim=0).cpu().detach().numpy()
            if count == 0:
                count+= 1
            policy_probs.append(p.tolist())

            action,action_indice = get_action_and_action_indice(cur_state,p,is_mapzero,eval)

            actions.append(action)
            action_indices.append(action_indice)
            if not eval:
                old_p,old_v,_ = old_model.initial_inference([cur_state])
                old_p = torch.nn.functional.softmax(old_p,dim=-1)
                old_policy_probs.append(old_p.squeeze().tolist())
                old_values.append(old_v.item())
            cur_state, reward,is_end_state = environment.step(cur_state,action)
            rewards.append(reward)
            temp_rewards.append(reward)
            reward_sum += reward
        mappings[i].append([temp_states,temp_rewards,temp_values])
    if not eval:
        rewards = numpy.array(rewards)
        rewards = (rewards - rewards.mean())/ (rewards.std() + 1e-8) 
        rewards= rewards.tolist()
        count = 0
        for i,mapping in enumerate(mappings):
            for k in range(len(mapping[0][2])):
                mapping[0][2][k] = rewards[count] 
                count+=1
            returns = UtilTrain.make_target_pre_train(mapping[0])
            all_returns += returns
    return [states,actions,action_indices,rewards,values,policy_probs,all_returns,old_policy_probs,old_values]

def adjust_kl_penalty(kl_divergence, target_kl, c_kl, kl_factor=2.0):
    if kl_divergence > target_kl * 1.5:
        c_kl *= kl_factor 
    elif kl_divergence < target_kl / 1.5:
        c_kl /= kl_factor
    return c_kl

def loss_function(advantages,target_values,values,old_values,probs,old_probs,action_indices,w_entropy_loss=0.05  ,w_v_loss=1,w_kl = 1,clip_factor = 0.2):
    
    if advantages.size(0) > 1:
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

    dist = Categorical(probs)
    old_dist = Categorical(old_probs)
    cur_log_prob = dist.log_prob(action_indices)
    log_old_probs = old_dist.log_prob(action_indices)
    
    ratio = torch.exp(cur_log_prob - log_old_probs)

    kl_loss = torch.distributions.kl_divergence(old_dist, dist).mean()

    policy_loss_1  = -advantages * ratio

    policy_loss_2 = -advantages * torch.clamp(ratio, 1.0 - clip_factor, 1.0 + clip_factor)

    policy_loss = torch.mean(torch.max(policy_loss_1,policy_loss_2))

    entropy_loss = torch.mean(dist.entropy())

    value_clip = old_values + torch.clamp(values - old_values, -clip_factor, clip_factor)

    value_loss_1 = torch.nn.functional.mse_loss(target_values, values, reduction ='none')
    value_loss_2 = torch.nn.functional.mse_loss(target_values, value_clip,reduction = 'none')
    value_loss = torch.mean(torch.max(value_loss_1, value_loss_2))

    total_loss = policy_loss - entropy_loss *w_entropy_loss + value_loss * w_v_loss + kl_loss*w_kl
    return total_loss,w_v_loss*value_loss.item(), policy_loss.item(), w_entropy_loss*entropy_loss.item(), kl_loss.item()

def train(model_config,batch_size):
    config = model_config.get_config()
    path_to_ckpt_model = model_config.get_path_to_ckpt_model()
    if os.path.exists(path_to_ckpt_model):
        print()
        print(f'Model {config.model_name.value} from config {model_config.__class__.__name__} has already been trained. Please delete or move the file {path_to_ckpt_model} if you want to train again.')
        print()
        return None

    if config.seed:
        generator = torch.manual_seed(config.seed)
        numpy.random.seed(config.seed)


    cgra = model_config.get_cgra()
    environment = config.environment
    arch_dims = config.arch_dims
    num_pes = arch_dims[0]*arch_dims[1]
    collate_fn = UtilTrain.collate_fn_decorator_pre_train(config)
    initial_states = [] 

    path = Path(config.path_to_train_dataset)
    files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
    
    for file in files:
        dfg_graph = NetworkXGraph(file)
        if len(dfg_graph.get_vertices()) + UtilDFG.num_nodes_for_graph_be_balanced(list(dfg_graph.get_vertices()),list(dfg_graph.get_edges())) > num_pes:
            continue
        node_to_operation = UtilDFG.generate_node_to_operation_by_networkx_graph(dfg_graph.graph,"opcode")
        dfg = UtilInitialize.initialize_dfg_from_class_name(config.dfg_class_name,dfg_graph,node_to_operation)
        initial_state = UtilConfigs.get_mapping_state_by_model_name(config.model_name,dfg,cgra,dfg.get_next_node_to_be_mapped(),
                                        config.distance_func)
        initial_states.append(initial_state)
        
    is_mapzero = config.model_name == EnumModelName.MAPZERO
    
    model = config.class_model(*config.model_instance_args)
    model = model.to(config.device)

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.orthogonal_(p,gain=math.sqrt(2))
   
    initial_lr = 0.001
    optimizer = torch.optim.AdamW(model.parameters(),lr=initial_lr)
    old_model = copy.deepcopy(model)
    writer = SummaryWriter(config.results_path)

    print(
        f"\nTraining on {config.device}...\nRun tensorboard --logdir {config.results_path} and go to http://localhost:6006/ to see in real time the training performance.\n"
    )

    hp_table = [
        f"| {key} | {value} |" for key, value in config.__dict__.items()
    ]
    writer.add_text(
        "Hyperparameters",
        "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
    )

    writer.add_text(
        "Model summary",
        str(model).replace("\n", " \n\n"),
    )

    total_train_time = 0
    batch_size = len(initial_states)
    device = config.device
    c_kl = 1
    kl_loss = 0

    decay_step = 1
    decay_rate = 0.99
    iters = 3 
    for epoch in range(config.epochs):
        temp_mappings = generate_mappings(initial_states,environment,model,old_model,is_mapzero,eval=False)

        train_dataset = TrainMappingDataset(temp_mappings)
        train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,num_workers=4,collate_fn=collate_fn)

        init_train_time = time.time()
        epoch_value_loss = 0
        epoch_policy_loss = 0
        epoch_entropy_loss = 0
        epoch_kl_loss = 0
        model.train()
        w_entropy_loss = max(0.01,1 - epoch / config.epochs - 0.9)
        for k in range(iters):
            for (states,actions,action_indices,rewards,batch_values,policy_probs,returns,old_probs,old_values) in train_dataloader:
                optimizer.zero_grad()
                returns = torch.tensor(returns).to(device)
                advantages = returns - torch.tensor(batch_values).to(device)
                old_values = torch.tensor(old_values).to(device)
                old_probs = torch.tensor(old_probs).to(device)
                action_indices = torch.tensor(action_indices).long().to(device)
            
                policy_logits,values,_ = model.initial_inference(
                    states
                )
                policy_probs = torch.nn.functional.softmax(policy_logits,dim=-1)

                c_kl = adjust_kl_penalty(kl_loss, target_kl=0.01, c_kl=c_kl)
                c_kl = max(min(c_kl, 10), 0.01)
                total_loss,value_loss, policy_loss, entropy_loss,kl_loss = loss_function(advantages,returns,values.squeeze(-1),old_values,policy_probs.squeeze(0),old_probs,action_indices,w_entropy_loss,1,c_kl,0.2)
                total_loss.backward()   
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
                optimizer.step()
                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
                epoch_entropy_loss += entropy_loss
                epoch_kl_loss += kl_loss

        update_lr(optimizer,initial_lr,decay_rate,epoch,decay_step)
        old_model = copy.deepcopy(model)

            
        temp_rewards = generate_mappings(initial_states,environment,model,None,is_mapzero,eval=True)[3]
        epoch_mean_rewards = numpy.array(temp_rewards).mean()
        
        epoch_policy_loss/=(len(train_dataloader)*iters)
        epoch_value_loss/=(len(train_dataloader)*iters)
        epoch_entropy_loss/=(len(train_dataloader)*iters)
        epoch_kl_loss/=(len(train_dataloader)*iters)
        epoch_total_loss =  epoch_policy_loss + epoch_value_loss + epoch_entropy_loss

        lr = optimizer.param_groups[0]['lr']
        total_loss = epoch_policy_loss + epoch_value_loss

        final_train_time = time.time()
        total_train_time += final_train_time - init_train_time
        writer.add_scalar("1.Loss/1.Total_Loss",total_loss,epoch)
        writer.add_scalar("1.Loss/2.Policy_Loss",epoch_policy_loss,epoch)
        writer.add_scalar("1.Loss/3.Value_Loss",epoch_value_loss,epoch)
        writer.add_scalar("1.Loss/3.Epoch_Entropy_Loss",epoch_entropy_loss,epoch)
        writer.add_scalar("1.Loss/3.Epoch_KL_Loss",epoch_kl_loss,epoch)

        writer.add_scalar("2.Eval/1.Mean_Reward",epoch_mean_rewards,epoch)
        writer.add_scalar("3.Control/1.Learning_Rate",lr,epoch)
        print(f'Epoch: {epoch + 1}. Total Loss: {epoch_total_loss}. Value Loss: {epoch_value_loss:.5f}. Policy Loss: {epoch_policy_loss:.5f}. Entropy Loss: {epoch_entropy_loss:.3f}. KL Loss: {epoch_kl_loss:.3f} | Eval: Mean Reward: {epoch_mean_rewards:.4f} | lr: {lr:.6f}. Time: {final_train_time-init_train_time:.3f}')

    torch.save(model.state_dict(),path_to_ckpt_model)
    print()
    print(f'Model saved in {path_to_ckpt_model}')
    print()
    train_results = TrainingResults(config.model_name.value,config.arch_dims,model_config.type_interconnnections.value)
    train_results.training_time = total_train_time
    train_results.save_csv()

if __name__ == "__main__":
    args = sys.argv
    config_module = args[1]
    config_class = args[2]
    interconnection_style = args[3]
    interconnection_style = EnumInterconnections(interconnection_style)
    arch_dims = tuple(map(int,args[4].split('x')))
    mode = EnumMode(args[5])
    model_config = UtilModule.load_class_from_file(config_module,config_class,interconnection_style,arch_dims,mode)
    batch_size = 64    
    train(model_config,batch_size)