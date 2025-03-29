from torch_geometric.nn import GATConv
import torch
from src.rl.states.mapping_state_interface import MappingStateInterface
from src.models.graph_embedding_generation import GraphEmbeddingGeneration
from src.models.general_policy_net import GeneralPolicyNet
from src.models.value_net import ValueNet
from src.rl.environments.interface_environment import InterfaceEnvironment
from src.models.abstract_network import AbstractNetwork
from torch_geometric.data import Data, Batch
import numpy as np

class SmartMap(AbstractNetwork):
    def __init__(self,dim_dfg_feat,dim_cgra_feat,out_dim,max_actions,dtype,environment:InterfaceEnvironment):
        super().__init__()
        negative_slope = 0.02
        self.graph_embed_generation = GraphEmbeddingGeneration(dim_dfg_feat,dim_cgra_feat,out_dim,negative_slope,dtype,4)
        self.dtype = dtype
        self.value_net = ValueNet(out_dim,out_dim,negative_slope,dtype)
        self.policy_net = GeneralPolicyNet(out_dim,dim_cgra_feat,out_dim,out_dim,negative_slope,dtype)
        self.environment = environment
        self.dim_features = dim_cgra_feat
        self.max_actions = max_actions

    def forward(self,dfg_graph,cgra_graph,vertex_to_be_mapped_feature,actions,pad_mask_actions,pad_dfg_mask,pad_cgra_mask):
        state_vector = self.graph_embed_generation(dfg_graph,
                                                    cgra_graph,
                                                    vertex_to_be_mapped_feature,
                                                    pad_dfg_mask,
                                                    pad_cgra_mask)
        p = self.policy_net(state_vector,actions)
        v = self.value_net(state_vector)
        p = torch.where(pad_mask_actions,-torch.inf,p)
        return p,v

    def initial_inference(self,states):
        max_legal_actions = self.get_max_legal_actions(states)
        bsz = len(states)
        if max_legal_actions == 0:
            return torch.zeros((bsz,self.max_actions)).fill_(-torch.inf),torch.tensor,torch.zeros((bsz,1)),torch.zeros((bsz,1))
        
        max_legal_actions = self.max_actions
        batch_dfg = []
        batch_cgra = []
        batch_node_to_be_mapped = []
        batch_pad_mask_dfg = []
        batch_pad_mask_cgra = []
        batch_pad_mask_actions = []

        batch_action_features = []
        max_dfg_nodes = self.get_max_dfg_nodes(states)
        max_cgra_nodes = self.get_max_cgra_nodes(states)
        for state in states:
            dfg_features,dfg_edges_index,cgra_features,cgra_edges_index,vertex_to_be_mapped_feature,action_features = state.generate_model_args()
            temp_pad_mask = np.ones((len(dfg_features),))
            while len(dfg_features) < max_dfg_nodes:
                dfg_features.append(np.zeros_like(dfg_features[0]).tolist())
                temp_pad_mask = np.concatenate([temp_pad_mask,np.array([0])])

            temp_pad_mask_actions = np.zeros((len(action_features),))
            while len(action_features) < max_legal_actions:
                action_features.append(np.zeros_like(action_features[0]).tolist())
                temp_pad_mask_actions = np.concatenate([temp_pad_mask_actions,np.array([1])])
            
            temp_pad_mask_cgra = np.ones((len(cgra_features),))
            while len(cgra_features) < max_cgra_nodes:
                cgra_features.append(np.zeros_like(cgra_features[0]).tolist())
                temp_pad_mask_cgra = np.concatenate([temp_pad_mask_cgra,np.array([0])])
            
            dfg_data = Data(x=torch.tensor(dfg_features,dtype=self.dtype), edge_index=torch.tensor(dfg_edges_index,dtype=torch.long))
            batch_dfg.append(dfg_data)

            cgra_data = Data(x= torch.tensor(cgra_features,dtype=self.dtype), edge_index=torch.tensor(cgra_edges_index,dtype=torch.long))
            batch_cgra.append(cgra_data)

            batch_node_to_be_mapped.append(vertex_to_be_mapped_feature)

            batch_action_features.append(action_features)
            batch_pad_mask_dfg.append(temp_pad_mask.tolist())
            batch_pad_mask_cgra.append(temp_pad_mask_cgra.tolist())
            batch_pad_mask_actions.append(temp_pad_mask_actions.tolist())

        batch_action_features = torch.tensor(batch_action_features,dtype=self.dtype)
        batch_node_to_be_mapped = torch.tensor(batch_node_to_be_mapped,dtype=self.dtype)
        batch_dfg = Batch.from_data_list(batch_dfg)
        batch_cgra = Batch.from_data_list(batch_cgra)
        batch_pad_mask_dfg = torch.tensor(batch_pad_mask_dfg).bool()
        batch_pad_mask_cgra = torch.tensor(batch_pad_mask_cgra).bool()
        batch_pad_mask_actions = torch.tensor(batch_pad_mask_actions).bool()

        device = next(self.parameters()).device

        batch_cgra = batch_cgra.to(device)
        batch_dfg = batch_dfg.to(device)
        batch_node_to_be_mapped = batch_node_to_be_mapped.to(device)
        batch_action_features = batch_action_features.to(device)
        batch_pad_mask_dfg = batch_pad_mask_dfg.to(device)
        batch_pad_mask_cgra = batch_pad_mask_cgra.to(device)
        batch_pad_mask_actions = batch_pad_mask_actions.to(device)

        p,v = self.forward(batch_dfg,batch_cgra,batch_node_to_be_mapped,batch_action_features,batch_pad_mask_actions,batch_pad_mask_dfg,batch_pad_mask_cgra)
        return p,v,torch.torch.zeros_like(v).fill_(0.)

    def recurrent_inference(self,states,actions): 
        bsz = len(states)
        max_legal_actions = self.get_max_legal_actions(states)
        if max_legal_actions == 0:
            return torch.zeros((bsz,1)),torch.zeros((bsz,1)), torch.zeros((bsz,self.max_actions)).fill_(-torch.inf),states
        max_legal_actions = self.max_actions

        batch_dfg,batch_cgra, batch_node_to_be_mapped, batch_mask = [], [], torch.tensor([],dtype=self.dtype),torch.tensor([],dtype=self.dtype)
        next_states, rewards = [], []
        batch_pad_mask_dfg = [] 
        batch_pad_mask_cgra = []
        batch_pad_mask_actions = []
        batch_action_features = []
        max_dfg_nodes = self.get_max_dfg_nodes(states)
        max_cgra_nodes = self.get_max_cgra_nodes(states)
        for state,action in zip(states,actions):
            next_state,reward,_ = self.environment.step(state,action.item())
            rewards.append(reward)
            next_states.append(next_state)
            dfg_features,dfg_edges_index,cgra_features,cgra_edges_index,vertex_to_be_mapped_feature,action_features = next_state.generate_model_args()
            
            temp_pad_mask = np.ones((len(dfg_features),))
            while len(dfg_features) < max_dfg_nodes:
                dfg_features.append(np.zeros_like(dfg_features[0]).tolist())
                temp_pad_mask = np.concatenate([temp_pad_mask,np.array([0])])

            temp_pad_mask_actions = np.zeros((len(action_features),))
            while len(action_features) < max_legal_actions:
                action_features.append(np.zeros((self.dim_features,)).tolist())
                temp_pad_mask_actions = np.concatenate([temp_pad_mask_actions,np.array([1])])
            
            temp_pad_mask_cgra = np.ones((len(cgra_features),))
            while len(cgra_features) < max_cgra_nodes:
                cgra_features.append(np.zeros_like(cgra_features[0]).tolist())
                temp_pad_mask_cgra = np.concatenate([temp_pad_mask_cgra,np.array([0])])
            
            dfg_data = Data(x=torch.tensor(dfg_features,dtype=self.dtype), edge_index=torch.tensor(dfg_edges_index,dtype=torch.long))
            batch_dfg.append(dfg_data)

            cgra_data = Data(x= torch.tensor(cgra_features,dtype=self.dtype), edge_index=torch.tensor(cgra_edges_index,dtype=torch.long))
            batch_cgra.append(cgra_data)
            batch_node_to_be_mapped = torch.cat((batch_node_to_be_mapped,torch.tensor([vertex_to_be_mapped_feature],dtype=self.dtype)),dim=0)

            batch_action_features.append(action_features)
            batch_pad_mask_dfg.append(temp_pad_mask.tolist())
            batch_pad_mask_cgra.append(temp_pad_mask_cgra.tolist())
            batch_pad_mask_actions.append(temp_pad_mask_actions.tolist())

        batch_dfg = Batch.from_data_list(batch_dfg)
        batch_cgra = Batch.from_data_list(batch_cgra)
        batch_action_features = torch.tensor(batch_action_features,dtype=self.dtype)
        batch_pad_mask_dfg = torch.tensor(batch_pad_mask_dfg).bool()
        batch_pad_mask_cgra = torch.tensor(batch_pad_mask_cgra).bool()
        batch_pad_mask_actions = torch.tensor(batch_pad_mask_actions).bool()
        device = next(self.parameters()).device
        
        batch_cgra = batch_cgra.to(device)
        batch_dfg = batch_dfg.to(device)
        batch_node_to_be_mapped = batch_node_to_be_mapped.to(device)
        batch_mask = batch_mask.to(device)
        
        batch_action_features = batch_action_features.to(device)
        batch_pad_mask_dfg = batch_pad_mask_dfg.to(device)
        batch_pad_mask_cgra = batch_pad_mask_cgra.to(device)
        batch_pad_mask_actions = batch_pad_mask_actions.to(device)

        p_next_state,v_next_state = self.forward(batch_dfg,batch_cgra,batch_node_to_be_mapped,batch_action_features,batch_pad_mask_actions,batch_pad_mask_dfg,batch_pad_mask_cgra)
        return v_next_state,torch.tensor(rewards,dtype=self.dtype),p_next_state, next_states
    
    def get_max_dfg_nodes(self,states):
        max_nodes = 0
        for state in states:
            max_nodes = max(state.dfg.len_vertices(),max_nodes)
        return max_nodes
    
    def get_max_cgra_nodes(self,states):
        max_nodes = 0
        for state in states:
            max_nodes = max(state.cgra.len_vertices(),max_nodes)
        return max_nodes

    def get_max_legal_actions(self,states):
        max_len_actions = 0
        for state in states:
            max_len_actions = max(len(state.legal_actions),max_len_actions)
        return max_len_actions

  
  
