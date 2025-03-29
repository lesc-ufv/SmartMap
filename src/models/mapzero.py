from torch_geometric.nn import GATConv
import torch
from src.models.graph_embedding_generation import GraphEmbeddingGeneration
from src.models.mapzero_policy_net import MapZeroPolicyNet
from src.models.value_net import ValueNet
from src.rl.environments.interface_environment import InterfaceEnvironment
from src.models.abstract_network import AbstractNetwork
from torch_geometric.data import Data, Batch
import numpy as np

class MapZero(AbstractNetwork):
    def __init__(self,dim_feat_dfg,dim_feat_cgra,out_dim,n_heads,len_action_space,dtype,environment:InterfaceEnvironment):
        super().__init__()
        negative_slope = 0.02
        self.graph_embed_generation = GraphEmbeddingGeneration(dim_feat_dfg,dim_feat_cgra,out_dim,negative_slope,dtype,n_heads)
        self.dtype = dtype
        self.value_net = ValueNet(out_dim,out_dim,negative_slope,dtype)
        self.policy_net = MapZeroPolicyNet(out_dim,out_dim,len_action_space,negative_slope,dtype)
        self.environment = environment

    def forward(self,dfg_graph,cgra_graph,vertex_to_be_mapped_feature,mask,dfg_pad_mask,cgra_pad_mask):
        state_vector = self.graph_embed_generation(dfg_graph,
                                                    cgra_graph,
                                                    vertex_to_be_mapped_feature,
                                                    dfg_pad_mask,
                                                    cgra_pad_mask)
        p = self.policy_net(state_vector)
        v = self.value_net(state_vector)
        p = torch.where(mask == -torch.inf,-torch.inf,p)
        return p,v

    def initial_inference(self,states):
        batch_dfg = []
        batch_cgra = []
        batch_node_to_be_mapped = []
        batch_mask = []
        batch_pad_mask_dfg = []
        max_dfg_nodes = self.get_max_dfg_nodes(states)
        for state in states:
            dfg_features,dfg_edges_index,cgra_features,cgra_edges_index,vertex_to_be_mapped_feature,mask = state.generate_model_args()
            temp_pad_mask = np.ones((len(dfg_features),))
            while len(dfg_features) < max_dfg_nodes:
                dfg_features.append(np.zeros_like(dfg_features[0]).tolist())
                temp_pad_mask = np.concatenate([temp_pad_mask,np.array([0])])

            dfg_data = Data(x=torch.tensor(dfg_features,dtype=self.dtype), edge_index=torch.tensor(dfg_edges_index,dtype=torch.long))
            batch_dfg.append(dfg_data)

            cgra_data = Data(x= torch.tensor(cgra_features,dtype=self.dtype), edge_index=torch.tensor(cgra_edges_index,dtype=torch.long))
            batch_cgra.append(cgra_data)

            batch_node_to_be_mapped.append(vertex_to_be_mapped_feature)

            batch_mask.append(mask)

            batch_pad_mask_dfg.append(temp_pad_mask.tolist())

        batch_pad_mask_cgra = torch.ones((len(states),batch_cgra[0].num_nodes)).bool()
        batch_mask = torch.tensor(batch_mask,dtype=self.dtype)
        batch_node_to_be_mapped = torch.tensor(batch_node_to_be_mapped,dtype=self.dtype)
        batch_dfg = Batch.from_data_list(batch_dfg)
        batch_cgra = Batch.from_data_list(batch_cgra)
        batch_pad_mask_dfg = torch.tensor(batch_pad_mask_dfg).bool()
        device = next(self.parameters()).device
        
        batch_cgra = batch_cgra.to(device)
        batch_dfg = batch_dfg.to(device)
        batch_node_to_be_mapped = batch_node_to_be_mapped.to(device)
        batch_mask = batch_mask.to(device)
        batch_pad_mask_dfg = batch_pad_mask_dfg.to(device)
        batch_pad_mask_cgra = batch_pad_mask_cgra.to(device)

        p,v = self.forward(batch_dfg,batch_cgra,batch_node_to_be_mapped,batch_mask,batch_pad_mask_dfg,batch_pad_mask_cgra)
        return p,v,torch.torch.zeros_like(v).fill_(0.)

    def recurrent_inference(self,states,actions): 
        batch_dfg,batch_cgra, batch_node_to_be_mapped, batch_mask = [], [], torch.tensor([],dtype=self.dtype),torch.tensor([],dtype=self.dtype)
        next_states, rewards = [], []
        max_dfg_nodes = self.get_max_dfg_nodes(states)
        batch_pad_mask_dfg = []

        for state,action in zip(states,actions):
            next_state,reward,_ = self.environment.step(state,action.item())
            rewards.append(reward)
            next_states.append(next_state)
            dfg_features,dfg_edges_index,cgra_features,cgra_edges_index,vertex_to_be_mapped_feature,mask = next_state.generate_model_args()
            temp_pad_mask = np.ones((len(dfg_features),))

            while len(dfg_features) < max_dfg_nodes:
                dfg_features.append(np.zeros_like(dfg_features[0]).tolist())
                temp_pad_mask = np.concatenate([temp_pad_mask,np.array([0])])

            dfg_data = Data(x=torch.tensor(dfg_features,dtype=self.dtype), edge_index=torch.tensor(dfg_edges_index,dtype=torch.long))
            batch_dfg.append(dfg_data)

            cgra_data = Data(x= torch.tensor(cgra_features,dtype=self.dtype), edge_index=torch.tensor(cgra_edges_index,dtype=torch.long))
            batch_cgra.append(cgra_data)
            batch_node_to_be_mapped = torch.cat((batch_node_to_be_mapped,torch.tensor([vertex_to_be_mapped_feature],dtype=self.dtype)),dim=0)
            batch_mask = torch.cat((batch_mask,torch.tensor([mask],dtype=self.dtype)),dim=0)
            batch_pad_mask_dfg.append(temp_pad_mask.tolist())

        batch_dfg = Batch.from_data_list(batch_dfg)
        batch_pad_mask_cgra = torch.ones((len(states),batch_cgra[0].num_nodes)).bool()
        batch_cgra = Batch.from_data_list(batch_cgra)
        batch_pad_mask_dfg = torch.tensor(batch_pad_mask_dfg).bool()
        
        
        device = next(self.parameters()).device

        batch_cgra = batch_cgra.to(device)
        batch_dfg = batch_dfg.to(device)
        batch_node_to_be_mapped = batch_node_to_be_mapped.to(device)
        batch_mask = batch_mask.to(device)
        batch_pad_mask_cgra = batch_pad_mask_cgra.to(device)
        batch_pad_mask_dfg = batch_pad_mask_dfg.to(device)


        p_next_state,v_next_state = self.forward(batch_dfg,batch_cgra,batch_node_to_be_mapped,batch_mask,batch_pad_mask_dfg,batch_pad_mask_cgra)
        return v_next_state,torch.tensor(rewards,dtype=self.dtype),p_next_state, next_states
    def get_max_dfg_nodes(self,states):
        max_nodes = 0
        for state in states:
            max_nodes = max(state.dfg.len_vertices(),max_nodes)
        return max_nodes