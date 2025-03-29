from torch_geometric.nn import GATConv
import torch
class GraphEmbeddingGeneration(torch.nn.Module):
    def __init__(self, dim_feat_dfg,dim_feat_cgra, dim_out,
                slope_leaky_relu, dtype,n_heads):
        super(GraphEmbeddingGeneration,self).__init__()
        self.dim_out = dim_out
        self.gat_dfg = GATConv(in_channels=dim_feat_dfg,out_channels=dim_out,heads=n_heads,concat=False) 
        self.gat_cgra = GATConv(in_channels=dim_feat_cgra,out_channels=dim_out,heads=n_heads,concat=False) 
        self.fc_metadata = torch.nn.Linear(dim_feat_dfg,dim_out,dtype=dtype)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3*dim_out,dim_out,dtype=dtype),
                                       torch.nn.Linear(dim_out,dim_out,dtype=dtype))
        self.slope_leaky_relu = slope_leaky_relu 

    def forward(self,dfg_graph,cgra_graph,vertex_to_be_mapped_feature,dfg_padding_mask,cgra_padding_mask):
        batch_size_dfg = dfg_graph.num_graphs
        
        batch_size_cgra = cgra_graph.num_graphs
        
        #(all_nodes,features)
        embeddings_dfg = self.gat_dfg(dfg_graph.x,edge_index = dfg_graph.edge_index)

        final_embeddings_dfg = torch.zeros(batch_size_dfg,embeddings_dfg.size(-1)).to((next(self.parameters()).device))

        #(batch,nodes,features)
        embeddings_dfg = torch.reshape(embeddings_dfg,(batch_size_dfg,-1,self.dim_out))
        for i in range(batch_size_dfg):
            mask = dfg_padding_mask[i]
            masked_embedd = embeddings_dfg[i][mask]
            final_embeddings_dfg[i] = masked_embedd.mean(dim=-2)

        #(batch,features)
        # graph_embeddings_dfg = embeddings_dfg.mean(dim=-2)

        #(all_nodes,features)
        embeddings_cgra = self.gat_cgra(cgra_graph.x, edge_index = cgra_graph.edge_index)
        
        #(batch,nodes,features)
        embeddings_cgra = torch.reshape(embeddings_cgra,(batch_size_cgra,-1,self.dim_out))
        
        final_embeddings_cgra = torch.zeros(batch_size_cgra,embeddings_cgra.size(-1)).to((next(self.parameters()).device))
        for i in range(batch_size_cgra):
            mask = cgra_padding_mask[i]
            masked_embedd = embeddings_cgra[i][mask]
            final_embeddings_cgra[i] = masked_embedd.mean(dim=-2)

        #(batch,features)
        # graph_embeddings_cgra = embeddings_cgra.mean(dim=-2)
        
        #(batch,features)
        embedding_vertex_to_be_mapped = self.fc_metadata(vertex_to_be_mapped_feature)
        
        #(batch,3*features)
        state_vector = torch.concatenate([final_embeddings_dfg,final_embeddings_cgra,embedding_vertex_to_be_mapped],dim=-1)
        
        #(batch,3*features)
        final_embedding = self.mlp(state_vector)

        return torch.nn.functional.leaky_relu(final_embedding,self.slope_leaky_relu)

