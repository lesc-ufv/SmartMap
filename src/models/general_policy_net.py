import torch

class GeneralPolicyNet(torch.nn.Module):
    def __init__(self,dim_input_state,dim_input_actions,dim_out_actions,dim_out,slope_leaky_relu,dtype):
        super(GeneralPolicyNet,self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_input_state+dim_out_actions,dim_out,dtype=dtype),
                                       torch.nn.Linear(dim_out,dim_out,dtype=dtype))
        self.linear = torch.nn.Linear(dim_input_actions,dim_out_actions)                            
        self.fc = torch.nn.Linear(dim_out,1,dtype=dtype)
        self.slope_leaky_relu = slope_leaky_relu

    def forward(self,state_vector,actions):
        #state_vector = (batch,d_out1), actions = (batch,actions,d_out2)
        actions_embed = self.linear(actions)
        x = state_vector.unsqueeze(1).repeat((1,actions.size(1),1))
        x = torch.concat([x,actions_embed],dim=-1)
        x = self.mlp(x)
        x = torch.nn.functional.leaky_relu(x,self.slope_leaky_relu)
        x = self.fc(x)
        x = torch.nn.functional.leaky_relu(x,self.slope_leaky_relu)
        return x.squeeze(-1)
