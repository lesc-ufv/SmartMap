import torch

class ValueNet(torch.nn.Module):
    def __init__(self,dim_input,dim_output,leaky_relu_slope,dtype):
        super(ValueNet,self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_input,dim_output,dtype=dtype),
            torch.nn.Linear(dim_output,dim_output,dtype=dtype)
            )
        self.fc = torch.nn.Linear(dim_output,1,dtype=dtype)
        self.slope_leaky_relu = leaky_relu_slope

    def forward(self,state_vector):
        x = self.mlp(state_vector)
        x = torch.nn.functional.leaky_relu(x,self.slope_leaky_relu)
        x = self.fc(x)
        return torch.nn.functional.leaky_relu(x,self.slope_leaky_relu)
