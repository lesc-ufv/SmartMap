import torch
class MapZeroPolicyNet(torch.nn.Module):
    def __init__(self,dim_input,dim_out,size_space_action,slope_leaky_relu,dtype):
        super(MapZeroPolicyNet,self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_input,dim_out,dtype=dtype),
                                       torch.nn.Linear(dim_out,dim_out,dtype=dtype))
        self.fc = torch.nn.Linear(dim_out,size_space_action,dtype=dtype)
        self.slope_leaky_relu = slope_leaky_relu
 
    def forward(self,state_vector):
        x = self.mlp(state_vector)
        x = torch.nn.functional.leaky_relu(x,self.slope_leaky_relu)
        x = self.fc(x)
        x = torch.nn.functional.leaky_relu(x,self.slope_leaky_relu)
        return x
if __name__ == "__main__":
    model = MapZeroPolicyNet(32,32,16,0.4,torch.float)
    state_representation = torch.rand((32),dtype=torch.float)
    assert model(state_representation).shape[0] == 16

