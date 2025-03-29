import torch
from abc import ABC, abstractmethod
from src.utils.util_pytorch import UtilPytorch
class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, state):
        pass

    @abstractmethod
    def recurrent_inference(self, state, action):
        pass

    def get_weights(self):
        return UtilPytorch.dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

