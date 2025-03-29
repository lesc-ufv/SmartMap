from abc import ABC,abstractmethod
class ConfigInterface:
    @abstractmethod
    def __init__(self,type_arch,arch_dims,num_vertices,mode):
        ...
    
    @abstractmethod
    def get_config(self):
        ...
    
    @abstractmethod
    def get_cgra(self):
        ...
    
    @abstractmethod
    def get_type_interconnnections(self):
        ...
    @abstractmethod
    def get_path_to_ckpt_model(self):
        ...