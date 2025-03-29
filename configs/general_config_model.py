from configs.config_interface import ConfigInterface
from src.graphs.graphs.networkx_graph import  NetworkXGraph
from src.utils.util_configs import UtilConfigs
from src.enums.enum_functionalitie_pe import EnumFunctionalitiePE
from src.utils.util_initialize import UtilInitialize
import os
class GeneralConfigModel(ConfigInterface):
    def __init__(self,type_interconnections,arch_dims,model_name,
                environment,class_model,model_instance_args,
                dfg_class_name,cgra_class_name, mode):
        num_simulations = UtilConfigs.get_num_simulations_by_type_interconnections_and_arch_dims(type_interconnections,arch_dims)
        dataset_range = UtilConfigs.get_dataset_range_by_arch_dims(arch_dims)
        replay_buffer_size = UtilConfigs.get_replay_buffer_size_by_type_interconnections_and_arch_dims(type_interconnections,arch_dims)

        config_args = [model_name,environment,replay_buffer_size, num_simulations,dataset_range,arch_dims,
                       class_model,model_instance_args,dfg_class_name,cgra_class_name,mode]
        config_class = UtilConfigs.get_arch_config_class_by_type_interconnections(type_interconnections)
        self.config = config_class(*config_args)
        self.type_interconnnections = type_interconnections
        cgra_graph = NetworkXGraph()
        all_functionalities = [EnumFunctionalitiePE.ARITHMETIC,EnumFunctionalitiePE.LOGICAL,EnumFunctionalitiePE.MEMORY_ACCESS]
        pe_to_functionalities = {}
        arch_name = f'{arch_dims[0]}x{arch_dims[1]}_{self.config.arch_name}'
        for i in range(arch_dims[0]*arch_dims[1]):
            pe_to_functionalities[i] = all_functionalities
        self.cgra =  UtilInitialize.initialize_cgra_from_class_name(self.config.cgra_class_name,cgra_graph,arch_dims,
                                                        pe_to_functionalities,self.config.interconnection_style,arch_name,*self.config.interconnections)
        path_to_ckpts = 'results/checkpoints/'
        os.makedirs(path_to_ckpts,exist_ok=True)
        self.path_to_ckpt_model = f'{path_to_ckpts}{self.config.model_name.value}_{self.type_interconnnections.value}_{self.config.arch_dims[0]}x{self.config.arch_dims[1]}.pth'

    def get_config(self):
        return self.config

    def get_cgra(self):
        return self.cgra
    def get_type_interconnnections(self):
        return self.type_interconnnections
    def get_path_to_ckpt_model(self):
        return self.path_to_ckpt_model