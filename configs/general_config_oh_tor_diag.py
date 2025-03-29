from src.config import Config
import torch
from src.utils.softmax_temperature import SoftmaxTemperature
from configs.general_config import GeneralConfig
from src.enums.enum_interconnections import EnumInterconnections
from src.utils.util_interconnections import UtilInterconnections
class GeneralConfig_OH_TOR_DIAG(GeneralConfig):
    def __init__(self, model_name,environment,
                replay_buffer_size,num_simulations,dataset_range,arch_dims,
               class_model,model_instance_args,
                dfg_class_name,cgra_class_name,mode):
        super().__init__(model_name,'OH_TOR_DIAG',EnumInterconnections.OH_TOR_DIAG,environment,
                         replay_buffer_size,num_simulations,dataset_range,arch_dims,
                      class_model,model_instance_args,
                         [UtilInterconnections.generate_one_hop_interconnection_by_pe_pos,
                         UtilInterconnections.generate_toroidal_interconnection_by_pe_pos,
                         UtilInterconnections.generate_diagonal_interconnection_by_pe_pos],
                         dfg_class_name,cgra_class_name,mode)