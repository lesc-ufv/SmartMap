
from configs.general_config import GeneralConfig
from src.enums.enum_interconnections import EnumInterconnections
from src.utils.util_interconnections import UtilInterconnections

class GeneralConfigMESH(GeneralConfig):
    def __init__(self, model_name,environment,
                replay_buffer_size,num_simulations,dataset_range,arch_dims,
                class_model,model_instance_args,
                dfg_class_name, cgra_class_name,mode):
        super().__init__(model_name,'MESH',EnumInterconnections.MESH,environment,
                         replay_buffer_size,num_simulations,dataset_range,arch_dims,
                         class_model,model_instance_args,
                         [UtilInterconnections.generate_mesh_interconnection_by_pe_pos],
                         dfg_class_name, cgra_class_name,mode)