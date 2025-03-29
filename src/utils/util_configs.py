from src.enums.enum_interconnections import EnumInterconnections
from configs.general_config_mesh import GeneralConfigMESH
from configs.general_config_one_hop import GeneralConfig_ONE_HOP
from configs.general_config_oh_tor_diag import GeneralConfig_OH_TOR_DIAG
from src.utils.yott_util import YOTTUtil
from src.rl.states.mapping_state_mapzero import MappingStateMapZero
from src.rl.states.mapping_state_yoto import MappingStateYOTO

from src.enums.enum_model_name import EnumModelName
class UtilConfigs:

    @staticmethod
    def get_training_steps(num_vertices,type_interconnections,arch_dims):
        num_vertices = int(num_vertices)
        if type_interconnections == EnumInterconnections.MESH:
            start = 400
        elif type_interconnections == EnumInterconnections.ONE_HOP:
            start = 350
        elif type_interconnections == EnumInterconnections.OH_TOR_DIAG:
            start = 300
        
        n = num_vertices - 3
        
        return min((start + (n-1)*25),500)

    @staticmethod
    def get_replay_buffer_size_by_type_interconnections_and_arch_dims(type_interconnections,arch_dims):
        return 64
    
    @staticmethod
    def get_num_simulations_by_type_interconnections_and_arch_dims(type_interconnections,arch_dims):
        if type_interconnections == EnumInterconnections.MESH:
            start = 200
        elif type_interconnections == EnumInterconnections.ONE_HOP:
            start =  150
        elif type_interconnections == EnumInterconnections.OH_TOR_DIAG:
            start =  100 
            
        if arch_dims == (4,4):
            n = 1
        elif arch_dims == (8,8):
            n = 2
        elif arch_dims == (10,10) or arch_dims == (16,16):
            n = 3
        return start + (n-1)*50

    @staticmethod
    def get_dataset_range_by_arch_dims(arch_dims):
        if arch_dims == (4,4):
            return range(4,17)
        if arch_dims == (8,8) or arch_dims == (10,10) or  arch_dims == (16,16):
            return range(4,31)
    @staticmethod
    def get_arch_config_class_by_type_interconnections(type_interconnections):
        if type_interconnections == EnumInterconnections.MESH:
            return GeneralConfigMESH
        if type_interconnections == EnumInterconnections.ONE_HOP:
            return GeneralConfig_ONE_HOP
        if type_interconnections == EnumInterconnections.OH_TOR_DIAG:
            return GeneralConfig_OH_TOR_DIAG
        raise ValueError(f'Config class not implemented for {type_interconnections}')
    @staticmethod
    def get_distance_func_by_type_interconnections(type_interconnections):
        if type_interconnections == EnumInterconnections.MESH:
            return YOTTUtil.get_mesh_distance
        if type_interconnections == EnumInterconnections.ONE_HOP:
            return YOTTUtil.get_one_hop_distance
        raise ValueError(f'Distance func not implemented for {type_interconnections}')
    
    @staticmethod
    def get_mapping_state_by_model_name(model_name,dfg,cgra,id_node_to_be_mapped,dist_func = None):
        if model_name == EnumModelName.MAPZERO:
            return MappingStateMapZero(dfg,cgra,id_node_to_be_mapped)
        if model_name == EnumModelName.SMARTMAP:
            return MappingStateYOTO(dfg,cgra,id_node_to_be_mapped)
        if model_name == EnumModelName.YOTT_SMARTMAP:
            return MappingStateYOTT(dfg,cgra,id_node_to_be_mapped,dist_func)
        raise ValueError(f'MappingState class not implemented for {model_name}')

