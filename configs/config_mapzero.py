from configs.general_config_model import GeneralConfigModel
from src.rl.environments.mapzero_environment import MapZeroEnvironment
from src.graphs.dfgs.dfg_mapzero import DFGMapZero
from src.graphs.cgras.cgra_mapzero import CGRAMapzero
from src.models.mapzero import MapZero
import torch
from src.enums.enum_dfg import EnumDFG
from src.enums.enum_model_name import EnumModelName
from src.enums.enum_cgra import EnumCGRA

class ConfigMapzero(GeneralConfigModel):
    def __init__(self,type_interconnections,arch_dims,mode):
        model_name = EnumModelName.MAPZERO
        environment = MapZeroEnvironment()
        model_instance_args = [10,7,32,4,arch_dims[0]*arch_dims[1],torch.float32,environment]
        dfg_class_name = EnumDFG.MAPZERO_DFG
        cgra_class_name = EnumCGRA.CGRA_MAPZERO
        super().__init__(type_interconnections,arch_dims,model_name,environment,MapZero,
                            model_instance_args,dfg_class_name,cgra_class_name,mode)