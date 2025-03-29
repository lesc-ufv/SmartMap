from configs.general_config_model import GeneralConfigModel
from src.rl.environments.smartmap_environment import SmartMapEnvironment
from src.models.smartmap import SmartMap
import torch
from src.enums.enum_dfg import EnumDFG
from src.enums.enum_model_name import EnumModelName
from src.enums.enum_cgra import EnumCGRA

class ConfigSmartMap(GeneralConfigModel):
    def __init__(self,type_interconnections,arch_dims,mode):
        model_name = EnumModelName.SMARTMAP
        environment = SmartMapEnvironment()
        model_instance_args = [9,7,32,arch_dims[0]*arch_dims[1],torch.float32,environment]
        dfg_class_name = EnumDFG.YOTO_DFG
        cgra_class_name = EnumCGRA.CGRA_TRAVERSAL
        super().__init__(type_interconnections,arch_dims,model_name,environment,SmartMap,
                            model_instance_args,dfg_class_name,cgra_class_name,mode)
