from src.graphs.dfgs.dfg_mapzero import DFGMapZero
from src.graphs.dfgs.yoto_dfg import YOTODFG
from src.enums.enum_dfg import EnumDFG
from src.graphs.cgras.cgra_mapzero import CGRAMapzero
from src.graphs.cgras.cgra_traversal import CGRATraversal
from src.enums.enum_cgra import EnumCGRA
class UtilInitialize:
    @staticmethod
    def initialize_dfg_from_class_name(class_name,*args):

        if class_name == EnumDFG.MAPZERO_DFG:
            return DFGMapZero(*args)

        if class_name == EnumDFG.YOTO_DFG:
            return YOTODFG(*args)


        raise ValueError(f"EnumDFG {class_name} not implemented.")

    @staticmethod
    def initialize_cgra_from_class_name(class_name,*args):

        if class_name == EnumCGRA.CGRA_MAPZERO:
            return CGRAMapzero(*args)

        if class_name == EnumCGRA.CGRA_TRAVERSAL:
            return CGRATraversal(*args)
        
        raise ValueError(f"EnumCGRA {class_name} not implemented.")
