from src.utils.util_calculations import UtilCalculations
class YOTTUtil:
    @staticmethod
    def get_mesh_distance(pe_pos1,pe_pos2):
        return UtilCalculations.calc_dist_manhattan(pe_pos1,pe_pos2)
    @staticmethod
    def get_one_hop_distance(pe_pos1,pe_pos2):
        return round(1/2*YOTTUtil.get_mesh_distance(pe_pos1,pe_pos2))
