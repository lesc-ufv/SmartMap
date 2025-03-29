class UtilCalculations:
    @staticmethod
    def calc_dist_manhattan(pe_pos1, pe_pos2):
        return abs(pe_pos1[0] - pe_pos2[0]) + abs(pe_pos1[1] - pe_pos2[1]) 