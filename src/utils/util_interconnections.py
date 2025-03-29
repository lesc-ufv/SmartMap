from src.utils.util_cgra import UtilCGRA
class UtilInterconnections:
    """
    Class to generate interconnection styles for CGRAs.
    """
    @staticmethod
    def generate_interconnection_by_adj_positions_and_pe_pos(adj_positions,pe_pos,dim_cgra):
        """
        Generates connections for a PE at a given position (pe_pos) based on a pattern of relative connections (adj_positions).

        Args:
            adj_positions (list[tuple[int, int]]): Relative positions to be added to pe_pos.
            pe_pos (tuple[int, int]): Position of the PE.
            dim_cgra (tuple[int, int]): Dimensions of the CGRA.

        Returns:
            list: Interconnections of the PE at the position 'pe_pos'.

        Tested:
            True
        """

        return [(pe_pos[0]+adj_position[0],pe_pos[1]+adj_position[1])  for adj_position in adj_positions if \
                not UtilCGRA.pe_pos_is_out_of_border([pe_pos[0]+adj_position[0],pe_pos[1]+adj_position[1]],dim_cgra)]
    @staticmethod
    def generate_mesh_interconnection_by_pe_pos(pe_pos,dim_cgra):
        """
        Generates a mesh connections for a PE at a given position (pe_pos).

        Args:
            pe_pos (tuple[int, int]): Position of the PE.
            dim_cgra (tuple[int, int]): Dimensions of the CGRA.

        Returns:
            list: Mesh interconnections of the PE at the position 'pe_pos'.

        Tested:
            True
        """
        adj_positions = [(-1,0),(1,0),(0,1),(0,-1)]
        return UtilInterconnections.generate_interconnection_by_adj_positions_and_pe_pos(adj_positions,pe_pos,dim_cgra)
        
    @staticmethod
    def generate_diagonal_interconnection_by_pe_pos(pe_pos,dim_cgra):
        """
        Generates a diagonal connections for a PE at a given position (pe_pos).

        Args:
            pe_pos (tuple[int, int]): Position of the PE.
            dim_cgra (tuple[int, int]): Dimensions of the CGRA.

        Returns:
            list: Diagonal interconnections of the PE at the position 'pe_pos'.

        Tested:
            True
        """
        
        adj_positions = [(-1,-1),(-1,1),(1,-1),(1,1)]
        return UtilInterconnections.generate_interconnection_by_adj_positions_and_pe_pos(adj_positions,pe_pos,dim_cgra)
    @staticmethod
    def generate_one_hop_interconnection_by_pe_pos(pe_pos,dim_cgra):
        """
        Generates a one-hop connections for a PE at a given position (pe_pos).

        Args:
            pe_pos (tuple[int, int]): Position of the PE.
            dim_cgra (tuple[int, int]): Dimensions of the CGRA.

        Returns:
            list: One-hop interconnections of the PE at the position 'pe_pos'.

        Tested:
            True
        """
        
        adj_positions = [(-2,0),(2,0),(0,2),(0,-2)]
        return UtilInterconnections.generate_interconnection_by_adj_positions_and_pe_pos(adj_positions,pe_pos,dim_cgra) + \
            UtilInterconnections.generate_mesh_interconnection_by_pe_pos(pe_pos,dim_cgra)
    
    @staticmethod
    def generate_toroidal_interconnection_by_pe_pos(pe_pos,dim_cgra):
        """
        Generates a toroidal connections for a PE at a given position (pe_pos).

        Args:
            pe_pos (tuple[int, int]): Position of the PE.
            dim_cgra (tuple[int, int]): Dimensions of the CGRA.

        Returns:
            list: Toroidal interconnections of the PE at the position 'pe_pos'.

        Tested:
            True
        """
        if UtilCGRA.pe_pos_is_border(pe_pos,dim_cgra):
            adj_positions = [(0,dim_cgra[1]-1),(0,-(dim_cgra[1]-1)),(dim_cgra[0]-1,0),(-(dim_cgra[0]-1),0)] 
            return UtilInterconnections.generate_interconnection_by_adj_positions_and_pe_pos(adj_positions,pe_pos,dim_cgra)
        return []