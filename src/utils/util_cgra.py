class UtilCGRA:
    @staticmethod
    def pe_pos_is_out_of_border(pe_pos:tuple[int],dim_cgra:tuple[int]):
        """
        Verifies if a position is out of CGRA dimensions.
        Args:
            pe_pos (tuple[int]): 2-dimensional coordinates.
            dim_cgra (tuple[int]): Dimensions of the CGRA (nxm), where n is the number of rows
                                   and m is the number of columns.
        Returns:
            bool: True if the position is out of the CGRA border, False otherwise.
        Tested:
            False
        """
        pe_i,pe_j = pe_pos
        i,j = dim_cgra
        return pe_i < 0 or pe_i >=i or pe_j < 0 or pe_j >= j
    @staticmethod
    def pe_pos_is_border(pe_pos,dim_cgra):
        """
        Verifies if a position is on the CGRA border.
        Args:
            pe_pos (tuple[int]): 2-dimensional coordinates.
            dim_cgra (tuple[int]): Dimensions of the CGRA (nxm), where n is the number of rows
                                   and m is the number of columns.
        Returns:
            bool: True if the position is on the CGRA border, False otherwise.
        Tested:
            False
        """
        return pe_pos[0] == (dim_cgra[0] - 1) or pe_pos[1] == (dim_cgra[1] - 1) or pe_pos[0] == 0 or pe_pos[1] == 0
    @staticmethod
    def pe_pos_is_corner(pe_pos,dim_cgra):
        """
        Verifies if a position is in the corner of the CGRA.
        Args:
            pe_pos (tuple[int]): 2-dimensional coordinates.
            dim_cgra (tuple[int]): Dimension of the CGRA (nxm) where n is the number of rows
            and m the number of columns.
        Returns:
            bool: True if position is on the CGRA corner, False otherwise.
        Tested:
            False
        """
        i,j = pe_pos
        return (i == 0 and j == 0) or (i == 0 and j == dim_cgra[1] - 1) or (i == dim_cgra[0]-1 and j == 0)  or (i== (dim_cgra[0] - 1) and j == dim_cgra[1]-1)
