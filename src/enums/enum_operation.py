from enum import Enum


class EnumOperation(Enum):
    """
    Enum used to identify operations in DFGs.
    """
    ADD = ("add",0)
    MULT = ("mul",1)
    CONST = ("const",2)
    LOAD = ("load",3)
    OUTPUT = ("output",4)
    STORE = ('store',5)

    @staticmethod
    def enum_operation_by_type(type_str):
        """
            Get a EnumOperation class by type of the operation.
            Args:
                type_str (str): Type of the operation.        
            Returns:
                EnumOperation: EnumOperation class the contains the string in type_str.
            Tested:
                True
        """
        dic_operations = {'add': EnumOperation.ADD,
                'mul':EnumOperation.MULT,
                'load':EnumOperation.LOAD,
                'const': EnumOperation.CONST,
                'output':EnumOperation.OUTPUT,
                'store': EnumOperation.STORE}
        return dic_operations[type_str] if type_str in dic_operations else None
    
    @staticmethod
    def enum_operation_by_id(id_operation):
        """
        Get a EnumOperation class by id of the operation.

        Args:
            id_operation (int): Id of the EnumOperation.
        Returns:
            EnumOperation:  EnumOperation class corresponding the id.
        Tested:
            True
        """
        dic_operations = {0: EnumOperation.ADD,
                1:EnumOperation.MULT,
                3:EnumOperation.LOAD,
                2: EnumOperation.CONST,
                4:EnumOperation.OUTPUT,
                5: EnumOperation.STORE}
        return dic_operations[id_operation] if id_operation in dic_operations else None
    
    @staticmethod
    def index_type():
        """
        Returns the index of the EnumOperation values corresponding to the type of operation.
        """
        return  0
    
    @staticmethod 
    def index_id():
        """
        Returns the index of the EnumOperation values corresponding to the id of operation.
        """
        return 1