from enum import Enum

class EnumFunctionalitiePE(Enum):
    LOGICAL = 0
    ARITHMETIC = 1
    MEMORY_ACCESS = 2
    NONE = 3

    @staticmethod
    def len():
        return len(vars(EnumFunctionalitiePE))