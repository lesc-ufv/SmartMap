from enum import Enum

class EnumInterconnections(Enum):
    MESH = "MESH"
    ONE_HOP = "ONE_HOP"
    OH_TOR_DIAG = "OH_TOR_DIAG"
    @staticmethod
    def get_enum_by_value(value):
        for member in EnumInterconnections:
            if member.value == value:
                return member
        assert False