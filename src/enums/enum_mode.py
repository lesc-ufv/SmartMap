from enum import Enum, auto

class EnumMode(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'
    DATA_GENERATION = 'DATA_GENERATION'
    FINETUNE = 'FINETUNE'
    ZERO_SHOT = 'ZERO_SHOT'
    @staticmethod
    def get_enum_by_value(value):
        for member in EnumMode:
            if member.value == value:
                return member
        return None

