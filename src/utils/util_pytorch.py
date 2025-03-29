import torch
class UtilPytorch:
    @staticmethod
    def dict_to_cpu(dictionary):
        cpu_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, torch.Tensor):
                cpu_dict[key] = value.cpu()
            elif isinstance(value, dict):
                cpu_dict[key] = UtilPytorch.dict_to_cpu(value)
            else:
                cpu_dict[key] = value
        return cpu_dict