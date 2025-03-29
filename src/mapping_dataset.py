from torch.utils.data.dataloader import Dataset
class MappingDataset(Dataset):
    def __init__(self,mappings):
        self.mappings = mappings
        self.len = len(mappings)
    def __getitem__(self,i):
        return self.mappings[i]
    def __len__(self):
        return self.len