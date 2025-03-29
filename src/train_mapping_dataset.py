from torch.utils.data.dataloader import Dataset
class TrainMappingDataset(Dataset):
    def __init__(self,mappings):
        self.mappings = mappings
        self.len = len(mappings[0])

    def __getitem__(self,index):
        returns = []
        for i in range(len(self.mappings)):
            returns.append(self.mappings[i][index])
        return returns
    def __len__(self):
        return self.len