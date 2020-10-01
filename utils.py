import torch

class MOADataset:
    def __init__(self, features, targets):
        # both features and targets are numpy arrays
        self.features = features
        self.targets = targets
    
    def __getitem__(self):
        return {
            "x": torch.tensor(self.features[item, :], dtype = torch.float),
            "y": torch.tensor(self.targets[item, :], dtype = torch.float)
        }
    def __len__(self):
        return self.features.shape[0]