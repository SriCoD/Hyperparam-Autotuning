import torch
import torch.nn as nn


class MOADataset:
    def __init__(self, features, targets):
        # both features and targets are numpy arrays
        self.features = features
        self.targets = targets

    def __getitem__(self):
        return {
            "x": torch.tensor(self.features[item, :], dtype=torch.float),
            "y": torch.tensor(self.targets[item, :], dtype=torch.float),
        }

    def __len__(self):
        return self.features.shape[0]


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)
