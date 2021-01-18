import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm
from tools import AvgMeter


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.optimizer = None
        self.lr_scheduler = None
        self.current_epoch = None

    def fit(
        self, train_dataset, valid_dataset, criterion, epochs, batch_size, device="cuda"
    ):
        train_loader, valid_loader = self.build_loaders(
            train_dataset, valid_dataset, batch_size
        )
        self.device = torch.device(device)
        self.criterion = criterion
        self.to(self.devcie)

        self.set_optimizer()
        self.set_lr_scheduler()

        for i in range(epochs):
            self.current_epoch = i + 1
            print(f"Epoch {self.current_epoch}")
            print(f"Current Learning Rate: {self.get_lr():.4f}")
            print("*" * 20)
            self.train()
            train_loss, train_metrics = self.one_epoch(train_loader, mode="train")
            self.eval()
            with torch.no_grad():
                valid_loss, valid_metrics = self.one_epoch(valid_loader, mode="valid")
            # print loss and metric
            print("*" * 20)

        # Logic to handle Saving the best model and lr_scheduler

    def one_epoch(self, loader, mode):
        metrics = {}
        loss_meter = AvgMeter()
        for xb, yb in tqdm(loader):
            xb, yb = xb.to(device), yb.to(device)
            preds = self(xb)
            loss = self.criterion(preds, yb)
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if isinstance(self.lr_scheduler, optim.lr_scheduler.OneCycleLR):
                    self.lr_scheduler.step()

            loss_meter.update(loss.item(), count=xb.size(0))
            metrics = self.update_metrics(preds.detach(), yb, metrics)

        return loss_meter, metrics

    def update_metrics(self, preds, target, metrics):
        # Logic to handle metrics calc.
        return metrics

    def set_optimizer(self):
        self.optimizer = optim.Adam(
            self.parameters(), lr=1e-4
        )  # Hard coded optimizer! Review needed

    def set_lr_scheduler(self):
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def build_loaders(self, train_dataset, valid_dataset, batch_size):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, valid_loader

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]


from torchvision.models import resnet18

device = torch.device("cuda")
model = resnet18(pretrained=False).to(device)

print(device)
print(next(model.parameters()).device == device)

