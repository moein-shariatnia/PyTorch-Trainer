import torch
from torch import nn, optim

import copy
from tqdm import tqdm
from tools import AvgMeter


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.optimizer = None
        self.lr_scheduler = None
        self.current_epoch = None
        self.current_lr = None
        self.current_batch = 0
        self.best_model_weights = None
        self.best_loss = float("inf")

    def fit(
        self,
        train_dataset,
        valid_dataset,
        criterion,
        epochs,
        batch_size,
        device="cuda",
        file_name="model.pt",
    ):
        train_loader, valid_loader = self.build_loaders(
            train_dataset, valid_dataset, batch_size
        )
        self.device = torch.device(device)
        self.criterion = criterion
        self.to(self.device)

        self.set_optimizer()
        self.set_lr_scheduler()

        self.best_model_weights = copy.deepcopy(self.state_dict())
        self.metrics = {"train": {}, "valid": {}}

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            print("*" * 30)
            print(f"Epoch {self.current_epoch}")
            self.current_lr = self.get_lr()
            print(f"Current Learning Rate: {self.current_lr:.4f}")
            self.train()
            train_loss = self.one_epoch(train_loader, mode="train")
            self.eval()
            with torch.no_grad():
                valid_loss = self.one_epoch(valid_loader, mode="valid")

            self.epoch_end(valid_loss, file_name=file_name)
            self.log_metrics(train_loss, valid_loss)
            print("*" * 30)

    def epoch_end(self, valid_loss, file_name):
        if valid_loss.avg < self.best_loss:
            self.best_loss = valid_loss.avg
            self.best_model_weights = copy.deepcopy(self.state_dict())
            torch.save(self.state_dict(), file_name)
            print("Saved best model!")
        
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(valid_loss.avg)
            if self.current_lr != self.get_lr():
                print("Loading best model weights!")
                self.load_state_dict(torch.load(file_name, map_location=self.device))
    
    def log_metrics(self, train_loss, valid_loss):
        print(f"Train Loss: {train_loss.avg:.4f}")
        for key, value in self.metrics["train"][self.current_epoch].items():
            print(f"Train {key}: {value}")
        print(f"Valid Loss: {valid_loss.avg:.4f}")
        for key, value in self.metrics["valid"][self.current_epoch].items():
            print(f"Valid {key}: {value}")


    def one_epoch(self, loader, mode):
        metrics = self.get_metrics()
        if metrics.get(self.current_epoch, None) == None:
           metrics[self.current_epoch] = {}
        loss_meter = AvgMeter()
        for xb, yb in tqdm(loader):
            self.current_batch += 1
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self(xb)
            loss = self.criterion(preds, yb)
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if isinstance(self.lr_scheduler, optim.lr_scheduler.OneCycleLR):
                    self.lr_scheduler.step()

            loss_meter.update(loss.item(), count=xb.size(0))
            self.update_metrics(preds.detach(), yb)
            self.current_batch = 0

        return loss_meter

    def get_metrics(self):
        return self.metrics["train"] if self.training else self.metrics["valid"]
    
    def update_metrics(self, preds, target):
        # Logic to handle metrics calc.
        raise Exception()

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