import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms as Transforms

from trainer import Model
from tools import AvgMeter


class MyModel(Model):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
        self.model = model

    def forward(self, x):
        x = torch.cat([x] * 3, dim=1)
        return self.model(x)

    def update_metrics(self, preds, target, metrics):
        if self.current_batch == 1:
            metrics["Accuracy"] = AvgMeter()

        preds = preds.argmax(dim=1)
        accuracy = (preds == target).float().mean()
        metrics["Accuracy"].update(accuracy, count=preds.size(0))
        return metrics


transforms = Transforms.Compose([Transforms.ToTensor(),])

train_dataset = datasets.MNIST(
    root="C:\Moein\AI\Datasets", train=True, download=False, transform=transforms
)
valid_dataset = datasets.MNIST(
    root="C:\Moein\AI\Datasets", train=False, download=False, transform=transforms
)


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.data = [data for data in dataset]
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

model = MyModel()
model.fit(
    MyDataset(train_dataset), MyDataset(valid_dataset), nn.CrossEntropyLoss(), 5, 512, file_name="mnist.pt"
)

