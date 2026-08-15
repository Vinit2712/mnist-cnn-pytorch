import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader , random_split

train_data = datasets.MNIST(
    root="../data",
    train = True,
    download = True,
    transform = ToTensor()
)

train_data,val_data = random_split(
    train_data,
    [54000 , 6000]
)

test_data = datasets.MNIST(
    root = "../data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_loader = DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True
)

val_loader = DataLoader(
    val_data,
    batch_size = 64,
    shuffle = False
)

test_Loader = DataLoader(
    test_data,
    batch_size = 64,
    shuffle = False
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("First label:", labels[0])