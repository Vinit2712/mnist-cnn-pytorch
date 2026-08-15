import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 3 
        )

        self.pool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3
        )

        self.fc = nn.Linear(
            in_features = 32 * 5 * 5,
            out_features = 10
        )

        

    def forward(self , x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x , 1)
        x = self.fc(x)
        return x

    


if __name__ == "__main__":
    model = CNN()
    x = torch.randn(64,1,28,28)
    output = model(x)

    print("Input Shape: ",x.shape)
    print("Output Shape: ",output.shape)
