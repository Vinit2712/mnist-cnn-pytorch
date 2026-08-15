import torch
import wandb
from torch import nn
from torch import optim

from model import CNN
from dataset import train_loader , val_loader

model = CNN()

wandb.init(
    project = "mnist-cnn",
    config = {
        "epochs":5,
        "batch_size":64,
        "learning_rate":0.001,
        "optimizer":"Adam",
        "architecture":"2-layer CNN"
    }
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

train_losses = []
val_losses = []
epochs = 5 

best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images , labels in train_loader:

        #Forward Pass

        outputs = model(images)

        #Calculate Loss
        
        loss = criterion(outputs,labels)

        #Backpropogation

        optimizer.zero_grad()
        loss.backward()

        #Update Weights

        optimizer.step()

        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    total_val_loss = 0

    model.eval()

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(val_loader)

    val_losses.append(average_val_loss)

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(
            model.state_dict(),
            "mnist_cnn_best.pth"
        )
        print("Best model saved!")

    print(
    f"Epoch [{epoch + 1}/{epochs}] "
    f"Train Loss: {average_loss:.4f} "
    f"Val Loss: {average_val_loss:.4f}"
    )

    wandb.log({
        "train_loss": average_loss,
        "val_loss":average_val_loss,
        "epoch": epoch + 1
    })
wandb.finish()
