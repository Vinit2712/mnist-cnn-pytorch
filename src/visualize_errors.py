import torch
import matplotlib.pyplot as plt

from model import CNN
from dataset import test_Loader

model = CNN()

model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

wrong_images = []
wrong_prediction = []
wrong_labels = []

with torch.no_grad():
    for images , labels in test_Loader:
        outputs = model(images)
        _, predicted = torch.max(outputs , 1)
        for i in range(len(labels)):
            if labels[i] == 5 and predicted[i] == 3:
                wrong_images.append(images[i])
                wrong_prediction.append(predicted[i])
                wrong_labels.append(labels[i])

plt.figure(figsize = (10,5))

for i in range(min(10 , len(wrong_images))):
    plt.subplot(2,5,i+1)
    plt.imshow(
        wrong_images[i].squeeze(),
        cmap = "gray"
    )
    plt.title(
        f"Actual: {wrong_labels[i].item()}\n"
        f"Pred: {wrong_prediction[i].item()}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()