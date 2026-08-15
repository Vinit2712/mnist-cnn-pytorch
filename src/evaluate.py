import torch 
from sklearn.metrics import confusion_matrix

from model import CNN
from dataset import test_Loader

model = CNN()
model.load_state_dict(torch.load("mnist_cnn_best.pth"))

model.eval()

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for images , labels in test_Loader:
        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

accuracy = 100 * correct/total
print(f"Test Accuracy : {accuracy:.2f}%")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n Confusion Matrix: ",cm)