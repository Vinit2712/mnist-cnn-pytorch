Absolutely. Here's a **single clean README.md** you can copy directly into your project.

````markdown
# MNIST CNN Digit Classifier

A Convolutional Neural Network built with PyTorch to classify handwritten digits from 0 to 9 using the MNIST dataset.

This project was built to understand the complete deep learning workflow, including CNN architecture, forward propagation, loss calculation, backpropagation, optimization, validation, evaluation, error analysis, and experiment tracking.


## Project Overview

The objective of this project is to build a CNN that can recognize handwritten digits from images.

The complete workflow is:

text
MNIST Dataset
      ↓
Data Loading & Batching
      ↓
Convolutional Neural Network
      ↓
Forward Pass
      ↓
Cross-Entropy Loss
      ↓
Backpropagation
      ↓
Adam Optimizer
      ↓
Updated Weights
      ↓
Validation
      ↓
Final Test Evaluation
      ↓
Error Analysis

##  Dataset

The project uses the **MNIST handwritten digit dataset**.

* 60,000 original training images
* 10,000 test images
* Image size: `28 × 28`
* Grayscale images
* 1 input channel
* 10 classes: digits `0–9`

The original training dataset was split into:

* 54,000 training images
* 6,000 validation images

The 10,000 test images were kept separate for final evaluation.

##  CNN Architecture

The CNN consists of two convolutional blocks followed by a fully connected classification layer.

Input Image
[1 × 28 × 28]
      │
      ▼
┌─────────────────┐
│ Conv2D          │
│ 1 → 16 channels │
│ Kernel: 3 × 3   │
└─────────────────┘
      │
      ▼
    ReLU
      │
      ▼
┌─────────────────┐
│ MaxPool2D       │
│ Kernel: 2 × 2   │
│ Stride: 2       │
└─────────────────┘
      │
      ▼
[16 × 13 × 13]
      │
      ▼
┌─────────────────┐
│ Conv2D          │
│ 16 → 32 channel │
│ Kernel: 3 × 3   │
└─────────────────┘
      │
      ▼
    ReLU
      │
      ▼
┌─────────────────┐
│ MaxPool2D       │
│ Kernel: 2 × 2   │
│ Stride: 2       │
└─────────────────┘
      │
      ▼
[32 × 5 × 5]
      │
      ▼
    Flatten
      │
      ▼
[800 features]
      │
      ▼
┌─────────────────┐
│ Linear Layer    │
│ 800 → 10        │
└─────────────────┘
      │
      ▼
[10 Logits]
      │
      ▼
Predicted Digit


### Why these layers?

**Convolution**

Extracts visual features from the image, such as edges, curves, and shapes.

**ReLU**

Introduces non-linearity and removes negative activations.

**Max Pooling**

Reduces the spatial dimensions while retaining important features.

**Flatten**

Converts the 3D feature representation for each image into a 1D feature vector.

32 × 5 × 5 = 800

**Linear Layer**

Maps the 800 extracted features to 10 output logits representing digits `0–9`.

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Framework     | PyTorch          |
| Architecture  | 2-Layer CNN      |
| Optimizer     | Adam             |
| Learning Rate | 0.001            |
| Batch Size    | 64               |
| Epochs        | 5                |
| Loss Function | CrossEntropyLoss |

## Training Process

For every batch of 64 images, the following process occurs:


Images
  ↓
Forward Pass
  ↓
CNN Predictions
  ↓
10 Logits
  ↓
CrossEntropyLoss
  ↓
Backpropagation
  ↓
Calculate Gradients
  ↓
Adam Optimizer
  ↓
Update Weights

### Batches and Epochs

The training dataset contains 60,000 images.

With a batch size of 64:

60,000 / 64 ≈ 938 iterations per epoch

Therefore:

1 iteration → 64 images
1 epoch → entire training dataset
5 epochs → training dataset seen 5 times

The model does not reset its learned weights between epochs. The weights are continuously updated through backpropagation and optimization.

##  Training and Validation

Training and validation loss were tracked during training.

Example results:

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |        0.2890 |          0.0936 |
|     2 |        0.0790 |          0.0650 |
|     3 |        0.0599 |          0.0561 |
|     4 |        0.0484 |          0.0469 |
|     5 |        0.0415 |          0.0510 |

The validation loss reached its lowest value at **Epoch 4**.

Although the training loss continued to decrease during Epoch 5, the validation loss increased slightly. This provided an early indication of overfitting.

The best validation model was therefore saved using model checkpointing.

## Results

### Test Accuracy

The CNN achieved:

**98.80% test accuracy**

This means the model correctly classified approximately 98.8% of the 10,000 previously unseen MNIST test images.

##  Confusion Matrix

A confusion matrix was used to understand which digits the model struggled to distinguish.

The largest individual confusions were:


Actual 5 → Predicted 3 : 14 images
Actual 8 → Predicted 0 : 12 images


The model particularly struggled with some handwritten `5`s that visually resembled `3`s.


## Error Visualization

Misclassified images were visualized to understand the model's mistakes.

For example, the following types of handwritten `5`s were classified as `3`:

These examples demonstrate that the model's errors were not completely random. Some handwritten styles contain visual patterns that overlap with other digits.

##  Experiment Tracking with Weights & Biases

Weights & Biases (W&B) was used to track the training experiment.

The following metrics were logged:

* Training loss
* Validation loss
* Epoch

The W&B dashboard was used to visualize the learning curves and monitor the training process.

##  Project Structure

MNIST-CNN/
│
├── data/
│   └── MNIST/
│
├── images/
│   ├── wandb-training.png
│   └── misclassified_5_as_3.png
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize_errors.py
│
├── mnist_cnn_best.pth
├── requirements.txt
└── README.md


##  Installation

### 1. Clone the repository


git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MNIST-CNN


### 2. Create a virtual environment


python -m venv venv


### 3. Activate the virtual environment

#### Windows

venv\Scripts\activate


### 4. Install dependencies

pip install -r requirements.txt


## ▶ Running the Project

### Train the model


python src/train.py

This trains the CNN, tracks the experiment using W&B, validates the model after every epoch, and saves the model with the best validation loss.

### Evaluate the model

python src/evaluate.py

This evaluates the trained model on the 10,000-image MNIST test set.

### Visualize model errors

python src/visualize_errors.py

This displays examples of misclassified images.

##  Technologies Used

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Matplotlib
* Weights & Biases

---

##  Key Concepts Learned

Through this project, I learned:

* Tensors and tensor shapes
* Batch size and epochs
* DataLoaders
* CNN architecture
* Convolution and kernels
* Input and output channels
* Feature maps
* Padding and spatial dimensions
* ReLU activation
* Max Pooling
* Flattening tensors
* Linear layers
* Logits
* Cross-Entropy Loss
* Forward propagation
* Backpropagation
* Gradients
* Gradient descent
* Adam optimizer
* Training vs validation vs test data
* Generalization
* Overfitting
* Model checkpointing
* Confusion matrices
* Error visualization
* Experiment tracking with W&B

##  Future Improvements

Possible improvements to the model include:

* Data augmentation
* Dropout
* Batch normalization
* Learning-rate scheduling
* Early stopping
* Hyperparameter tuning
* Deeper CNN architectures
* Testing on handwritten images outside MNIST
* Comparing different CNN architectures


##  Project Outcome

This project provided hands-on experience with the complete deep learning workflow:

Dataset
   ↓
Preprocessing
   ↓
CNN Architecture
   ↓
Training
   ↓
Backpropagation
   ↓
Validation
   ↓
Model Selection
   ↓
Testing
   ↓
Error Analysis
   ↓
Experiment Tracking

The final CNN achieved 98.80% test accuracy on unseen MNIST handwritten digit images.