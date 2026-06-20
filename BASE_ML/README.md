# Kaagle link : https://www.kaggle.com/code/jerusonj/sml-task-2-base
# Collab Link : https://colab.research.google.com/drive/1gfeueSIGWTNlhsKlrTsRZkE-VBULVJAc?usp=sharing

# 📌 Project Overview
This repository contains a Convolutional Neural Network (CNN) built entirely from scratch using PyTorch to classify the CIFAR-10 dataset.
The primary objective of this project is to demonstrate a fundamental understanding of Residual Learning. By manually implementing residual blocks, skip connections, and handling dimension mismatches, this architecture completely avoids pre-built models (like torchvision.models.resnet) to deeply explore how skip connections solve the vanishing gradient problem and allow for deeper, more accurate networks.

# 🧠 Architecture Design:
The network is a custom ResNet built to the specific task constraints:Stages: 3 distinct residual stages.Blocks: Each stage contains 2 residual blocks (a [2, 2, 2] configuration).Channel Progression: The feature maps expand across stages (16-->32-->64) while spatial dimensions are halved via a stride of 2.Skip Connections: Implemented a custom identity shortcut. When dimension mismatches occur during downsampling, a 1 x 1 convolutional projection layer is triggered to align tensor shapes.Classifier Head: A Global Average Pooling (GAP) layer condenses the final 8 x 8 x 64 tensor into a flat vector, feeding into a single fully connected layer for the 10 classes.

# 📊 Training & Performance Metrics
The model was trained on a Tesla T4 GPU using the following hyperparameters:
Epochs: 50
Optimizer: Adam (Initial Learning Rate = 0.001)
Scheduler: StepLR (Decays LR by a factor of 0.1 at Epoch 20)
Loss Function: Cross-Entropy Loss
Regularization: Random Cropping (padding=4), Horizontal Flipping, and Batch Normalization.

# Final Results
Training Accuracy: ~92.32%

Validation Accuracy: ~88.09%
