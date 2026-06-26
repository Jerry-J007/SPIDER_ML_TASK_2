**Project Overview**

**Kaagle link for base Task 3 :** https://www.kaggle.com/code/jerusonj/sml-task-2-base-3

**Collab link for base Task 3 :** https://colab.research.google.com/drive/1I9lcVCBnK_9S_HeVFEyxwSEgvCQJGiyZ?usp=sharing

**Kaagle link for LSTM Vs Transformer :** https://www.kaggle.com/code/jerusonj/lstm-vs-transformer

**Collab Link for LSTM Vs Transformer :** https://colab.research.google.com/drive/1Rrq--5KA7P4gQNfkNqAvAqLz4GFbqJgH?usp=sharing

This repository contains the implementation for Level 3 of the SPIDER Base ML Task. The objective of this project is to predict the temperature 12 hours into the future using a 72-hour historical window of the Jena Climate dataset.

In this level, an Encoder-only Transformer architecture was implemented using PyTorch, featuring a manually constructed Sinusoidal Positional Encoding mechanism. This model's performance is rigorously compared against a Custom LSTM built in Level 2.

**Repository Structure**

Below is a breakdown of the files included in this repository:

Analysis.pdf

The comprehensive final report. It details the architectural design (adhering to the 64 hidden dimensions, 4 attention heads, and 2 encoder layers constraints), the theoretical differences between Attention and Recurrence, and information about the comparative metrics (MSE, MAE, Huber Loss).

sml_task_2_base_3 (2).ipynb
The main training notebook. This contains:

    .Data preprocessing (downsampling 10-minute intervals to hourly averages).

    .Implementation of the manual Sinusoidal Positional Encoding.

    .The Encoder-only Transformer model definition.

    .The complete training loop utilizing gradient clipping and a ReduceLROnPlateau learning rate scheduler.

lstm_vs_transformer (1).ipynb

The comparative evaluation notebook. This script loads the trained weights for both the LSTM (Level 2) and the Transformer (Level 3) to run inference on the same unseen test split, generating the final side-by-side Matplotlib forecast plots.

Transformer_weights (2).pth

The saved PyTorch state dictionary containing the fully trained weights for the Transformer model, allowing for direct inference without retraining.

**Key Results**

The Transformer successfully outperformed the Custom LSTM across all major regression metrics, demonstrating superior handling of sudden atmospheric shifts and long-range dependencies without the "lag effect" commonly seen in recurrent models.
<img width="1102" height="287" alt="Screenshot 2026-06-26 221826" src="https://github.com/user-attachments/assets/57f59b9e-fb30-4ac8-8fa4-884a656bc0dd" />

**Technologies Used**

.PyTorch (nn.Module, TransformerEncoderLayer, MSELoss, HuberLoss)

.Pandas & NumPy (Data processing and downsampling)

.Scikit-Learn (StandardScaler)

.Matplotlib (Visualization)
