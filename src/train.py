import numpy as np
from modeling import model 
from tqdm import tqdm
from data_process import *
import torch

# Define size and number of batches
bs = 32
num_batches = len(X_train) // bs

# Instantiate objects from classes 'CrossEntropyLoss' and 'Optim.Adam'
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training the model:
for epoch in range(10):
    for b in tqdm(range(num_batches)):
        x = X_train[b * bs : (b + 1) * bs]
        y = y_train[b * bs : (b + 1) * bs]
        y_hat = model(x)  # prediction from the model

        # Calculate loss function with labels and predictions as parameters
        loss = loss_fn(y_hat, y)

        # Gradients represent the magnitude and direction in which the weights should be adjusted to optimize the model.
        optimizer.zero_grad()  # Gradient reset to 0
        loss.backward()  # Calculate how to adjust model's weights to improve predictions
        optimizer.step()  # Apply calculated changes
    print(f"Epoch {epoch+1} loss: {loss.item():.3f}")