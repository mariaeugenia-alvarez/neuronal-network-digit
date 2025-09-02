from sklearn.datasets import fetch_openml
import numpy as np
import torch

# Constants for maintainability
TRAIN_SIZE = 60000
PIXEL_MAX_VALUE = 255.0

# Load data and assign to the variable mnist
mnist = fetch_openml("mnist_784", version=1)

# Extract data and labels and assign to X, y
X, y = mnist["data"].values.astype(np.float32), mnist["target"].values.astype(int)

# Separate data into training (first 60000) and test (last 10000)
X_train, X_test = torch.from_numpy(X[:60000] / 255.0), torch.from_numpy(
    X[60000:] / 255.0
)
y_train, y_test = torch.from_numpy(y[:60000]), torch.from_numpy(y[60000:])