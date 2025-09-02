from sklearn.datasets import fetch_openml
import numpy as np
import torch

# descargo datos y se asigna a la variable mnist
mnist = fetch_openml("mnist_784", version=1)

# extraemos datos y etiquetas de datos y los asignamos a X,y
X, y = mnist["data"].values.astype(np.float32), mnist["target"].values.astype(int)

# separo datos en entrenamiento (los primeros 60000) y test (los últimos 10000)
X_train, X_test = torch.from_numpy(X[:60000] / 255.0), torch.from_numpy(
    X[60000:] / 255.0
)
y_train, y_test = torch.from_numpy(y[:60000]), torch.from_numpy(y[60000:])