import torch
from torch.nn import Sequential as S
from torch.nn import Linear as L
from torch.nn import ReLU as R

# defino el modelo y lo asigno a model
model= S(L(784, 128), R(), L(128, 10))