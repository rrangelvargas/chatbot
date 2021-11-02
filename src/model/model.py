import torch

from src.utils import USE_CUDA


class Model:
    def __init__(self):
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')





