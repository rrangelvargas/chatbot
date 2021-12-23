import torch

# variavel global para determinar se vai ser usado CPU ou GPU para o treinamento da rede neural
USE_CUDA = torch.cuda.is_available()
