import torch

t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(torch.einsum('ij->i', t))
