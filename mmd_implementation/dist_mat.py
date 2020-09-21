from torch import cdist
import torch

p = torch.randn(10,2)
q = 2 + torch.randn(10,2)s

cdist(p,q)
