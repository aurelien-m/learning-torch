import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(tensor)
print(tensor.dtype)
print(tensor.device)
print(tensor.shape)
print(tensor.requires_grad)

x = torch.empty(size=(3, 3))
print(x)

x = torch.zeros((3, 3))
print(x)

x = torch.rand((3, 3))
print(x)

x = torch.ones((3, 3))
print(x)

x = torch.eye(5, 5)
print(x)

x = torch.arange(start=0, end=5, step=1)
print(x)

x = torch.linspace(start=0.1, end=1, steps=10)
print(x)

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)

x = torch.empty(size=(1, 5)).uniform_(0, 1)
print(x)

x = torch.diag(torch.ones(3))
print(x)

tensor = torch.arange(4)
print(tensor.bool())
print(tensor.float()) # mostly used: can be trained with any GPU
print(tensor.short())
print(tensor.long())
print(tensor.half()) # float 16: can be used to be trained on rtx 2000+ series
print(tensor.double())

import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array = tensor.numpy()

