import torch
from base_model import ft_net

net = ft_net(751)

a = torch.randn(16 ,3, 256, 128)

b = net(a)

print(b.size())
