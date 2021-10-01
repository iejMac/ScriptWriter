import model
import torch

mod = model.Transformer(5, 2, 1)
x = torch.randn((1, 2, 5))

out = mod(x)
print(out.shape)

