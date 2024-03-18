import torch
import pocket

a = torch.rand(10, 1000)
b, p = pocket.ops.pca(a)
print(f"Reduced dimenality: {b.shape}.\nPrincipal components: {p.shape}")