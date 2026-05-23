import torch
import torch.nn as nn

layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
model = nn.TransformerEncoder(layer, num_layers=2)

t1 = torch.randn(5, 16, requires_grad=True)
t2 = torch.randn(8, 16, requires_grad=True)

nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)
out = model(nt)
out_list = out.unbind()
cls_out = torch.stack([o[0] for o in out_list])
loss = cls_out.sum()
loss.backward()

print("t1 grad:", t1.grad is not None)
