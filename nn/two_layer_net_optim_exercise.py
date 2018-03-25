import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

twoLayerNet = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))
loss_fn = nn.MSELoss(size_average=False)
lr = 1e-4
optimizer = optim.Adam(twoLayerNet.parameters(), lr)
for t in range(500):
    y_pre = twoLayerNet(x)
    loss = loss_fn(y_pre, y)
    print(t, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
dummy = 0




