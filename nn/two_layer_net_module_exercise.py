import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        h = self.linear1(x)
        h_relu = self.relu(h)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
lr = 1e-4

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

twoLayerNet = TwoLayerNet(D_in, H, D_out)

optimizer = optim.Adam(twoLayerNet.parameters(), lr)
loss_fn = nn.MSELoss(size_average=False)

for t in range(500):
    y_pred = twoLayerNet(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

dummy = 0




