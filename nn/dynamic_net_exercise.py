import torch
import torch.nn as nn
from torch.autograd import Variable
import random

class SomeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SomeLayerNet, self).__init__()
        self.linear_first = nn.Linear(D_in, H)
        self.linear_middle = nn.Linear(H, H)
        self.linear_last = nn.Linear(H, D_out)
        self.relu = nn.ReLU()
    def forward(self, x, n):
        h = self.linear_first(x)
        h_relu = self.relu(h)
        n_middle = random.randint(0, n)
        print ('n_middle : ', n_middle)
        for _ in range(n_middle):
            h = self.linear_middle(h_relu)
            h_relu = self.relu(h)
        y_pred = self.linear_last(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
lr = 1e-4
n = 3
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

someLayerNet = SomeLayerNet(D_in, H, D_out)

loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(someLayerNet.parameters(), lr, momentum=0.9)

for t in range(1000):
    y_pred = someLayerNet(x, n)
    loss = loss_fn(y_pred, y)
    print (t, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

dummy = 0
