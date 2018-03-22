import torch
import torch.nn as nn
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

twoLayerNet = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))
loss_fn = nn.MSELoss(size_average=False)
lr = 1e-4
for t in range(500):
    y_pred = twoLayerNet(x)
    ####################################################################################################
    #   The params for loss function should in order of [prediction, target]. not [target, prediction]
    #   So in this case "loss = loss_fn(y, y_pred)" is wrong.
    loss = loss_fn(y_pred, y)
    ####################################################################################################
    print(t, loss.data[0])
    twoLayerNet.zero_grad()
    loss.backward()
    for param in twoLayerNet.parameters():
        param.data -= lr * param.grad.data
dummy = 0


