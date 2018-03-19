import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

'''
x = torch.randn(N, D_in).type(torch.FloatTensor)
y = torch.randn(N, D_out).type(torch.FloatTensor)
'''
x = Variable(torch.randn(N, D_in).type(torch.FloatTensor))
y = Variable(torch.randn(N, D_out).type(torch.FloatTensor))

w1 = Variable(torch.randn(D_in, H).type(torch.FloatTensor), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(torch.FloatTensor), requires_grad=True)

lr = 1e-6
for t in range(500):
    #w1.grad.data.zero_()
    #w2.grad.data.zero_()

    h = x.mm(w1)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(w2)
    t1 = y_pred - y
    t2 = t1.pow(2)
    loss = t2.sum()
    t3 = loss.data[0]
    print(t, t3)
    #print('w1.grad.data[0, 0] before : ', w1.grad.data[0, 0])
    #print('w2.grad.data[0, 0] before : ', w2.grad.data[0, 0])
    loss.backward()
    print('w1.grad.data[0, 0] after : ', w1.grad.data[0, 0])
    print('w2.grad.data[0, 0] after : ', w2.grad.data[0, 0])
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()

dummy = 0




























