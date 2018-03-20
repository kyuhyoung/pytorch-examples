import torch
from torch.autograd import Variable

class RectifiedLinearUnit(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min = 0)
    def backward(self, grad_output):
        saved_input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[saved_input < 0] = 0
        return grad_input


N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(torch.FloatTensor))
y = Variable(torch.randn(N, D_out).type(torch.FloatTensor))

w1 = Variable(torch.randn(D_in, H).type(torch.FloatTensor), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(torch.FloatTensor), requires_grad=True)

lr = 1e-6
for t in range(500):

    if t > 0:
        print('w1.grad.data[0, 0] before zero_() : ', w1.grad.data[0, 0])
        print('w2.grad.data[0, 0] before zero_() : ', w2.grad.data[0, 0])
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        print('w1.grad.data[0, 0] after zero_() : ', w1.grad.data[0, 0])
        print('w2.grad.data[0, 0] after zero_() : ', w2.grad.data[0, 0])

    relu = RectifiedLinearUnit()
    h = x.mm(w1)
    h_relu = relu(h)

    y_pred = h_relu.mm(w2)
    t1 = y_pred - y
    t2 = t1.pow(2)
    loss = t2.sum()
    t3 = loss.data[0]
    print(t, t3)
    loss.backward()
    print('w1.grad.data[0, 0] after loss.backward() : ', w1.grad.data[0, 0])
    print('w2.grad.data[0, 0] after loss.backward() : ', w2.grad.data[0, 0])
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
dummy = 0





