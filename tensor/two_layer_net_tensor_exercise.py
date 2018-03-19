import torch
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).type(torch.FloatTensor)
y = torch.randn(N, D_out).type(torch.FloatTensor)

w1 = torch.randn(D_in, H).type(torch.FloatTensor)
w2 = torch.randn(H, D_out).type(torch.FloatTensor)

lr = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(w2)

    t1 = y_pred - y
    t2 = t1.pow(2)
    loss = t2.sum()
    print(t, loss)
    d_loss_over_d_y_pred = 2.0 * t1
    d_loss_over_d_w2 = h_relu.t().mm(d_loss_over_d_y_pred)
    d_loss_over_d_h_relu = d_loss_over_d_y_pred.mm(w2.t())
    t5 = torch.ones_like(h_relu)
    t5[h < 0] = 0
    d_loss_over_d_h = torch.mul(d_loss_over_d_h_relu, t5)
    d_loss_over_d_w1 = x.t().mm(d_loss_over_d_h)
    w2 -= lr * d_loss_over_d_w2
    w1 -= lr * d_loss_over_d_w1

dummy = 0