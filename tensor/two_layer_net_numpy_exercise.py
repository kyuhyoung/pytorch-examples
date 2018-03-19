import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

lr = 1e-6
for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    t1 = y_pred - y
    t2 = np.square(t1)
    #t3 = t2.sum(axis = 1)
    #t4 = t3.sum(axis = 0)
    loss = t2.sum()
    print(t, loss)
    d_loss_over_d_y_pred = 2.0 * (y_pred - y)
    d_loss_over_d_w2 = h_relu.T.dot(d_loss_over_d_y_pred)

    d_loss_over_d_h_relu = d_loss_over_d_y_pred.dot(w2.T)
    t5 = np.ones_like(h_relu)
    t5[h < 0] = 0
    d_loss_over_d_h = np.multiply(d_loss_over_d_h_relu, t5)
    d_loss_over_d_w1 = x.T.dot(d_loss_over_d_h)

    w2 -= lr * d_loss_over_d_w2
    w1 -= lr * d_loss_over_d_w1

dummy = 0