import torch
from torch import nn
import lib.d2l as d2l


def corr2d(X, K):
    """
    Compute 2D cross-correlation
        X: input
        K: kernel
    """
    k_height, k_width = K.shape

    # Create the output matrix and fill with zeros.
    Y = torch.zeros((
        X.shape[0] - k_height + 1,
        X.shape[1] - k_width + 1
    ))

    # For each row of the output:
    for i in range(Y.shape[0]):
        # For each column of the output
        for j in range(Y.shape[1]):
            # Set Y_ij to the sum of the corresponding, kernel-sized, subsection of X
            # multiplied element-wise by the kernel.
            Y[i, j] = (X[i:(i + k_height), j:(j + k_width)] * K).sum()

    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones((6, 8))
# For every row, set the [2:6) slice to 0's.
X[:, 2:6] = 0
# print(X)
# tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.]])
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
# print(Y)
# We see that:
#   0 indicates no edge.
#   1 indicates a white to black edge
#   -1 indicates a black to white edge
# tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])

conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f"epoch {i + 1}, loss {l.sum():.3f}")

# epoch 2, loss 14.239
# epoch 4, loss 4.133
# epoch 6, loss 1.408
# epoch 8, loss 0.529
# epoch 10, loss 0.209

learned_kernel = conv2d.weight.data.reshape((1, 2))
print(learned_kernel)
# We see that the learned_kernel is similar to the [1, -1] kernel we were training it to identify.
# tensor([[ 0.9714, -0.9978]])