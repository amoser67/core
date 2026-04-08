import torch
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


def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    # zip: zip((a, b), (c, d)) => Iterator((a, c), (b, d))
    return sum(corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)



def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(Y1)
print(Y2)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6