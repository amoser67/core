import torch
from torch import nn
import lib.d2l as d2l
from utils import print_val_loss


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode.
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average.
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # Assert X is 2 or 4 dimensions.
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension.
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later.
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used.
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average.
        # Unclear what 'momentum' is, assume it's similar in concept to a learning rate.
        moving_mean = ((1.0 - momentum) * moving_mean) + (momentum * mean)
        moving_var = ((1.0 - momentum) * moving_var) + (momentum * var)
    Y = (gamma * X_hat) + beta  # Scale and shift
    # tensor.data: A legacy attribute used to access the underlying tensor data
    # while bypassing the Autograd engine. Recommended to use something like
    # .detach() or .item() instead.
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer.
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively.
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1.
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located.
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var.
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y



class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.2, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120),
            BatchNorm(120, num_dims=2), nn.ReLU(), nn.LazyLinear(84),
            BatchNorm(84, num_dims=2), nn.ReLU(),
            nn.LazyLinear(num_classes))


trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
d2l.plt.show()
print_val_loss(model)

# With sigmoid and avg pooling: .36
# With relu and max pooling: 0.29606112837791443

# print(model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,)))
# (
#   tensor([1.4334, 1.9905, 1.8584, 2.0740, 2.0522, 1.8877], device='cuda:0', grad_fn=<ViewBackward0>),
#   tensor([ 0.7354, -1.3538, -0.2567, -0.9991, -0.3028,  1.3125], device='cuda:0', grad_fn=<ViewBackward0>)
# )