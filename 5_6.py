import torch
from torch import nn
import lib.d2l as d2l


device = torch.device("cuda:0")


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X, device=device)
    # Get a tensor of shape X composed of 1's and 0's.
    mask = (torch.rand(X.shape, device=device) > dropout).float()
    # Multiply X by mask, element-wise, then divide by 1 - dropout, also element-wise.
    return mask * X / (1.0 - dropout)


# X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print('dropout_p = 0:', dropout_layer(X, 0))
# print('dropout_p = 0.5:', dropout_layer(X, 0.5))
# print('dropout_p = 1:', dropout_layer(X, 1))


class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2, dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)


class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens_1), nn.ReLU(), nn.Dropout(dropout_1),
            nn.LazyLinear(num_hiddens_2), nn.ReLU(), nn.Dropout(dropout_2),
            nn.LazyLinear(num_outputs))
    
    
hparams = {"num_outputs": 10, "num_hiddens_1": 256, "num_hiddens_2": 256,
           "dropout_1": 0.75, "dropout_2": 0.25, "lr": 0.1}

model = DropoutMLP(**hparams)
model.to(device)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
d2l.plt.show()
