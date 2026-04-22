import torch
from torch import nn
import lib.d2l as d2l
from utils import print_val_loss


class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        super().__init__()
        self.save_hyperparameters()
        # Interval [1, T], e.g. [1, 1000]
        self.time = torch.arange(1, T + 1, dtype=torch.float32)
        # Sine function with noise.
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2

    def get_dataloader(self, train):
        # [
        #   [x[0], ... , x[T-tau-0]],
        #   [x[1], ... , x[T-tau+1]],
        #   [x[2], ... , x[T-tau+2]],
        #   [x[3], ... , x[T-tau+3]]
        # ]
        # features = []
        # for i in range(self.tau * 2):
        #     if i == 0 or i % 2 == 0:
        #         features.append(self.x[i: self.T - (self.tau * 2) + i])
        features = [self.x[i: self.T - self.tau + i] for i in range(self.tau)]

        # [
        #  [x[0], x[1], x[2], x[3]],
        #  [x[1], x[2], x[3], x[4]],
        #  ...
        #  [x[T-tau-0], x[T-tau-1], x[T-tau-2], x[T-tau-3]]
        # ]
        self.features = torch.stack(features, 1)

        # [
        #   x[tau],
        #   x[tau+1],
        #   ...,
        #   x[T-1]
        # ]
        self.labels = self.x[self.tau:].reshape((-1, 1))
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.features, self.labels], train, i)


# d2l.plt.show()


data = Data()
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
print_val_loss(model)


def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(torch.stack(features[i : i+data.tau], 1))
        features.append(preds.reshape(-1))
    return features[data.tau:]
#
# steps = (1, 4, 16, 64)
# preds = k_step_pred(steps[-1])
# d2l.plot(data.time[data.tau+steps[-1]-1:],
#          [preds[k - 1].detach().numpy() for k in steps], 'time', 'x',
#          legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
# d2l.plt.show()


"""
Exercises

1. Improve the model.
    a. Incorporate more than past 4 observations. How many do you really need.
        Tested up to 28 and it doesn't really seem like it helps much.
    b. How many obs needed if no noise?
        Much fewer. 12 got val loss of .0015.
    c. Can you incorporate older obs and keep feature count the same?
        Using step size of 2, val loss = .0228
        Using step size of 1, val loss = .06
        Surprisingly, skipping some previous steps results in better result. This is probably because it gives
        the model a better idea of the overall shape of the function.
    d. Train with more epochs, what happens?
        Doesn't seem to make much of a difference.

3. To what extent does causality apply to text.
    Yes, but to a lesser extent. Sine is a deterministic function. Given some x and step size s,
    sin(x + s) can always be determined by some f(sin(x), s). For text, this is not the case.
    "I am going to the " could be followed by "bank", "store", etc. That said, Claude Shannon
    states redundancy of English language is about 50%, meaning causality does have an effect.
    
4. Give example where latent autoregressive model might be needed to capture the dynamic of the data.
    Text might be a use case for latent model, since predicting the next word depends not only on the previous
    x words, but also on the overall context.  
"""