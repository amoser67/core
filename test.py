import torch
from torch import nn
import d2l
import time


class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens),
            nn.ReLU(),
            nn.LazyLinear(num_outputs)
        )


lib_t0 = time.perf_counter()

device = torch.device("cuda:0")
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
model.to(device)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=8, num_gpus=1)
trainer.fit(model, data)
d2l.plt.show()

print(f'Lib time: {time.perf_counter() - lib_t0:.2f} sec')

