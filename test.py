import torch
from torch import nn
import lib.d2l as d2l
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


# torch.cuda.synchronize()
# lib_t0 = time.perf_counter()
#
# device = torch.device("cuda:0")
# model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
# model.to(device)
# data = d2l.FashionMNIST(batch_size=256)
# trainer = d2l.Trainer(max_epochs=8, num_gpus=1)
# trainer.fit(model, data)
# d2l.plt.show()
# torch.cuda.synchronize()
# print(f'Lib time: {time.perf_counter() - lib_t0:.2f} sec')

# sizes = [800, 1152, 1025, 1026, 1028, 1032, 1024]
# for size in sizes:
size = 1032
size_times = []
torch.cuda.synchronize()
for i in range(100):
    A = torch.randn(size, size, device='cuda:0')
    b = torch.randn(size, device='cuda:0')
    t0 = time.perf_counter()
    c = A @ b
    size_times.append(time.perf_counter() - t0)
torch.cuda.synchronize()
print(f'Size: {size}, average time: {sum(size_times) / len(size_times):.6f} sec')

# 1024: 0.000492 sec
# 1025: 0.000495 sec
# 1026: 0.000491 sec
# 1028: 0.000482 sec
# 1032: 0.000476 sec
# 1152: 0.000487 sec