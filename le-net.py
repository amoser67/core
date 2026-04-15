import torch
from torch import nn
import lib.d2l as d2l


device = torch.device("cuda:0")


def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)



class LeNet(d2l.Classifier):
    """The LeNet-5 model."""
    def __init__(
        self,
        lr=0.1,
        num_classes=10,
        conv_window_size=5,
        conv_1_padding=2,
        num_output_channels_1=6,
        num_output_channels_2=16
    ):
        super().__init__()
        self.save_hyperparameters()
        """
        - Each conv layer uses a 5x5 kernel and a sigmoid activation function.
        - Conv layers map spatially arranged inputs to a number of 2-dimensional feature maps, typically increasing #
        channels.
        - Each 2x2 pooling operation reduces dimensionality by a factor of 4 via spacial downsampling. 
        - The 10 dimensional final output layer corresponds to the number of possible output classes.
        """
        self.net = nn.Sequential(
            nn.LazyConv2d(num_output_channels_1, kernel_size=conv_window_size, padding=conv_1_padding), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(num_output_channels_2, kernel_size=conv_window_size), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Flatten(),

            nn.LazyLinear(120), nn.ReLU(),
            nn.LazyLinear(84), nn.ReLU(),
            nn.LazyLinear(num_classes))

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


# Summarize model when processing a 28x28 single channel input image.
# model = LeNet()
# model.layer_summary((1, 1, 28, 28))
"""
Conv2d output shape:	 torch.Size([1, 6, 28, 28])
Sigmoid output shape:	 torch.Size([1, 6, 28, 28])

AvgPool2d output shape:	 torch.Size([1, 6, 14, 14])

Conv2d output shape:	 torch.Size([1, 16, 10, 10])
Sigmoid output shape:	 torch.Size([1, 16, 10, 10])

AvgPool2d output shape:	 torch.Size([1, 16, 5, 5])

Flatten output shape:	 torch.Size([1, 400])

Linear output shape:	 torch.Size([1, 120])
Sigmoid output shape:	 torch.Size([1, 120])

Linear output shape:	 torch.Size([1, 84])
Sigmoid output shape:	 torch.Size([1, 84])

Linear output shape:	 torch.Size([1, 10])
"""

#
# class Reshape(torch.nn.Module):
#     def forward(self, x):
#         return x.view(-1, 1, 28, 28)
#
#
# net = torch.nn.Sequential(
#     Reshape(),
#     nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#
#     nn.Flatten(),
#
#     nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
#     nn.Linear(120, 84), nn.ReLU(),
#     nn.Linear(84, 10)
# )
#
#
#
# trainer = d2l.Trainer(max_epochs=15, num_gpus=1)
# data = d2l.FashionMNIST(batch_size=128)
# print(data.get_dataloader(True))
# x_first_relu_layer = net[0:3](data.get_dataloader(True)).cpu().detach().numpy()[0:9, 1, :, :]
#
# d2l.show_images(x_first_relu_layer.reshape(9, 28, 28), 1, 9)
#
# trainer = d2l.Trainer(max_epochs=15, num_gpus=1)
# data = d2l.FashionMNIST(batch_size=128)
# model = LeNet(
#     lr=0.1,
#     # conv_window_size=5,
#     # conv_1_padding=2,
#     # num_output_channels_1=8,
#     # num_output_channels_2=16
# )
# # model.to(device)
# model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
# trainer.fit(model, data)
# d2l.plt.show()
# print("Val loss:")
# print(float(model.board.data["val_loss"][-1].y))






"""
Exercise 1
    Implement and test the following changes:
        1. Replace average pooling with max-pooling.
        2. Replace the softmax layer with ReLU.
        
    1 by itself: Slight improvement
    1 and 2 (replacing final sigmoid with ReLU): Significant improvement, error from about .7 to .5
    
    Replacing all sigmoid activation functions with ReLU further reduces error. Error reduced to around .3
"""

"""
Exercise 2
    Try to adjust size of network to improve results, while keeping improvements from exercise 1.
    
    Note: numbers shown are val loss.
    Starting value is ~.305
    
    1. Adjust conv window size.
        Size 3: 0.290
            padding 1: 0.32966
        Size 5: .309
        Size 7: 0.3195
            padding 3: 0.3185
        
    2. Adjust the number of output channels
        conv_1=6,  conv_2=16: 0.3029
        conv_1=8,  conv_2=16: 0.2886
        conv_1=10, conv_2=16: 0.32568
        conv_1=8,  conv_2=20: 0.3089
        conv_1=8,  conv_2=12: 0.3078
        
    Combine best of 1 and 2:
        k=3, p=1, conv_1=8: 0.3094
        Hmmm,
        k=3, p=1, conv_1=6: 0.3255
        k=5, p=2, conv_1=8: 0.3051
        
    Not noticing significant differences so far.
    
    3. Adjust the number of convolution layers.
        Add conv(20, 2)+ReLU after second maxpool: 0.3242
            padding=1: 0.31648
        Duplicate second conv before maxpooling: 0.3307618498802185
        
            
    4. Adjust the number of fully connected layers
        Add lin(240) before lin(120): 0.3053807318210602
        Add lin(32) after lin(84): 0.3208458721637726
        Add lin(64) after lin(84): 0.34769612550735474
        Add lin(16) after lin(84): 0.31056660413742065
        Swap lin(120) with lin(180): 0.30179473757743835
        
    5. Adjust the learning rates and other details (initialization, # epochs)
    lr=.08 num_epochs=15: 0.2919
    lr=.1 num_epochs=15: 0.28537
    
"""

"""

Exercise 4
Display the activations of the first and second layer of LeNet for different inputs (e.g. sweaters and coats)

"""

# DataModule
data = d2l.FashionMNIST(batch_size=256)
pic = data.val.data[:2,:].type(torch.float32).unsqueeze(dim=1)
print(pic)


# torch.Module > d2l.Module > d2l.Classifier
# model = LeNet(lr=0.1)
#
# # Init weights.
# model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
#
# # d2l.Trainer
# trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
#
# trainer.fit(model, data)
#
# # Shows a sneaker and a long sleeve shirt image.
# # shape [2, 28, 28] => [2, 1, 28, 28]
# pic = data.val.data[:2,:].type(torch.float32).unsqueeze(dim=1)
# pic = pic.to(device)
# d2l.show_images(pic.squeeze().cpu(),1,2)
# # d2l.plt.show()
#
# # d2l.show_images(model.net[0](pic).squeeze().cpu().detach().numpy().reshape(-1,28,28),4,8)
# d2l.show_images(model.net[:2](pic).squeeze().detach().cpu().numpy().reshape(-1,28,28),4,8)
# d2l.plt.show()
