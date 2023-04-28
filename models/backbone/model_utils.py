import torch
import torch.nn as nn
import numpy as np

class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, bias=True):
        super(Conv2D, self).__init__()

        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias = bias))
        layers.append(nn.BatchNorm2d(num_features=output_dim, eps=1e-3, momentum=0.01))
        layers.append(nn.ReLU())
        self.convolution = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution(x)

class Conv2DSequence(nn.Module):
    """Block with 2D convolutions after each other with ReLU activation"""
    def __init__(self, input_dim, output_dim, kernel=3, depth=2):
        super(Conv2DSequence, self).__init__()
        layers = []

        for i in range(depth-1):
            layers.append(Conv2D(input_dim, input_dim, kernel_size=kernel, bias = False))

        layers.append(Conv2D(input_dim, output_dim, kernel_size=kernel, bias = True))

        self.convolution = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution(x)

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def mu_init(num_expert, latent_dim):
    switch = np.random.rand(num_expert, latent_dim)
    for i in range(latent_dim):
        switch[:2 ** latent_dim, i] = 2 ** i * (
        2 ** (latent_dim - 1 - i) * [0] + 2 ** (latent_dim - 1 - i) * [1])

    return switch