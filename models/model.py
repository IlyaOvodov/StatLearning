import torch
import torchvision.models as tvmodels
from .resnet_k import ResNet18 as ResNet18_kuangliu

class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True)
    )

def create_tiny_model(NUM_CLASSES = 10, device='cuda'):
    NUM_CLASSES = 10
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, NUM_CLASSES, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last, device=device)
    return model


def create_model(config, NUM_CLASSES = 10, device='cuda'):
    if config.model.type == 'tiny':
        model = create_tiny_model(NUM_CLASSES, device)
    elif config.model.type == 'ResNet18':
        model = tvmodels.resnet18()
    elif config.model.type == 'ResNet34':
        model = tvmodels.resnet34()
    elif config.model.type == 'ResNet18_kuangliu':
        model = ResNet18_kuangliu() # https://github.com/kuangliu/pytorch-cifar
    model=model.to(device=device)
    return model