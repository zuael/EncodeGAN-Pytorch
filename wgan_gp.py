import math
import torch
import torch.nn as nn
import torch.nn.init as init

class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        res_arch_init(self)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResGenerator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 4 * 4 * 1024)

        self.blocks = nn.Sequential(
            ResGenBlock(1024, 512),
            ResGenBlock(512, 256),
            ResGenBlock(256, 128),
            ResGenBlock(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        res_arch_init(self)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 1024, 4, 4)
        return self.blocks(inputs)


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))
        res_arch_init(self)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]

        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

        res_arch_init(self)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 64),
            ResDisBlock(64, 128, down=True),
            ResDisBlock(128, 256, down=True),
            ResDisBlock(256, 512, down=True),
            ResDisBlock(512, 1024, down=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.linear = nn.Linear(1024, 1)

        res_arch_init(self)

    def forward(self, image):
        x = self.model(image)
        feature = torch.flatten(x, start_dim=1)
        adv = self.linear(feature)

        return adv, feature


def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
