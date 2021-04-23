import torch
import torch.nn as nn
import math
import torch.nn.init as init

class OptimizedResencblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(2))
        res_arch_init(self)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResencBlock(nn.Module):
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
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]

        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

        res_arch_init(self)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class Autoencorder_res(nn.Module):

    def __init__(self, inchannels, latent_dim, image_size):
        super(Autoencorder_res, self).__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            OptimizedResencblock(inchannels, 64),
            ResencBlock(64, 128, down=True),
            ResencBlock(128, 256, down=True),
            ResencBlock(256, 512, down=True),
            ResencBlock(512, 1024, down=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)

        res_arch_init(self)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, image):
        result = self.model(image)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var

class Encorder(nn.Module):

    def __init__(self, inchannels, latent_dim, image_size):
        super(Encorder, self).__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            OptimizedResencblock(inchannels, 64),
            ResencBlock(64, 128, down=True),
            ResencBlock(128, 256, down=True),
            ResencBlock(256, 512, down=True),
            ResencBlock(512, 1024, down=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(1024, latent_dim)

        res_arch_init(self)

    def forward(self, image):
        result = self.model(image)
        result = torch.flatten(result, start_dim=1)

        z = self.fc(result)

        return z

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