import torch
import torch.nn as nn
import math
from wgan_gp import OptimizedResDisblock, ResDisBlock
import torch.nn.init as init

class Autoencorder_res(nn.Module):

    def __init__(self, inchannels, latent_dim, image_size):
        super(Autoencorder_res, self).__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            OptimizedResDisblock(inchannels, 64),
            ResDisBlock(64, 128, down=True),
            ResDisBlock(128, 256, down=True),
            ResDisBlock(256, 512, down=True),
            ResDisBlock(512, 1024, down=True),
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

    def forward(self, input):
        result = self.model(input)
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
            OptimizedResDisblock(inchannels, 64),
            ResDisBlock(64, 128, down=True),
            ResDisBlock(128, 256, down=True),
            ResDisBlock(256, 512, down=True),
            ResDisBlock(512, 1024, down=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(1024, latent_dim)

        res_arch_init(self)

    def forward(self, input):
        result = self.model(input)
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