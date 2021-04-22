import torch.autograd as autograd
import torch.nn.functional as F
import torch
from torchsummary import summary
import torch.nn as nn
from autoencoder import Autoencorder_res, Encorder
from wgan_gp import ResGenerator, ResDiscriminator

encoder = {
    'auto': Autoencorder_res,
    'enc': Encorder
}

class VAEGAN(nn.Module):
    def __init__(self, args):
        super(VAEGAN, self).__init__()
        self.mode = args.mode
        self.gpu = args.gpu
        self.E_mode = args.E_mode
        self.lambda_gp = args.lambda_gp
        self.latent_dim = args.latent_dim
        self.kld_weight = args.batch_size / args.imgs_num

        device = torch.device('cuda' if args.gpu else 'cpu')
        # Initialize generator and discriminator
        self.G = ResGenerator(args.latent_dim).to(device)
        summary(self.G, [(args.latent_dim,)], batch_size=args.batch_size,
                device='cuda' if args.gpu else 'cpu')

        self.D = ResDiscriminator().to(device)
        summary(self.D, [(args.channels, args.img_size, args.img_size)],
                batch_size=args.batch_size, device='cuda' if args.gpu else 'cpu')

        self.E = encoder[args.E_mode](args.channels, args.latent_dim, args.img_size).to(device)
        summary(self.E, [(args.channels, args.img_size, args.img_size)],
                batch_size=args.batch_size, device='cuda' if args.gpu else 'cpu')

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_E = torch.optim.Adam(self.E.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        Lambda = lambda step: 1 - step / args.total_steps

        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=Lambda)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=Lambda)
        self.sched_E = torch.optim.lr_scheduler.LambdaLR(self.optimizer_E, lr_lambda=Lambda)

    def train_G_E(self, real_imgs):
        noise = torch.randn((real_imgs.shape[0], self.latent_dim), device=real_imgs.device)

        if self.E_mode == 'auto':
            latent, mu, log_var = self.E(real_imgs)
            recons_imgs = self.G(z=latent)
            recon_adv, recons_f = self.D(recons_imgs)
            _, real_f = self.D(real_imgs)

            gen_imgs = self.G(noise)
            gen_adv, _ = self.D(gen_imgs)

            G_loss = - 0.5 * gen_adv.mean() - 0.5 * recon_adv.mean()

            recons_loss = F.mse_loss(recons_f, real_f.detach())
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            E_loss = recons_loss + self.kld_weight * kld_loss

            loss = G_loss + E_loss

        elif self.E_mode == 'enc':
            gen_imgs = self.G(noise)
            latent = self.E(gen_imgs.detach())
            gen_adv, _ = self.D(gen_imgs)

            G_loss = 0.5 * gen_adv.mean()

            recons_loss = F.mse_loss(latent, noise)
            kld_loss = torch.mean(-0.5 * torch.sum(latent, dim=1), dim=0)
            E_loss = recons_loss + self.kld_weight * kld_loss

            loss = G_loss + E_loss
        else:
            pass

        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.optimizer_E.step()

        errG_E = {
            'loss': loss.item(),
            'g_loss': G_loss.item(),
            'e_loss': E_loss.item(),
            'reconstruction_Loss': recons_loss.item(),
            'kld': kld_loss.item()
        }

        return errG_E
    
    def train_D(self, real_imgs):
        noise = torch.randn((real_imgs.shape[0], self.latent_dim), device=real_imgs.device)
        recon_imgs = self.G(self.E(real_imgs)[0]).detach()
        gen_imgs = self.G(z=noise).detach()

        real_adv, _ = self.D(image=real_imgs)
        recon_adv, _ = self.D(image=recon_imgs)
        gen_adv, _ = self.D(image=gen_imgs)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter

            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        wd = real_adv.mean() - 0.5 * (gen_adv.mean() + recon_adv.mean())
        df_loss = -wd
        df_gp = 0.5 * (gradient_penalty(self.D, real_imgs, gen_imgs)
                       + gradient_penalty(self.D, real_imgs, recon_imgs))

        d_loss = df_loss + self.lambda_gp * df_gp

        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()

        errD = {
            'd_loss':d_loss.item(),
            'df_loss':df_loss.item(),
            'df_gp':df_gp.item()
        }

        return errD

    def step(self):
        self.sched_G.step()
        self.sched_D.step()
        self.sched_E.step()

    def train(self):
        self.G.train()
        self.D.train()
        self.E.train()

    def eval(self):
        self.G.eval()
        self.D.eval()
        self.E.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'E': self.E.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_E': self.optimizer_E.state_dict(),
            'sched_G': self.sched_G.state_dict(),
            'sched_D': self.sched_D.state_dict(),
            'sched_E': self.sched_E.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)

        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'E' in states:
            self.E.load_state_dict(states['E'])
        if 'optimizer_G' in states:
            self.optimizer_G.load_state_dict(states['optimizer_G'])
        if 'optimizer_D' in states:
            self.optimizer_D.load_state_dict(states['optimizer_D'])
        if 'optimizer_E' in states:
            self.optimizer_E.load_state_dict(states['optimizer_E'])
        if 'sched_G' in states:
            self.sched_G.load_state_dict(states['sched_G'])
        if 'sched_D' in states:
            self.sched_D.load_state_dict(states['sched_D'])
        if 'sched_E' in states:
            self.sched_E.load_state_dict(states['sched_E'])


    def save_G_E(self, path):
        states = {
            'G': self.G.state_dict(),
            'E': self.E.state_dict()
        }
        torch.save(states, path)