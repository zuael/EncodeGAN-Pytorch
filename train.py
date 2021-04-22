import argparse
import os
import json
import datetime

from data import CelebA
from multiprocessing import cpu_count
from vaegan import VAEGAN
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from os.path import join
from helpers import add_scalar_dict
from tqdm._tqdm import trange
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import torch
import random
import math
import numpy as np

attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infiniteloop(dataloader):
    while True:
        for x, _ in iter(dataloader):
            yield x


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=50000, help="number of epochs of training")
    parser.add_argument("--n_d", type=int, default=3, help="# of d updates per g update")
    parser.add_argument("--n_d", type=int, default=2, help="# of d updates per g update")
    parser.add_argument("--imgs_num", type=int, default=200000, help="length of dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--test_size", dest="test_size", type=int, default=50, help="size of the test_batches")

    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=cpu_count(),
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=bool, default=True, help="whether to use gpu")

    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument("--save_interval", type=int, default=2000, help="interval between weight save")
    parser.add_argument("--E_mode", type=str, default='enc')
    parser.add_argument("--lambda_gp", dest='lambda_gp', type=float, default=10.0)

    parser.add_argument('--is_resume', dest='is_resume', type=bool, default=False)
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--event_name', dest='event_name', type=str, default=None)
    parser.add_argument('--load_iter', dest='load_iter', type=int, default=0)
    parser.add_argument('--data_save_root', dest='data_save_root', type=str, default='')
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data_path', dest='data_path', type=str, default='')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='')
    parser.add_argument("--experiment_name", dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I-%M%p on %B %d_%Y"))
    return parser.parse_args()

args = parse()
print(args)

# make dirs
os.makedirs(join(args.data_save_root, args.experiment_name), exist_ok=True)
os.makedirs(join(args.data_save_root, args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join(args.data_save_root, args.experiment_name, 'sample_training'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'summary'), exist_ok=True)
ms_file_name = join('m1s1_np.npz')

with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.num_workers)

test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_size,
                             num_workers=args.num_workers,
                             shuffle=True, drop_last=False)

looper = infiniteloop(train_dataloader)
test_looper = infiniteloop(test_dataloader)

print('Training images:', len(train_dataset))
set_seed(args.seed)

cudnn.benchmark = True
print('load model')
vaegan = VAEGAN(args)

if args.is_resume:
    vaegan.load(os.path.join('output', args.experiment_name, 'checkpoint', 'weights.' + str(args.load_iter) + '.pth'))
    ea = event_accumulator.EventAccumulator(join('output', args.experiment_name, 'summary', args.event_name))
    ea.Reload()
    loss = ea.scalars.Items('E/loss')
    step = loss[-1][1]
else:
    step = 0

print('learning rate:{}'.format(vaegan.optimizer_E.param_groups[0]['lr']))
print("load_iteration:{}".format(step))
device = torch.device('cuda' if args.gpu else 'cpu')

fixed_image = next(test_looper).to(device)
fixed_noise = torch.randn((args.n_samples ** 2, args.latent_dim)).to(device)

writer = SummaryWriter(join(args.data_save_root, args.experiment_name, 'summary'))

with trange(step, args.total_steps, dynamic_ncols=True) as pbar:
    for it in pbar:
        writer.add_scalar('LR/learning_rate', vaegan.optimizer_E.param_groups[0]['lr'], it + 1)
        vaegan.train()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(args.n_d):
            imgs = next(looper)
            # Configure input
            imgs = imgs.to(device)
            errD = vaegan.train_D(real_imgs=imgs)
        add_scalar_dict(writer, errD, it + 1, 'D')

        # -----------------
        #  Train Generator
        # -----------------
        errG = vaegan.train_G(real_imgs=imgs)
        add_scalar_dict(writer, errG, it + 1, 'G')

        # -----------------
        #  Train encoder
        # -----------------
        for _ in range(args.n_e):
            imgs = next(looper)
            imgs = imgs.to(device)
            errE = vaegan.train_E(real_imgs=imgs)
        add_scalar_dict(writer, errE, it + 1, 'E')

        pbar.set_postfix(iter=it + 1, d_loss=errD['d_loss'], g_loss=errG['g_loss'], e_loss=errE['e_loss'])

        if (it + 1) % args.save_interval == 0:
            vaegan.save(os.path.join(
                args.data_save_root, args.experiment_name,
                'checkpoint', 'weights.{:d}.pth'.format(it)
            ))

        if (it + 1) % args.sample_interval == 0:
            vaegan.eval()
            with torch.no_grad():
                samples = torch.zeros(fixed_image.shape[0] * 2,
                                      args.channels, args.img_size, args.img_size)
                samples = samples.cuda() if args.gpu else samples
                if args.E_mode == 'auto':
                    _, latent, _ = vaegan.E(fixed_image)
                else:
                    latent = vaegan.E(fixed_image)
                recon_image = vaegan.G(z=latent)

                samples[range(0, fixed_image.shape[0] * 2, 2)] = fixed_image
                samples[range(1, fixed_image.shape[0] * 2, 2)] = recon_image
                save_image(
                    samples, join(args.data_save_root, args.experiment_name,
                                  'sample_training', 'recon_It:{:d}.jpg'.format(it + 1)),
                    nrow=round(math.sqrt(args.test_size * 2)), normalize=True, range=(-1., 1.))

                samples = vaegan.G(z=fixed_noise)
                save_image(
                    samples, join(args.data_save_root, args.experiment_name,
                                  'sample_training', 'noise_It:{:d}.jpg'.format(it + 1)),
                    nrow=round(math.sqrt(args.test_size * 2)), normalize=True, range=(-1., 1.))
        # set learning rate
        vaegan.step()

    writer.close()
