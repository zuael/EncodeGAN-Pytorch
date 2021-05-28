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
from torchvision import transforms
from os.path import join
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", dest="test_size", type=int, default=100, help="size of the test_batches")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=cpu_count(),
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=bool, default=False, help="whether to use gpu")

    parser.add_argument('--weight_path', dest='weight_path', type=str, default=None)
    parser.add_argument('--setting_path', dest='setting_path', type=str, default=None)
    parser.add_argument('--data_save_root', dest='data_save_root', type=str, default='')
    parser.add_argument('--test_data_path', dest='test_data_path', type=str, default='')
    parser.add_argument("--experiment_name", dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I-%M%p on %B %d_%Y"))
    return parser.parse_args()


args_ = parse()
print(args_)


with open(join(args_.setting_path), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_size = args_.test_size
args.n_samples = args_.n_samples
args.num_workers = args_.num_workers
args.gpu = args_.gpu
args.weight_path = args_.weight_path
args.test_data_path = args_.test_data_path
args.data_save_root = args_.data_save_root
args.experiment_name = args_.experiment_name

os.makedirs(join(args.data_save_root, args.experiment_name), exist_ok=True)

device = torch.device('cuda' if args.gpu else 'cpu')
print(device)
cudnn.benchmark = True
print('load model')
vaegan = VAEGAN(args)

vaegan.load(os.path.join(args.weight_path))
vaegan.eval()


# get test image
image_test_name = os.listdir(args.test_data_path)

tf = transforms.Compose([
    transforms.CenterCrop(170),
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


image_test = torch.cat([tf(Image.open(join('test_data', image_test_name[i]))).unsqueeze(0)
                        for i in range(len(image_test_name))], dim=0).to(device)


# Set the original latent and the target latent randomly
orignal_latent = vaegan.E(image_test.to(device))
idx = torch.randperm(image_test.shape[0])
image_target = image_test[idx].contiguous()
target_latent = orignal_latent[idx].contiguous()

if isinstance(orignal_latent, tuple):
    orignal_latent = orignal_latent[0]
    target_latent = target_latent[0]

for i in range(orignal_latent.shape[0]):
    samples = []
    samples.append(image_test[i].unsqueeze(0))
    diff_latent = (target_latent[i] - orignal_latent[i]) / 10
    for j in range(0,11):
        with torch.no_grad():
            tmp = vaegan.G((orignal_latent[i] + diff_latent * j).unsqueeze(0))
        samples.append(tmp)

    samples.append(image_target[i].unsqueeze(0))
    samples = torch.cat(samples, dim=0)
    save_image(samples, join(args.data_save_root, args.experiment_name, image_test_name[i]),
               nrow=13, normalize=True, range=(-1., 1.))
    print('{} is done!'.format(image_test_name[i]))