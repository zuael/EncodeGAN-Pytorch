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
attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", dest="test_size", type=int, default=100, help="size of the test_batches")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=cpu_count(),
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=bool, default=True, help="whether to use gpu")

    parser.add_argument('--weight_path', dest='weight_path', type=str, default=None)
    parser.add_argument('--setting_path', dest='setting_path', type=str, default=None)
    parser.add_argument('--data_save_root', dest='data_save_root', type=str, default='')
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data_path', dest='data_path', type=str, default='')
    parser.add_argument('--test_data_path', dest='test_data_path', type=str, default='')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='')
    parser.add_argument('--attrs_change_path', dest='attrs_change_path', type=str, default=None)
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
args.data_save_root = args_.data_save_root
args.data_path = args_.data_path
args.test_data_path = args_.test_data_path
args.attr_path = args_.attr_path
args.attrs_change_path = args_.attrs_change_path
args.experiment_name = args_.experiment_name


os.makedirs(join(args.data_save_root, args.experiment_name), exist_ok=True)
test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_size,
                             num_workers=args.num_workers,
                             shuffle=True, drop_last=False)

print('Test images:', len(test_dataset))
device = torch.device('cuda' if args.gpu else 'cpu')
cudnn.benchmark = True
print('load model')
vaegan = VAEGAN(args)

vaegan.load(os.path.join(args.weight_path))
vaegan.eval()

pbar = tqdm(total=args.imgs_num, dynamic_ncols=True, leave=False,
            disable=False, desc="Projection Images")

attrs_num = len(args.attrs)
attr_true = torch.zeros((attrs_num, args.latent_dim), dtype=float).to(device)
attr_False = torch.zeros((attrs_num, args.latent_dim), dtype=float).to(device)
attr_true_n = torch.zeros(attrs_num, 1).to(device)
attr_False_n = torch.zeros(attrs_num, 1).to(device)

if args.attrs_change_path is not None:
    attr_change = np.load(join(args.attrs_change_path))
    attr_change = torch.FloatTensor(attr_change).to(device)
else:
    for _ in range(args.imgs_num // args.test_size):
        image, label = next(iter(test_dataloader))

        image = image.to(device)
        label = label.to(device)

        latent = vaegan.E(image).detach()

        for i in range(label.shape[1]):
            mask = label[:,i].view(label.shape[0],-1)

            # update average latent
            attr_true[i] += (mask * latent).sum(dim=0)
            attr_true_n[i] += mask.sum()

            attr_False[i] += ((1 - mask) * latent).sum(dim=0)
            attr_False_n[i] += (1 - mask).sum()

        pbar.update(args.test_size)

    # get attr diff explore direction
    attr_true = attr_true / attr_true_n
    attr_False = attr_False / attr_False_n
    attr_change = (attr_true - attr_False).detach()

    np.save('change_direction.npy', np.array(attr_change.cpu()))

# get test image label
image_test_name = os.listdir(args.test_data_path)
att_list = open(args.attr_path, 'r', encoding='utf-8').readlines()[1].split()
atts = [att_list.index(att) + 1 for att in args.attrs]
images = np.loadtxt(args.attr_path, skiprows=2, usecols=[0], dtype=np.str)
labels = np.loadtxt(args.attr_path, skiprows=2, usecols=atts, dtype=np.int)
count = 0

label_test = np.zeros((len(image_test_name), attrs_num))

for i in range(len(images)):
    if images[i] in image_test_name:
        index = image_test_name.index(images[i])
        label_test[index] = labels[i]
        count += 1
    else:
        pass

    if count == len(image_test_name):
        break

tf = transforms.Compose([
    transforms.CenterCrop(170),
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image_test = torch.cat([tf(Image.open(join('test_data', image_test_name[i]))).unsqueeze(0)
                        for i in range(len(image_test_name))],dim=0).to(device)

label_test = torch.FloatTensor(label_test).to(device)

orignal_latent = vaegan.E(image_test.to(device))

for i in range(orignal_latent.shape[0]):
    flag = -label_test[i].view(label_test.shape[1], -1)
    change_latent = (orignal_latent[i] + flag * attr_change).detach()
    change_image = vaegan.G(torch.cat([orignal_latent[i].unsqueeze(0), change_latent], dim=0))
    samples = torch.cat([image_test[i].unsqueeze(0), change_image], dim=0)
    save_image(samples, join(args.data_save_root, args.experiment_name, image_test_name[i]),
               nrow=attrs_num, normalize=True, range=(-1., 1.))
