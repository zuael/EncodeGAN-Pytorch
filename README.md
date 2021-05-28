# EncodeGAN-PyTorch

A PyTorch implementation of EncodeGAN

Inverting 13 attributes respectively. From left to right: _Input, Reconstruction, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young_

## Requirements

* Python 3.6
* PyTorch
* TensorboardX

```bash
pip install -r requirements.txt
```

* Dataset
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
    * [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in `data/CelebA/img/img_align_celeba/*.jpg`
    * [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in `data/CelebA/anno/list_attr_celeba.txt`
  * [test_data]:you can use CelebA's test dataset, or select some face image from CelebA and add them in `test_data/*.jpg`.
```text
data
├── CelebA
    ├── img
            ├── img_align_celeba
    ├── anno
            ├── list_attr_celeba.txt
├── test_data
    ├── *.jpg
```

## Usage

#### To train an EncodeGAN on CelebA 128x128

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py --gpu=True --data_save_root=output --experiment_name=Encode_GAN --total_steps=100000 --latent_dim=128 --batch_size=32 --b1=0 --b2=0.999 --data_path='data/CelebA/img/img_align_celeba' --attr_path='data/CelebA/anno/list_attr_celeba.txt' --data_save_root='output' --E_mode='enc' --n_e=2
```

#### To test EncodeGan in interpolation capabilities on image

```bash
python face_latent_space_explore_test.py --gpu=True --data_save_root=output --experiment_name=face_latent_space_explore_test --weight_path=output/Encode_GAN/checkpoint/weights.99999.pth --setting_path=output/Encode_GAN/setting.txt --test_data_path=test_data
```

#### To test EncodeGan's face attribute editing ability
```bash
CUDA_VISIBLE_DEVICES=0 \
python face_attr_change_test.py --gpu=True --data_save_root=output --experiment_name=face_attr_change_test --data_path=data/CelebA/img/img_align_celeba --attr_path=data/CelebA/anno/list_attr_celeba.txt --data_save_root=output --weight_path=output/Encode_GAN/checkpoint/weights.99999.pth --setting_path=output/Encode_GAN/setting.txt --test_data_path=test_data
```

#### To visualize training details

```bash
tensorboard \
--logdir output/your_experiment_name/summary
```