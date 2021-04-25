#auto encoder & gan
python train.py --gpu=True --data_save_root=output --experiment_name=4_22 --total_steps=50000 --latent_dim=128 --batch_size=32 --b1=0 --b2=0.9 --data_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/img/img_align_celeba' --attr_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/anno/list_attr_celeba.txt' --data_save_root='output'

#encoder gan
python train.py --gpu=True --data_save_root=output --experiment_name=enc_gan_4_22 --total_steps=50000 --latent_dim=128 --batch_size=32 --b1=0 --b2=0.9 --data_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/img/img_align_celeba' --attr_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/anno/list_attr_celeba.txt' --data_save_root='output' --E_mode='enc'

#encode detach
python train.py --gpu=True --data_save_root=output --experiment_name=enc_gan_detach_4_22 --total_steps=50000 --latent_dim=128 --batch_size=32 --b1=0 --b2=0.999 --data_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/img/img_align_celeba' --attr_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/anno/list_attr_celeba.txt' --data_save_root='output' --E_mode='enc' --n_e=2

#auto encode detach
python train.py --gpu=True --data_save_root=output --experiment_name=auto_enc_gan_detach_4_23 --total_steps=50000 --latent_dim=128 --batch_size=64 --b1=0 --b2=0.999 --data_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/img/img_align_celeba' --attr_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/anno/list_attr_celeba.txt' --data_save_root='output' --E_mode='auto' --n_e=2

#explore att direction
python face_latent_explore.py --gpu=True --data_save_root=output --experiment_name=face_attr_explore --data_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/img/img_align_celeba' --attr_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/anno/list_attr_celeba.txt' --data_save_root='output' --weight_path=output/auto_enc_gan_detach_4_23/checkpoint/weights.49999.pth --setting_path=output/auto_enc_gan_detach_4_23/setting.txt --test_data_path=test_data
python face_latent_explore.py --gpu=True --data_save_root=output --experiment_name=face_attr_explore --data_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/img/img_align_celeba' --attr_path='/home/ma-user/work/AttGAN-PyTorch/data/CelebA/anno/list_attr_celeba.txt' --data_save_root='output' --weight_path=output/auto_enc_gan_detach_4_23/checkpoint/weights.49999.pth --setting_path=output/auto_enc_gan_detach_4_23/setting.txt --test_data_path=test_data --attrs_change_path=change_direction.npy