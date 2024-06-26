""" Basic configuration and settings for training the model"""
import torch
import os
# specify protection model architecture [resnet_9blocks, resnet_6blocks, resnet_2blocks, resnet_1blocks, unet_32, unet_64, unet_128, unet_256]
net_noise = 'unet_64'

# of gen filters in the last conv layer
ngf = 64

# instance normalization or batch normalization [instance | batch | none]
norm = 'instance'

# network initialization [normal | xavier | kaiming | orthogonal]
init_type = 'normal'

# scaling factor for normal, xavier and orthogonal.
init_gain = 0.02

no_dropout = False

input_nc = 3
output_nc = 3

# running device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SM
conv_dim = 64
c_dim = 5
repeat_num = 6
mid_layer = 8

# other config
max_psnr = 50.0  # PG weight save condition

# metrics
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0

# attrs
attrs = [
        "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Eyeglasses",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "No_Beard",
        "Pale_Skin",
        "Young"
    ]

# config the result path
train_save_path = "./Proactive_results/train" # Train Result Path
val_save_path = "./Proactive_results/val"    # Val Result Path
weight_save_path = "./Proactive_results/weight"  # PG Weights Save Path
logger_save_path = "./Proactive_results/"    # Logger Save Path

save_train_path_fgan = train_save_path + '/fgan'
save_val_path_fgan = val_save_path + '/fgan'
save_train_path_attgan = train_save_path + '/attgan'
save_val_path_attgan = val_save_path + '/attgan'
save_train_path_hisd = train_save_path + '/hisd'
save_val_path_hisd = val_save_path + '/hisd'

# mkdir
def path_isexists():
    if not os.path.exists(save_train_path_fgan):
        os.makedirs(save_train_path_fgan)
    if not os.path.exists(save_val_path_fgan):
        os.makedirs(save_val_path_fgan)
    if not os.path.exists(save_train_path_attgan):
        os.makedirs(save_train_path_attgan)
    if not os.path.exists(save_val_path_attgan):
        os.makedirs(save_val_path_attgan)
    if not os.path.exists(save_train_path_hisd):
        os.makedirs(save_train_path_hisd)
    if not os.path.exists(save_val_path_hisd):
        os.makedirs(save_val_path_hisd)
    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)

# deepfake model path
FGAN_path = "./checkpoints/FGAN/200000-G.ckpt"
HiSD_path = "./checkpoints/HiSD/gen_00600000.pt"
AttGAN_base_path = './networks/AttGAN'
AttGAN_path = './checkpoints/AttGAN'


# SA Model Path
mask_model_path = "./checkpoints/FAN/best-model_epoch-204_mae-0.0505_loss-0.1370.pth"