""" Basic configuration and settings for training the model"""
import torch
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