import torch
import torchvision.utils as vutils
import os
import numpy as np
from config import device
def save_grid_img(imgs, save_root_path, epoch):
    out_file = save_root_path + '/' + str(epoch) + '_result.png'
    x_concat = torch.cat(imgs, dim=-2)
    vutils.save_image(x_concat.data, out_file, nrow=14, normalize=True, range=(-1., 1.))


def path_isexists(opts):
    if not os.path.exists(opts.train_save_path):
        os.makedirs(opts.train_save_path)
    if not os.path.exists(opts.val_save_path):
        os.makedirs(opts.val_save_path)
    if not os.path.exists(opts.weight_save_path):
        os.makedirs(opts.weight_save_path)

def Y_mask(opts):
    mask = np.ones((1, 3, opts.img_size, opts.img_size))
    mask[:, 0, :, :] = 0
    mask = torch.Tensor(mask).to(device)
    mask.requires_grad = False
    return mask

def noise_clamp(adv_noise, img_size, sa_mask):
    adv_noise[:, :, :img_size // 2, :img_size // 2] = adv_noise[:, :, :img_size // 2, :img_size // 2].clone() * sa_mask
    adv_noise[:, :, :img_size // 2, img_size // 2:] = adv_noise[:, :, :img_size // 2, img_size // 2:].clone() * sa_mask
    adv_noise[:, :, img_size // 2:, :img_size // 2] = adv_noise[:, :, img_size // 2:, :img_size // 2].clone() * sa_mask
    adv_noise[:, :, img_size // 2:, img_size // 2:] = adv_noise[:, :, img_size // 2:, img_size // 2:].clone() * sa_mask
    return adv_noise

def tensor2numpy(x):
    img = (x[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                      torch.uint8).numpy()
    return img

def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)