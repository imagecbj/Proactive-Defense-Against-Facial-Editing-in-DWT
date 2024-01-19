import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from config import device
def prepare_lpips():
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model2 = lpips.LPIPS(net='vgg').to(device)
    return lpips_model, lpips_model2

def compute_metrics(x_ori, x_adv, lpips_model, lpips_model2):
    img_ori = (x_ori[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img_adv = (x_adv[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    psnr_value = psnr(img_ori, img_adv)
    ssim_value = ssim(img_ori, img_adv, channel_axis=2)
    lpips_alex = lpips_model(x_ori, x_adv).item()
    lpips_vgg = lpips_model2(x_ori, x_adv).item()
    return psnr_value, ssim_value, lpips_alex, lpips_vgg


def compute_psnr(x_ori, x_adv):
    img_ori = (x_ori[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img_adv = (x_adv[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    psnr_value = psnr(img_ori, img_adv)
    return psnr_value
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def get_perceptual_loss(vgg, input, adv):
    # get vgg features
    x_features = vgg(input)
    adv_features = vgg(adv)
    # calculate style loss
    loss_mse = torch.nn.MSELoss()
    x_gram = [gram(fmap) for fmap in x_features]
    adv_gram = [gram(fmap) for fmap in adv_features]
    style_loss = 0.0
    for j in range(4):
        style_loss += loss_mse(x_gram[j], adv_gram[j])
    style_loss = style_loss

    # calculate content loss (h_relu_2_2)
    xcon = x_features[1]
    acon = adv_features[1]
    content_loss = loss_mse(xcon, acon)
    return style_loss, content_loss