import numpy as np
import torch
from config import device

def rgb2ycbcr_np(img):
    #image as np.float, within range [0,255]

    A = np.array([[0.2568, 0.5041, 0.0979], [-0.1482, -0.2910, 0.4392], [0.4392, -0.3678, -0.0714]])
    ycbcr = img.dot(A.T)
    ycbcr[:,:,[1,2]] += 128
    ycbcr[:,:,0] += 16
    return ycbcr
def ycbcr_to_tensor(img_ycc):
    img_ycc = img_ycc.transpose(2,0,1) / 255.
    img_ycc_tensor = torch.Tensor(img_ycc)
    return img_ycc_tensor.unsqueeze(0)
def ycbcr2rgb_np(img):
    invA = np.array([[1.1644, 0.0, 1.5960], [1.1644, -0.3918, -0.8130], [1.1644, 2.0172, 0.0] ])
    img = img.astype(np.float)
    img[:,:,[1,2]] -= 128
    img[:,:,0] -= 16
    rgb = img.dot(invA.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.around(rgb)
def ycbcr_to_rgb(img_ycc):
    img_ycc = img_ycc.squeeze(0)
    img_ycc = img_ycc.permute(1,2,0).contiguous().view(-1,3).float()
    invA = torch.tensor([[1.164, 1.164, 1.164],
                        [0, -0.392, 2.0172],
                        [1.5960, -0.8130, 0]])

    invb = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0])
    invA, invb = invA.to(device), invb.to(device)
    invA.requires_grad = False
    invb.requires_grad = False
    img_ycc = (img_ycc + invb).mm(invA)
    img_ycc = img_ycc.view(256, 256, 3)
    img_ycc = img_ycc.permute(2,0,1)
    img_ycc = img_ycc.unsqueeze(0)
    img_ycc = torch.clamp(img_ycc, min=0., max=1.)
    return img_ycc