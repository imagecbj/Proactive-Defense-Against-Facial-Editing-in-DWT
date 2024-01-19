import torch
import torch.nn as nn
import networks.DWT.Unet_common as common
import config
from torchvision import transforms
import cv2
import numpy as np

class DWT(nn.Module):
    def __init__(self, opts):
        super(DWT, self).__init__()
        self.dwt = common.DWT().to(config.device)
        self.iwt = common.IWT().to(config.device)
        self.dwt.eval()
        self.iwt.eval()
        self.img_size = opts.img_size

    def get_subbands(self, x_dwt):
        x_LL = x_dwt.narrow(1, 0, 3)  # [1,3,128,128]---LL
        x_HL = x_dwt.narrow(1, 3, 3)
        x_LH = x_dwt.narrow(1, 6, 3)
        x_HH = x_dwt.narrow(1, 9, 3)
        return x_LL, x_HL, x_LH, x_HH

    def dwt_to_whole(self, x_LL, x_HL, x_LH, x_HH):
        line1 = torch.cat((x_LL, x_HL), dim=-1)
        line2 = torch.cat((x_LH, x_HH), dim=-1)
        result = torch.cat((line1, line2), dim=-2)
        return result

    def whole_to_dwt(self, x_whole):
        tmp_size = self.img_size // 2
        LL = x_whole[:, :, :tmp_size, :tmp_size]
        HL = x_whole[:, :, :tmp_size, tmp_size:]
        LH = x_whole[:, :, tmp_size:, :tmp_size]
        HH = x_whole[:, :, tmp_size:, tmp_size:]
        result = torch.cat((LL, HL, LH, HH), dim=1)
        return result