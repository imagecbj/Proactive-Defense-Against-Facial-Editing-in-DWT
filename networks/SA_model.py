import torch
import torch.nn as nn
from networks.PFAN.model import SODModel
import config
from torchvision import transforms
import cv2
import numpy as np

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.path = config.mask_model_path
        self.SA = SODModel().to(config.device)
        self.load_weights()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    def load_weights(self):
        chkpt = torch.load(self.path, map_location=config.device)
        self.SA.load_state_dict(chkpt['model'])
        self.SA.eval()

    def pad_resize_image(self, inp_img, out_img=None, target_size=None):

        h, w, c = inp_img.shape
        size = max(h, w)

        padding_h = (size - h) // 2
        padding_w = (size - w) // 2

        if out_img is None:
            # For inference
            temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if target_size is not None:
                temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
            return temp_x
        else:
            # For training and testing
            temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            temp_y = cv2.copyMakeBorder(out_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # print(inp_img.shape, temp_x.shape, out_img.shape, temp_y.shape)

            if target_size is not None:
                temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
                temp_y = cv2.resize(temp_y, (target_size, target_size), interpolation=cv2.INTER_AREA)
            return temp_x, temp_y

    def compute_mask(self, img):
        x_ori = (img[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                         torch.uint8).numpy()
        # compute saliency mask
        img_np = self.pad_resize_image(x_ori, None, 256)
        img_tor = img_np.astype(np.float32)
        img_tor = img_tor / 255.0
        img_tor = np.transpose(img_tor, axes=(2, 0, 1))
        img_tor = torch.from_numpy(img_tor).float()
        img_tor = self.normalize(img_tor)

        img_tor = img_tor.unsqueeze(0)
        img_tor = img_tor.to(config.device)
        pred_masks, _ = self.SA(img_tor)
        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(128, 128), mode='bilinear',
                                                     align_corners=False)
        return pred_masks
    def compute_mask_dct(self, img):
        x_ori = (img[0] / 2 + 0.5).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                         torch.uint8).numpy()
        # compute saliency mask
        img_np = self.pad_resize_image(x_ori, None, 256)
        img_tor = img_np.astype(np.float32)
        img_tor = img_tor / 255.0
        img_tor = np.transpose(img_tor, axes=(2, 0, 1))
        img_tor = torch.from_numpy(img_tor).float()
        img_tor = self.normalize(img_tor)

        img_tor = img_tor.unsqueeze(0)
        img_tor = img_tor.to(config.device)
        pred_masks, _ = self.SA(img_tor)
        # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(128, 128), mode='bilinear',
        #                                              align_corners=False)
        return pred_masks
