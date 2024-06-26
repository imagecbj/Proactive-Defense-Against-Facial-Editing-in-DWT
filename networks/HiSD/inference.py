# import packages
try:
    from core.utils import get_config
    from core.trainer import HiSD_Trainer
except:
    from networks.HiSD.core.utils import get_config
    from networks.HiSD.core.trainer import HiSD_Trainer
import argparse
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from config import device
# from configs.common_config import device
import numpy as np
import time
from config import HiSD_path

# use cpu by default

# device = 'cpu'

# load checkpoint

def prepare_HiSD():
    # device = 'cuda'
    config = get_config('./networks/HiSD/configs/celeba-hq_256.yaml')
    noise_dim = config['noise_dim']
    image_size = config['new_size']
    checkpoint = HiSD_path
    # checkpoint = 'HiSD/checkpoint_256_celeba-hq.pt'
    trainer = HiSD_Trainer(config)
    state_dict = torch.load(checkpoint)
    trainer.models.gen.load_state_dict(state_dict['gen_test'])
    Gen = trainer.models.gen
    Gen.to(device)
    Gen.eval()
    Gen.zero_grad()

    E = trainer.models.gen.encode
    T = trainer.models.gen.translate
    G = trainer.models.gen.decode
    M = trainer.models.gen.map
    F = trainer.models.gen.extract

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    reference_glass = './networks/HiSD/examples/reference_glasses_2.jpg'
    reference_black = './networks/HiSD/examples/reference_black_hair.jpg'
    reference_blond = './networks/HiSD/examples/reference_blond_hair.jpg'
    reference_brown = './networks/HiSD/examples/reference_brown_hair.jpg'
    reference_bangs = './networks/HiSD/examples/reference_brown_hair.jpg'
    reference_glass = transform(Image.open(reference_glass).convert('RGB')).unsqueeze(0).to(device)
    reference_black = transform(Image.open(reference_black).convert('RGB')).unsqueeze(0).to(device)
    reference_blond = transform(Image.open(reference_blond).convert('RGB')).unsqueeze(0).to(device)
    reference_brown = transform(Image.open(reference_brown).convert('RGB')).unsqueeze(0).to(device)
    reference_bangs = transform(Image.open(reference_bangs).convert('RGB')).unsqueeze(0).to(device)

    return transform, F, T, G, E,M, reference_glass, reference_black, reference_blond, reference_brown, reference_bangs, trainer.models.gen


# def inference_to_attack(x, transform, F, T, G, E, reference, gen):
#     attack = LinfPGDAttack()
#     with torch.no_grad():
#         c = E(x)
#         c_trg = c
#         s_trg = F(reference, 1)
#         c_trg = T(c_trg, s_trg, 1)
#         x_trg = G(c_trg)
#     attack.universal_perturb_HiSD(x.cuda(), transform, F, T, G, E, device, reference, x_trg, gen)






