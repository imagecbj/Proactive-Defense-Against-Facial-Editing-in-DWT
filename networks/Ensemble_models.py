import torch
import torch.nn as nn
from networks.FGAN import Generator as FGAN_Model
from networks.AttGAN.attgan import AttGAN
from networks.AttGAN.utils import find_model

from networks.HiSD.inference import prepare_HiSD

from networks.AttGAN.data import check_attribute_conflict



import config
import json
import argparse
from os.path import join

def init_attGAN(args_attack):
    with open(join(config.AttGAN_base_path, 'setting.txt'), 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    args.test_int = args_attack.AttGAN.attgan_test_int
    args.num_test = args_attack.global_settings.num_test
    args.gpu = args_attack.global_settings.gpu
    args.load_epoch = args_attack.AttGAN.attgan_load_epoch
    args.multi_gpu = args_attack.AttGAN.attgan_multi_gpu
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)

    attgan.load(find_model(config.AttGAN_path, args.load_epoch))
    attgan.eval()
    return attgan, args
def parse(args=None):
    with open(join('./networks/AttGAN/setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

def norm_feature(F1, F2, F3):
    l2_norm1 = torch.norm(F1)
    l2_norm2 = torch.norm(F2)
    l2_norm3 = torch.norm(F3)
    F1_norm = F1 / l2_norm1
    F2_norm = F2 / l2_norm2
    F3_norm = F3 / l2_norm3
    return F1_norm, F2_norm, F3_norm

def fusion_model(F1, F2, F3, weights):
    return weights[0] * F1 + weights[1] * F2 + weights[2] * F3

class Ensemble(nn.Module):
    def __init__(self, opts):
        super(Ensemble, self).__init__()
        self.conv_dim = config.conv_dim
        self.c_dim = config.c_dim
        self.repeat_num = config.repeat_num
        self.path = config.FGAN_path
        self.mid_layer = config.mid_layer
        self.selected_attrs = opts.selected_attrs
        self.args_attack = parse()
        # ensemble networks  fgan\attgan\hisd
        # load fgan
        self.fgan = FGAN_Model(self.conv_dim, self.c_dim, self.repeat_num).to(config.device)
        self.load_weights()
        # load attgan
        self.attgan, self.attgan_args = init_attGAN(self.args_attack)
        self.attgan.G.eval()
        ## load hisd
        self.transform, self.F_, self.T, self.G, self.E, self.M, self.reference_glass, self.reference_black, self.reference_blond, self.reference_brown, self.reference_bangs, self.gen_models = prepare_HiSD()
        self.w = torch.randn(1, 32).to(config.device)
        self.selected_attrs_index = [2, 3, 4, 7, 12]
        self.conv2 = torch.nn.Conv2d(1024, 256, 1).to(config.device)
        self.conv2.requires_grad_(False)
    def load_weights(self):
        ckpt = torch.load(self.path, map_location=lambda storage, loc: storage)
        self.fgan.load_state_dict(ckpt)

    def create_labels(self, c_org):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.

        hair_color_indices = []
        for i, attr_name in enumerate(self.selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(self.c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(config.device))
        return c_trg_list

    def create_labels_attgan(self, att_a):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        att_b_list = []
        for i in range(self.attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, self.attgan_args.attrs[i], self.attgan_args.attrs)
            att_b_list.append(tmp.to(config.device))
        return att_b_list

    def fgan_outs(self, x_real, c_trg_list):
        outs = [x_real.data]
        with torch.no_grad():
            for i, c_trg in enumerate(c_trg_list):
                gen_out = torch.tanh(x_real + self.fgan(x_real, c_trg)[0])
                outs.append(gen_out.data)
        return outs
    def attgan_outs(self, x_real, b_trg_list):
        outs = [x_real.data]
        with torch.no_grad():
            for i, att_b in enumerate(b_trg_list):
                if i in self.selected_attrs_index:
                    att_b_ = (att_b * 2 - 1) * self.attgan_args.thres_int
                    att_b_[..., i] = att_b_[..., i] * self.attgan_args.test_int / self.attgan_args.thres_int
                    gen_noattack = self.attgan.G(x_real, att_b_)
                    outs.append(gen_noattack.data)
        return outs
    def hisd_outs(self, x_real):
        outs = [x_real.data]
        c_ori = self.E(x_real)
        s_trg_glass = self.F_(self.reference_glass, 1)
        c_trg_ori = self.T(c_ori, s_trg_glass, 1)
        gen_noattack = self.G(c_trg_ori)
        outs.append(gen_noattack.data)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_out = self.G(c_trg_ori)
        outs.append(ori_out.data)
        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_out = self.G(c_trg_ori)
        outs.append(ori_out.data)
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_out = self.G(c_trg_ori)
        outs.append(ori_out.data)

        s_trg_bands = self.M(self.w, 0, 0)
        c_trg_ori = self.T(c_ori, s_trg_bands, 0)
        ori_out = self.G(c_trg_ori)
        outs.append(ori_out.data)
        return outs
    def ensemble_models_out(self, x_real, c_trg_list, b_trg_list):
        outs_fgan = [x_real.data]
        outs_attgan = [x_real.data]
        outs_hisd = [x_real.data]

        with torch.no_grad():
            for i, c_trg in enumerate(c_trg_list):
                gen_noattack = torch.tanh(x_real + self.fgan(x_real, c_trg)[0])
                outs_fgan.append(gen_noattack.data)
            for i, att_b in enumerate(b_trg_list):
                if i in self.selected_attrs_index:
                    att_b_ = (att_b * 2 - 1) * self.attgan_args.thres_int
                    att_b_[..., i] = att_b_[..., i] * self.attgan_args.test_int / self.attgan_args.thres_int
                    gen_noattack = self.attgan.G(x_real, att_b_)
                    outs_attgan.append(gen_noattack.data)
            c_ori = self.E(x_real)
            s_trg_glass = self.F_(self.reference_glass, 1)
            c_trg_ori = self.T(c_ori, s_trg_glass, 1)
            gen_noattack = self.G(c_trg_ori)
            outs_hisd.append(gen_noattack.data)

            s_trg = self.F_(self.reference_black, 2)
            c_trg_ori = self.T(c_ori, s_trg, 2)
            ori_out = self.G(c_trg_ori)
            outs_hisd.append(ori_out.data)
            s_trg = self.F_(self.reference_blond, 2)
            c_trg_ori = self.T(c_ori, s_trg, 2)
            ori_out = self.G(c_trg_ori)
            outs_hisd.append(ori_out.data)
            s_trg = self.F_(self.reference_brown, 2)
            c_trg_ori = self.T(c_ori, s_trg, 2)
            ori_out = self.G(c_trg_ori)
            outs_hisd.append(ori_out.data)

            s_trg_bands = self.M(self.w, 0, 0)
            c_trg_ori = self.T(c_ori, s_trg_bands, 0)
            ori_out = self.G(c_trg_ori)
            outs_hisd.append(ori_out.data)
        return outs_fgan, outs_attgan, outs_hisd

    def compute_loss(self, x_adv, x_real, c_trg_list):
        loss_out = 0.0
        criterion = torch.nn.MSELoss()

        feature_ori = torch.zeros((1, 256, 64, 64)).to(config.device)
        feature_adv = torch.zeros((1, 256, 64, 64)).to(config.device)
        for i, c_trg in enumerate(c_trg_list):
            ori_out, ori_mids = self.SM(x_real, c_trg)
            adv_out, adv_mids = self.SM(x_adv, c_trg)

            ori_mid = ori_mids[self.mid_layer]
            adv_mid = adv_mids[self.mid_layer]

            feature_ori += ori_mid
            feature_adv += adv_mid

            loss_out += (-criterion(ori_out, adv_out))

        feature_ori /= self.c_dim
        feature_adv /= self.c_dim

        loss_mid = -criterion(feature_ori, feature_adv)
        loss = loss_mid + loss_out
        return loss

    def ensemble_compute_loss(self, x_adv, x_real, c_trg_list, weights_softmax):
        criterion = torch.nn.MSELoss()

        # fgan
        feature_ori = torch.zeros((1, 256, 64, 64)).to(config.device)
        feature_adv = torch.zeros((1, 256, 64, 64)).to(config.device)
        for i, c_trg in enumerate(c_trg_list):
            _, ori_mids = self.fgan(x_real, c_trg)
            _, adv_mids = self.fgan(x_adv, c_trg)

            ori_mid = ori_mids[self.mid_layer]
            adv_mid = adv_mids[self.mid_layer]

            feature_ori += ori_mid
            feature_adv += adv_mid

        feature_ori /= self.c_dim
        feature_adv /= self.c_dim

        ori_mid0 = feature_ori
        adv_mid0 = feature_adv

        # attgan
        feature_ori1 = self.attgan.G(x_real, mode='enc')[-1]
        feature_adv1 = self.attgan.G(x_adv, mode='enc')[-1]
        ori_mid1 = torch.nn.functional.interpolate(self.conv2(feature_ori1), size=(64, 64), mode='bilinear',
                                                   align_corners=False)
        adv_mid1 = torch.nn.functional.interpolate(self.conv2(feature_adv1), size=(64, 64), mode='bilinear',
                                                   align_corners=False)

        # hisd
        c_ori = self.E(x_real)
        s_trg_glass = self.F_(self.reference_glass, 1)  # glass
        ori_mids_1 = self.T(c_ori, s_trg_glass, 1)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_2 = c_trg_ori
        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_3 = c_trg_ori
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_4 = c_trg_ori

        s_trg_bands = self.M(self.w, 0, 0)  # liuhai
        ori_mids_5 = self.T(c_ori, s_trg_bands, 0)
        ori_mids = (ori_mids_1 + ori_mids_2 + ori_mids_3 + ori_mids_4 + ori_mids_5) / 5.
        # hisd adv1
        c_adv1 = self.E(x_adv)
        s_trg_glass = self.F_(self.reference_glass, 1)
        adv_mids_1 = self.T(c_adv1, s_trg_glass, 1)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_2 = c_trg_ori
        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_3 = c_trg_ori
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_4 = c_trg_ori

        s_trg_bands = self.M(self.w, 0, 0)
        adv_mids_5 = self.T(c_adv1, s_trg_bands, 0)
        adv_mids = (adv_mids_1 + adv_mids_2 + adv_mids_3 + adv_mids_4 + adv_mids_5) / 5.

        ori_mid2 = ori_mids
        adv_mid2 = adv_mids


        ori_norm1, ori_norm2, ori_norm3 = norm_feature(ori_mid0, ori_mid1, ori_mid2)
        adv_norm1, adv_norm2, adv_norm3 = norm_feature(adv_mid0, adv_mid1, adv_mid2)

        feature_ori = fusion_model(ori_norm1, ori_norm2, ori_norm3, weights_softmax)
        feature_adv = fusion_model(adv_norm1, adv_norm2, adv_norm3, weights_softmax)

        loss = -criterion(feature_ori, feature_adv) * 10000000

        return loss
    def ensemble_compute_loss_nomerge(self, x_adv, x_real, c_trg_list, weights_softmax):
        criterion = torch.nn.MSELoss()

        loss_fgan, loss_attgan, loss_hisd = 0.0, 0.0, 0.0
        # fgan
        feature_ori = torch.zeros((1, 256, 64, 64)).to(config.device)
        feature_adv = torch.zeros((1, 256, 64, 64)).to(config.device)
        for i, c_trg in enumerate(c_trg_list):
            _, ori_mids = self.fgan(x_real, c_trg)
            _, adv_mids = self.fgan(x_adv, c_trg)

            ori_mid = ori_mids[self.mid_layer]
            adv_mid = adv_mids[self.mid_layer]

            loss_fgan += (-criterion(ori_mid, adv_mid))

            # feature_ori += ori_mid
            # feature_adv += adv_mid

        # feature_ori /= self.c_dim
        # feature_adv /= self.c_dim

        # ori_mid0 = feature_ori
        # adv_mid0 = feature_adv
        loss_fgan /= 5
        loss_fgan *= weights_softmax[0]

        # attgan
        feature_ori1 = self.attgan.G(x_real, mode='enc')[-1]
        feature_adv1 = self.attgan.G(x_adv, mode='enc')[-1]
        loss_attgan = -criterion(feature_ori1, feature_adv1)
        loss_attgan *= weights_softmax[1]
        # ori_mid1 = torch.nn.functional.interpolate(self.conv2(feature_ori1), size=(64, 64), mode='bilinear',
        #                                            align_corners=False)
        # adv_mid1 = torch.nn.functional.interpolate(self.conv2(feature_adv1), size=(64, 64), mode='bilinear',
        #                                            align_corners=False)

        # hisd
        c_ori = self.E(x_real)
        s_trg_glass = self.F_(self.reference_glass, 1)  # glass
        ori_mids_1 = self.T(c_ori, s_trg_glass, 1)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_2 = c_trg_ori
        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_3 = c_trg_ori
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_4 = c_trg_ori

        s_trg_bands = self.M(self.w, 0, 0)  # liuhai
        ori_mids_5 = self.T(c_ori, s_trg_bands, 0)
        ori_mids = (ori_mids_1 + ori_mids_2 + ori_mids_3 + ori_mids_4 + ori_mids_5) / 5.
        # hisd adv1
        c_adv1 = self.E(x_adv)
        s_trg_glass = self.F_(self.reference_glass, 1)
        adv_mids_1 = self.T(c_adv1, s_trg_glass, 1)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_2 = c_trg_ori
        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_3 = c_trg_ori
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_4 = c_trg_ori

        s_trg_bands = self.M(self.w, 0, 0)
        adv_mids_5 = self.T(c_adv1, s_trg_bands, 0)
        adv_mids = (adv_mids_1 + adv_mids_2 + adv_mids_3 + adv_mids_4 + adv_mids_5) / 5.

        ori_mid2 = ori_mids
        adv_mid2 = adv_mids

        loss_hisd = -criterion(ori_mid2, adv_mid2)
        loss_hisd *= weights_softmax[2]

        # ori_norm1, ori_norm2, ori_norm3 = norm_feature(ori_mid0, ori_mid1, ori_mid2)
        # adv_norm1, adv_norm2, adv_norm3 = norm_feature(adv_mid0, adv_mid1, adv_mid2)
        #
        # feature_ori = fusion_model(ori_norm1, ori_norm2, ori_norm3, weights_softmax)
        # feature_adv = fusion_model(adv_norm1, adv_norm2, adv_norm3, weights_softmax)

        loss = (loss_fgan + loss_attgan + loss_hisd) * 10000000

        return loss


    def ensemble_compute_loss_nofea(self, x_adv, x_real, c_trg_list, b_trg_list, weights_softmax):
        criterion = torch.nn.MSELoss()

        loss_fgan, loss_attgan, loss_hisd = 0.0, 0.0, 0.0

        # fgan
        feature_ori = torch.zeros((1, 256, 64, 64)).to(config.device)
        feature_adv = torch.zeros((1, 256, 64, 64)).to(config.device)
        for i, c_trg in enumerate(c_trg_list):
            ori_out_fgan, ori_mids = self.fgan(x_real, c_trg)
            adv_out_fgan, adv_mids = self.fgan(x_adv, c_trg)

            ori_mid = ori_mids[self.mid_layer]
            adv_mid = adv_mids[self.mid_layer]

            feature_ori += ori_mid
            feature_adv += adv_mid

            # compute end-to-end loss
            loss_fgan += (-criterion(ori_out_fgan, adv_out_fgan))

        feature_ori /= self.c_dim
        feature_adv /= self.c_dim

        ori_mid0 = feature_ori
        adv_mid0 = feature_adv

        loss_fgan *= weights_softmax[0]

        # attgan
        # attgan feature layer compute
        feature_ori1 = self.attgan.G(x_real, mode='enc')[-1]
        feature_adv1 = self.attgan.G(x_adv, mode='enc')[-1]
        ori_mid1 = torch.nn.functional.interpolate(self.conv2(feature_ori1), size=(64, 64), mode='bilinear',
                                                   align_corners=False)
        adv_mid1 = torch.nn.functional.interpolate(self.conv2(feature_adv1), size=(64, 64), mode='bilinear',
                                                   align_corners=False)

        # attgan end-to-end compute
        for i, att_b in enumerate(b_trg_list):
            if i in self.selected_attrs_index:
                att_b_ = (att_b * 2 - 1) * self.attgan_args.thres_int
                att_b_[..., i] = att_b_[..., i] * self.attgan_args.test_int / self.attgan_args.thres_int
                gen_noattack = self.attgan.G(x_real, att_b_)
                gen = self.attgan.G(x_adv, att_b_)
                loss_attgan += (-criterion(gen_noattack, gen))

        loss_attgan *= weights_softmax[1]


        # hisd
        # hisd
        c_ori = self.E(x_real)
        s_trg_glass = self.F_(self.reference_glass, 1)  # glass
        ori_mids_1 = self.T(c_ori, s_trg_glass, 1)
        gen_noattack1 = self.G(ori_mids_1)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_2 = c_trg_ori
        gen_noattack2 = self.G(ori_mids_2)

        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_3 = c_trg_ori
        gen_noattack3 = self.G(ori_mids_3)
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_4 = c_trg_ori
        gen_noattack4 = self.G(ori_mids_4)

        s_trg_bands = self.M(self.w, 0, 0)  # liuhai
        ori_mids_5 = self.T(c_ori, s_trg_bands, 0)
        gen_noattack5 = self.G(ori_mids_5)
        ori_mids = (ori_mids_1 + ori_mids_2 + ori_mids_3 + ori_mids_4 + ori_mids_5) / 5.
        # hisd adv1
        c_adv1 = self.E(x_adv)
        s_trg_glass = self.F_(self.reference_glass, 1)
        adv_mids_1 = self.T(c_adv1, s_trg_glass, 1)
        gen1 = self.G(adv_mids_1)
        loss_hisd += (-criterion(gen_noattack1, gen1))

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_2 = c_trg_ori
        gen2 = self.G(adv_mids_2)
        loss_hisd += (-criterion(gen_noattack2, gen2))

        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_3 = c_trg_ori
        gen3 = self.G(adv_mids_3)
        loss_hisd += (-criterion(gen_noattack3, gen3))

        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_4 = c_trg_ori
        gen4 = self.G(adv_mids_4)
        loss_hisd += (-criterion(gen_noattack4, gen4))

        s_trg_bands = self.M(self.w, 0, 0)
        adv_mids_5 = self.T(c_adv1, s_trg_bands, 0)
        gen5 = self.G(adv_mids_5)
        loss_hisd += (-criterion(gen_noattack5, gen5))

        loss_hisd *= weights_softmax[2]

        adv_mids = (adv_mids_1 + adv_mids_2 + adv_mids_3 + adv_mids_4 + adv_mids_5) / 5.

        ori_mid2 = ori_mids
        adv_mid2 = adv_mids

        ori_norm1, ori_norm2, ori_norm3 = norm_feature(ori_mid0, ori_mid1, ori_mid2)
        adv_norm1, adv_norm2, adv_norm3 = norm_feature(adv_mid0, adv_mid1, adv_mid2)

        feature_ori = fusion_model(ori_norm1, ori_norm2, ori_norm3, weights_softmax)
        feature_adv = fusion_model(adv_norm1, adv_norm2, adv_norm3, weights_softmax)

        loss_total = loss_fgan + loss_attgan + loss_hisd
        loss = -criterion(feature_ori, feature_adv) * 10000000 + loss_total

        return loss


    def ensemble_compute_loss_all_fea(self, x_adv, x_real, c_trg_list, b_trg_list, weights_softmax):
        criterion = torch.nn.MSELoss()

        loss_fgan, loss_attgan, loss_hisd = 0.0, 0.0, 0.0

        # fgan
        feature_ori = torch.zeros((1, 256, 64, 64)).to(config.device)
        feature_adv = torch.zeros((1, 256, 64, 64)).to(config.device)
        for i, c_trg in enumerate(c_trg_list):
            ori_out_fgan, ori_mids = self.fgan(x_real, c_trg)
            adv_out_fgan, adv_mids = self.fgan(x_adv, c_trg)

            ori_mid = ori_mids[self.mid_layer]
            adv_mid = adv_mids[self.mid_layer]

            feature_ori += ori_mid
            feature_adv += adv_mid

            # compute end-to-end loss
            loss_fgan += (-criterion(ori_out_fgan, adv_out_fgan))

        feature_ori /= self.c_dim
        feature_adv /= self.c_dim

        ori_mid0 = feature_ori
        adv_mid0 = feature_adv

        loss_fgan += (-criterion(ori_mid0, adv_mid0)) * 1000000
        loss_fgan *= weights_softmax[0]

        # attgan
        # attgan feature layer compute
        feature_ori1 = self.attgan.G(x_real, mode='enc')[-1]
        feature_adv1 = self.attgan.G(x_adv, mode='enc')[-1]
        # ori_mid1 = torch.nn.functional.interpolate(self.conv2(feature_ori1), size=(64, 64), mode='bilinear',
        #                                            align_corners=False)
        # adv_mid1 = torch.nn.functional.interpolate(self.conv2(feature_adv1), size=(64, 64), mode='bilinear',
        #                                            align_corners=False)

        # attgan end-to-end compute
        for i, att_b in enumerate(b_trg_list):
            if i in self.selected_attrs_index:
                att_b_ = (att_b * 2 - 1) * self.attgan_args.thres_int
                att_b_[..., i] = att_b_[..., i] * self.attgan_args.test_int / self.attgan_args.thres_int
                gen_noattack = self.attgan.G(x_real, att_b_)
                gen = self.attgan.G(x_adv, att_b_)
                loss_attgan += (-criterion(gen_noattack, gen))

        loss_attgan += (-criterion(feature_ori1, feature_adv1)) * 1000000
        loss_attgan *= weights_softmax[1]


        # hisd
        # hisd
        c_ori = self.E(x_real)
        s_trg_glass = self.F_(self.reference_glass, 1)  # glass
        ori_mids_1 = self.T(c_ori, s_trg_glass, 1)
        gen_noattack1 = self.G(ori_mids_1)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_2 = c_trg_ori
        gen_noattack2 = self.G(ori_mids_2)

        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_3 = c_trg_ori
        gen_noattack3 = self.G(ori_mids_3)
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_4 = c_trg_ori
        gen_noattack4 = self.G(ori_mids_4)

        s_trg_bands = self.M(self.w, 0, 0)  # liuhai
        ori_mids_5 = self.T(c_ori, s_trg_bands, 0)
        gen_noattack5 = self.G(ori_mids_5)
        ori_mids = (ori_mids_1 + ori_mids_2 + ori_mids_3 + ori_mids_4 + ori_mids_5) / 5.
        # hisd adv1
        c_adv1 = self.E(x_adv)
        s_trg_glass = self.F_(self.reference_glass, 1)
        adv_mids_1 = self.T(c_adv1, s_trg_glass, 1)
        gen1 = self.G(adv_mids_1)
        loss_hisd += (-criterion(gen_noattack1, gen1))

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_2 = c_trg_ori
        gen2 = self.G(adv_mids_2)
        loss_hisd += (-criterion(gen_noattack2, gen2))

        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_3 = c_trg_ori
        gen3 = self.G(adv_mids_3)
        loss_hisd += (-criterion(gen_noattack3, gen3))

        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_4 = c_trg_ori
        gen4 = self.G(adv_mids_4)
        loss_hisd += (-criterion(gen_noattack4, gen4))

        s_trg_bands = self.M(self.w, 0, 0)
        adv_mids_5 = self.T(c_adv1, s_trg_bands, 0)
        gen5 = self.G(adv_mids_5)
        loss_hisd += (-criterion(gen_noattack5, gen5))



        adv_mids = (adv_mids_1 + adv_mids_2 + adv_mids_3 + adv_mids_4 + adv_mids_5) / 5.

        ori_mid2 = ori_mids
        adv_mid2 = adv_mids

        loss_hisd += (-criterion(adv_mid2, ori_mid2)) * 1000000
        loss_hisd *= weights_softmax[2]

        # ori_norm1, ori_norm2, ori_norm3 = norm_feature(ori_mid0, ori_mid1, ori_mid2)
        # adv_norm1, adv_norm2, adv_norm3 = norm_feature(adv_mid0, adv_mid1, adv_mid2)
        #
        # feature_ori = fusion_model(ori_norm1, ori_norm2, ori_norm3, weights_softmax)
        # feature_adv = fusion_model(adv_norm1, adv_norm2, adv_norm3, weights_softmax)

        loss = loss_fgan + loss_attgan + loss_hisd
        # loss = -criterion(feature_ori, feature_adv) * 10000000 + loss_total

        return loss

    def ensemble_compute_loss_all(self, x_adv, x_real, c_trg_list, b_trg_list, weights_softmax):
        criterion = torch.nn.MSELoss()

        loss_fgan, loss_attgan, loss_hisd = 0.0, 0.0, 0.0

        # fgan

        for i, c_trg in enumerate(c_trg_list):
            ori_out_fgan, ori_mids = self.fgan(x_real, c_trg)
            adv_out_fgan, adv_mids = self.fgan(x_adv, c_trg)

            # compute end-to-end loss
            loss_fgan += (-criterion(ori_out_fgan, adv_out_fgan))


        loss_fgan *= weights_softmax[0]

        # attgan
        # attgan feature layer compute

        # attgan end-to-end compute
        for i, att_b in enumerate(b_trg_list):
            if i in self.selected_attrs_index:
                att_b_ = (att_b * 2 - 1) * self.attgan_args.thres_int
                att_b_[..., i] = att_b_[..., i] * self.attgan_args.test_int / self.attgan_args.thres_int
                gen_noattack = self.attgan.G(x_real, att_b_)
                gen = self.attgan.G(x_adv, att_b_)
                loss_attgan += (-criterion(gen_noattack, gen))


        loss_attgan *= weights_softmax[1]


        # hisd
        # hisd
        c_ori = self.E(x_real)
        s_trg_glass = self.F_(self.reference_glass, 1)  # glass
        ori_mids_1 = self.T(c_ori, s_trg_glass, 1)
        gen_noattack1 = self.G(ori_mids_1)

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_2 = c_trg_ori
        gen_noattack2 = self.G(ori_mids_2)

        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_3 = c_trg_ori
        gen_noattack3 = self.G(ori_mids_3)
        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_ori, s_trg, 2)
        ori_mids_4 = c_trg_ori
        gen_noattack4 = self.G(ori_mids_4)

        s_trg_bands = self.M(self.w, 0, 0)  # liuhai
        ori_mids_5 = self.T(c_ori, s_trg_bands, 0)
        gen_noattack5 = self.G(ori_mids_5)
        # hisd adv1
        c_adv1 = self.E(x_adv)
        s_trg_glass = self.F_(self.reference_glass, 1)
        adv_mids_1 = self.T(c_adv1, s_trg_glass, 1)
        gen1 = self.G(adv_mids_1)
        loss_hisd += (-criterion(gen_noattack1, gen1))

        s_trg = self.F_(self.reference_black, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_2 = c_trg_ori
        gen2 = self.G(adv_mids_2)
        loss_hisd += (-criterion(gen_noattack2, gen2))

        s_trg = self.F_(self.reference_blond, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_3 = c_trg_ori
        gen3 = self.G(adv_mids_3)
        loss_hisd += (-criterion(gen_noattack3, gen3))

        s_trg = self.F_(self.reference_brown, 2)
        c_trg_ori = self.T(c_adv1, s_trg, 2)
        adv_mids_4 = c_trg_ori
        gen4 = self.G(adv_mids_4)
        loss_hisd += (-criterion(gen_noattack4, gen4))

        s_trg_bands = self.M(self.w, 0, 0)
        adv_mids_5 = self.T(c_adv1, s_trg_bands, 0)
        gen5 = self.G(adv_mids_5)
        loss_hisd += (-criterion(gen_noattack5, gen5))

        loss_hisd *= weights_softmax[2]

        # ori_norm1, ori_norm2, ori_norm3 = norm_feature(ori_mid0, ori_mid1, ori_mid2)
        # adv_norm1, adv_norm2, adv_norm3 = norm_feature(adv_mid0, adv_mid1, adv_mid2)
        #
        # feature_ori = fusion_model(ori_norm1, ori_norm2, ori_norm3, weights_softmax)
        # feature_adv = fusion_model(adv_norm1, adv_norm2, adv_norm3, weights_softmax)

        loss = loss_fgan + loss_attgan + loss_hisd
        # loss = -criterion(feature_ori, feature_adv) * 10000000 + loss_total

        return loss

    def SM_out(self, x, c_trg_list):
        outs = [x.data]
        with torch.no_grad():
            for i, c_trg in enumerate(c_trg_list):
                gen_out = torch.tanh(x + self.SM(x, c_trg)[0])
                outs.append(gen_out.data)
        return outs

