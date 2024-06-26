import torch
import torch.nn as nn
import config
from networks.StarGAN import Generator as StarG
from networks.AttentionGAN import Generator as AttentionG


class Models(nn.Module):
    def __init__(self, opts):
        super(Models, self).__init__()
        self.conv_dim = config.conv_dim
        self.c_dim = config.c_dim
        self.repeat_num = config.repeat_num
        self.mid_layer = config.mid_layer
        self.selected_attrs = opts.selected_attrs
        self.model_name = opts.model_choice
        if self.model_name == 'stargan':
            self.path = opts.StarG_path
            self.model = StarG(conv_dim=self.conv_dim, c_dim=self.c_dim, repeat_num=self.repeat_num).to(config.device)
        else:
            self.path = opts.AttentionG_path
            self.model = AttentionG(conv_dim=self.conv_dim, c_dim=self.c_dim, repeat_num=self.repeat_num).to(config.device)
        self.load_weights()
    def load_weights(self):
        ckpt = torch.load(self.path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(ckpt)

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
    def model_out(self, x, c_trg_list):
        outs = [x.data]
        with torch.no_grad():
            for i, c_trg in enumerate(c_trg_list):
                if self.model_name == 'stargan':
                    gen,_ = self.model(x, c_trg)
                else:
                    gen,_,_ = self.model(x, c_trg)
                outs.append(gen.data)
        return outs
    def compute_loss2(self, x_adv, x_real, c_trg_list):
        loss_out = 0.0
        criterion = torch.nn.MSELoss()

        feature_ori = torch.zeros((1, 256, 64, 64)).to(config.device)
        feature_adv = torch.zeros((1, 256, 64, 64)).to(config.device)
        for i, c_trg in enumerate(c_trg_list):
            ori_out, ori_mids, _ = self.model(x_real, c_trg)
            adv_out, adv_mids, _ = self.model(x_adv, c_trg)

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
