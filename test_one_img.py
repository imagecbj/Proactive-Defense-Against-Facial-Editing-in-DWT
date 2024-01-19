
from argparse import ArgumentParser

from networks.PG_network import define_G as PG_Model
from config import no_dropout, init_type, init_gain, ngf, net_noise, norm, device, input_nc, output_nc, max_psnr


# fgan
from networks.SM_model import SM
from networks.SA_model import SA
from networks.DWT_model import DWT

from tools.color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from tools.tool import *
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('--SM_path', default="./checkpoints/200000-G.ckpt", type=str, help='SM Weight Path')
    config_parser.add_argument('--mask_model_path',
                               default="./checkpoints/FAN/best-model_epoch-204_mae-0.0505_loss-0.1370.pth", type=str,
                               help='Saliency Detection Model Weight Path')
    config_parser.add_argument('--PG_path', default="./checkpoints/PG.pth", type=str, help='PG Weight Path')
    config_parser.add_argument('--test_path',default='./test', type=str, help='Test Result Path')
    config_parser.add_argument('--test_img',default='test.jpg', type=str, help='Test Image Name')
    config_parser.add_argument('--eposilon', default=0.01, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], type=list, help='Attribute Selection')
    opts = config_parser.parse_args()

    print(opts)

    # prepare
    y_mask = Y_mask(opts)

    # net prepare
    SM_net = SM(opts)  # suggorate model
    SA_net = SA(opts)  # saliency detection model
    DWT_net = DWT(opts)  # dwt
    # PG load
    PG = PG_Model(input_nc, output_nc, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    checkpoint = torch.load(opts.PG_path)
    PG.load_state_dict(checkpoint['protection_net'])
    PG.to(device)
    PG.eval()


    img_path = os.path.join(opts.test_path, opts.test_img)
    img = Image.open(img_path)

    results = []
    tf = []
    tf.append(transforms.Resize(256))
    tf.append(transforms.ToTensor())
    tf.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = transforms.Compose(tf)

    img_tf = transform(img)
    img_t = img_tf.unsqueeze(0)
    x_real = img_t.to(device).clone().detach()
    x_ori = tensor2numpy(x_real)

    c_org = torch.Tensor([[0., 0., 0., 0., 1.]])
    c_trg_list = SM_net.create_labels(c_org)

    ori_outs = SM_net.SM_out(x_real, c_trg_list)

    # compute saliency mask
    sa_mask = SA_net.compute_mask(x_real)
    # convert RGB to YCbCr
    x_ycbcr = rgb2ycbcr_np(x_ori)
    x_y = ycbcr_to_tensor(x_ycbcr).cuda()
    x_dwt = DWT_net.dwt(x_y)
    x_LL, x_HL, x_LH, x_HH = DWT_net.get_subbands(x_dwt)
    reshape_img = DWT_net.dwt_to_whole(x_LL, x_HL, x_LH, x_HH)

    adv_noise = PG(reshape_img) * y_mask
    adv_noise = torch.clamp(adv_noise, -opts.eposilon, opts.eposilon)
    adv_noise = noise_clamp(adv_noise, opts.img_size, sa_mask)

    x_L_adv = reshape_img + adv_noise
    x_adv_dwt = DWT_net.whole_to_dwt(x_L_adv)
    x_adv = ycbcr_to_rgb(DWT_net.iwt(x_adv_dwt))
    adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())
    adv_outs = SM_net.SM_out(adv_A, c_trg_list)

    results.append(torch.cat(ori_outs, dim=0))
    results.append(torch.cat(adv_outs, dim=0))

    save_grid_img(results, opts.test_path, 0)

    print("=====Saved successful in 'test/result.png'======")









