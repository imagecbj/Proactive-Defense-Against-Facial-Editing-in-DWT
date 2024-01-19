from tqdm import tqdm
from torch.optim import SGD, Adam
import time
import datetime
from argparse import ArgumentParser

from networks.PG_network import define_G as PG_Model
from config import no_dropout, init_type, init_gain, ngf, net_noise, norm, device, input_nc, output_nc, max_psnr, STYLE_WEIGHT, CONTENT_WEIGHT

from data_loader import get_loader


from networks.SM_model import SM
from networks.SA_model import SA
from networks.DWT_model import DWT
from networks.vgg import Vgg16
from networks.FGAN import Discriminator as D

from logger import setup_logger
from tools.color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from tools.metrics_compute import compute_metrics, compute_psnr, prepare_lpips, get_perceptual_loss
from tools.tool import *
from torchvision import transforms

if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('--SM_path', default="./checkpoints/200000-G.ckpt", type=str, help='SM Weight Path')
    config_parser.add_argument('--mask_model_path',
                               default="./checkpoints/FAN/best-model_epoch-204_mae-0.0505_loss-0.1370.pth", type=str,
                               help='Saliency Detection Model Weight Path')
    config_parser.add_argument('--PG_path', default="./checkpoints/PG.pth", type=str, help='PG Weight Path')
    config_parser.add_argument('--test_path',default='./test/test_dataset', type=str, help='Test Result Path')
    config_parser.add_argument('--test_img',default='test.jpg', type=str, help='Test Image Name')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('--eposilon', default=0.01, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--dataset_path', default='/home/as/hh/data/CelebA-HQ-img/', type=str,
                               help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path', default='/home/as/hh/data/CelebAMask-HQ-attribute-anno.txt',
                               type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], type=list, help='Attribute Selection')
    opts = config_parser.parse_args()

    print(opts)

    # prepare
    y_mask = Y_mask(opts)

    # compute metrics prepare
    lpips_model, lpips_model2 = prepare_lpips()
    vgg = Vgg16().to(device)
    vgg.eval()
    criterion = torch.nn.MSELoss()

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

    test_dataloader = get_loader(opts.dataset_path, opts.attribute_txt_path, opts.selected_attrs, batch_size=opts.batch_size, mode='test')

    psnr_value, ssim_value, lpips_alexs, lpips_vggs = 0.0, 0.0, 0.0, 0.0
    succ_num, total_num = 0.0, 0.0
    l1_error, l2_error = 0.0, 0.0
    vgg_sum = 0.0
    psnr_adv, ssim_adv, lpips_alexs_adv, lpips_vggs_adv = 0.0, 0.0, 0.0, 0.0
    print("Start testing...")
    start_time = time.time()
    for idx, (img_a, c_org) in enumerate(tqdm(test_dataloader, desc='')):
        with torch.no_grad():
            x_real = img_a.to(device).clone().detach()
            c_trg_list = SM_net.create_labels(c_org)

            x_ori = tensor2numpy(x_real)
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

            results = []
            results.append(torch.cat(ori_outs, dim=0))
            results.append(torch.cat(adv_outs, dim=0))
            save_grid_img(results, opts.test_path, idx)

            ##### compute metrics #####
            # between clean image and defensed image
            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
            psnr_adv += psnr_temp
            ssim_adv += ssim_temp
            lpips_alexs_adv += lpips_alex
            lpips_vggs_adv += lpips_vgg

            for i in range(len(adv_outs)-1):
                psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(ori_outs[i+1], adv_outs[i+1], lpips_model, lpips_model2)
                l1_error += torch.nn.functional.l1_loss(ori_outs[i+1], adv_outs[i+1]).item()
                l2_error += torch.nn.functional.mse_loss(ori_outs[i+1], adv_outs[i+1]).item()
                loss_style, loss_content = get_perceptual_loss(vgg, ori_outs[i+1], adv_outs[i+1])
                vgg_per = (STYLE_WEIGHT * loss_style + CONTENT_WEIGHT * loss_content).item()
                vgg_sum += vgg_per
                psnr_value += psnr_temp
                ssim_value += ssim_temp
                lpips_alexs += lpips_alex
                lpips_vggs += lpips_vgg

                # ASR
                dis = criterion(ori_outs[i+1], adv_outs[i+1])
                # dis = ((ori_outs[i] * pred_round - adv_outs[i] * pred_round) ** 2).sum() / (pred_round.sum() * 3)
                if dis >= 0.05:
                    succ_num = succ_num + 1
                total_num = total_num + 1

    len_crg = 5
    psnr_adv /= (idx + 1)
    ssim_adv /= (idx + 1)
    lpips_alexs_adv /= (idx + 1)
    lpips_vggs_adv /= (idx + 1)

    asr = succ_num / total_num
    psnr_value /= (idx + 1) * len_crg
    ssim_value /= (idx + 1) * len_crg
    lpips_alexs /= (idx + 1) * len_crg
    lpips_vggs /= (idx + 1) * len_crg
    l1_error /= (idx + 1) * len_crg
    l2_error /= (idx + 1) * len_crg
    vgg_sum /= (idx + 1) * len_crg
    log_message = "\nThe Average Metrics between clean and defensed images:\n"
    log_message += f'psnr_adv: {psnr_adv:.3f}'
    log_message += f', ssim_adv: {ssim_adv:.4f}'
    log_message += f', lpips(alex)_adv: {lpips_alexs_adv:.5f}'
    log_message += f', lpips(vgg)_adv: {lpips_vggs_adv:.5f}'
    log_message += "\nThe Average Metrics between clean outputs and defensed outputs:\n"
    log_message += f'psnr: {psnr_value:.3f}'
    log_message += f', ssim: {ssim_value:.4f}'
    log_message += f', lpips(alex): {lpips_alexs:.5f}'
    log_message += f', lpips(vgg): {lpips_vggs:.5f}'
    log_message += f', l1_error: {l1_error:.5f}'
    log_message += f', l2_error: {l2_error:.5f}'
    log_message += f', vgg: {vgg_sum:.5f}'
    log_message += f', asr: {asr:.5f}'

    print(log_message)
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print("time use:" + str(et))


