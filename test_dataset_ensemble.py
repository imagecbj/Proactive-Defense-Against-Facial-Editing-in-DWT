
from tqdm import tqdm
import time
import datetime
from argparse import ArgumentParser

from networks.PG_network import define_G as PG_Model
from config import *

from data import CelebA
import torch.utils.data as data

from networks.Ensemble_models import Ensemble
from networks.Generalize_Model import Models as Black_models
from networks.SA_model import SA
from networks.DWT_model import DWT
from networks.vgg import Vgg16

from tools.color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from tools.metrics_compute import compute_metrics, compute_psnr, prepare_lpips, get_perceptual_loss
from tools.tool import *
from torchvision import transforms




if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('--flag', default=False, type=bool, help='is save results')
    config_parser.add_argument('--model_choice', default='attentiongan', type=str, help='compute metrics choice')

    config_parser.add_argument('--PG_path', default="./checkpoints/PG.pth", type=str, help='PG Weight Path')
    config_parser.add_argument('--StarG_path', default="./checkpoints/stargan/200000-G.ckpt", type=str,
                               help='Stargan Weight Path')
    config_parser.add_argument('--AttentionG_path', default="./checkpoints/attentiongan/200000-G.ckpt", type=str,
                               help='AttentionGAN Weight Path')
    config_parser.add_argument('--test_path', default='./test_dataset_ensemble', type=str, help='Test Result Path')
    config_parser.add_argument('--test_img', default='test.jpg', type=str, help='Test Image Name')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('--eposilon', default=0.02, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--dataset_path', default='/home/lab/workspace/dataset/CelebAMask-HQ/CelebA-HQ-img/',
                               type=str, help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path',
                               default='/home/lab/workspace/dataset/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt',
                               type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], type=list, help='Attribute Selection')
    opts = config_parser.parse_args()

    print(opts)


    save_path = opts.test_path + '/' + opts.model_choice
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # prepare
    y_mask = Y_mask(opts)

    # compute metrics prepare
    lpips_model, lpips_model2 = prepare_lpips()
    vgg = Vgg16().to(device)
    vgg.eval()
    criterion = torch.nn.MSELoss()

    # net prepare
    Models = Ensemble(opts)  # suggorate model
    GAN = Black_models(opts)
    SA_net = SA()  # saliency detection model
    DWT_net = DWT(opts)  # dwt
    # PG load
    PG = PG_Model(input_nc, output_nc, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    checkpoint = torch.load(opts.PG_path)
    PG.load_state_dict(checkpoint['protection_net'])
    PG.to(device)
    PG.eval()

    test_dataset = CelebA(opts.dataset_path, opts.attribute_txt_path,
                         opts.img_size, 'test', attrs,
                         opts.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=4,
        shuffle=False
    )
    print('The number of val Iterations = %d' % len(test_dataloader))
    # test_dataloader = get_loader(opts.dataset_path, opts.attribute_txt_path, opts.selected_attrs, batch_size=opts.batch_size, mode='test')

    psnr_value, ssim_value, lpips_alexs, lpips_vggs = 0.0, 0.0, 0.0, 0.0
    succ_num, total_num, n_dist = 0.0, 0.0, 0.0
    l1_error, l2_error = 0.0, 0.0
    vgg_sum = 0.0
    psnr_adv, ssim_adv, lpips_alexs_adv, lpips_vggs_adv = 0.0, 0.0, 0.0, 0.0
    print("Start testing...")
    start_time = time.time()
    for idx, (img_a, att_a, c_org, filename) in enumerate(tqdm(test_dataloader, desc='')):
        with torch.no_grad():
            x_real = img_a.to(device).clone().detach()
            c_trg_list = Models.create_labels(c_org)
            b_trg_list = Models.create_labels_attgan(att_a) # attgan

            x_ori = tensor2numpy(x_real)
            # ori_outs_fgan, ori_outs_attgan, ori_outs_hisd = Models.ensemble_models_out(x_real, c_trg_list, b_trg_list)
            if opts.model_choice == 'fgan':
                ori_outs = Models.fgan_outs(x_real, c_trg_list)
            elif opts.model_choice == 'attgan':
                ori_outs = Models.attgan_outs(x_real, b_trg_list)
            elif opts.model_choice == 'hisd':
                ori_outs = Models.hisd_outs(x_real)
            else:
                ori_outs = GAN.model_out(x_real, c_trg_list)


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
            if opts.model_choice == 'fgan':
                adv_outs = Models.fgan_outs(adv_A, c_trg_list)
            elif opts.model_choice == 'attgan':
                adv_outs = Models.attgan_outs(adv_A, b_trg_list)
            elif opts.model_choice == 'hisd':
                adv_outs = Models.hisd_outs(adv_A)
            else:
                adv_outs = GAN.model_out(adv_A, c_trg_list)
            # adv_outs_fgan, adv_outs_attgan, adv_outs_hisd = Models.ensemble_models_out(adv_A, c_trg_list, b_trg_list)

            # if opts.flag:
            #     fgan_results, attgan_results, hisd_results = [], [], []
            #     fgan_results.append(torch.cat(ori_outs_fgan, dim=0))
            #     fgan_results.append(torch.cat(adv_outs_fgan, dim=0))
            #     attgan_results.append(torch.cat(ori_outs_attgan, dim=0))
            #     attgan_results.append(torch.cat(adv_outs_attgan, dim=0))
            #     hisd_results.append(torch.cat(ori_outs_hisd, dim=0))
            #     hisd_results.append(torch.cat(adv_outs_hisd, dim=0))
            #     save_grid_img(fgan_results, save_path_fgan, idx)
            #     save_grid_img(attgan_results, save_path_attgan, idx)
            #     save_grid_img(hisd_results, save_path_hisd, idx)
            results = []
            results.append(torch.cat(ori_outs, dim=0))
            results.append(torch.cat(adv_outs, dim=0))
            save_grid_img(results, save_path, idx)


            ##### compute metrics #####
            # between clean image and defensed image
            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
            psnr_adv += psnr_temp
            ssim_adv += ssim_temp
            lpips_alexs_adv += lpips_alex
            lpips_vggs_adv += lpips_vgg

            # if opts.model_choice == 'fgan':
            #     ori_outs, adv_outs = ori_outs_fgan, adv_outs_fgan
            # if opts.model_choice == 'attgan':
            #     ori_outs, adv_outs = ori_outs_attgan, adv_outs_attgan
            # if opts.model_choice == 'hisd':
            #     ori_outs, adv_outs = ori_outs_hisd, adv_outs_hisd

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
                mask_d = abs(ori_outs[i+1] - x_real)
                mask_d = mask_d[0, 0, :, :] + mask_d[0, 1, :, :] + mask_d[0, 2, :, :]
                mask_d[mask_d > 0.5] = 1
                mask_d[mask_d < 0.5] = 0
                if (((ori_outs[i+1] * mask_d - adv_outs[i+1] * mask_d) ** 2).sum() / (mask_d.sum() * 3)) >= 0.05:
                    n_dist += 1

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
    mask_asr = n_dist / total_num
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
    log_message += f', mask_asr: {mask_asr:.5f}'

    print(log_message)
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print("time use:" + str(et))


