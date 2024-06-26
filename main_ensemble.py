from tqdm import tqdm
from torch.optim import SGD, Adam
from argparse import ArgumentParser

from networks.PG_network import define_G as PG_Model
from config import *

from data import CelebA
import torch.utils.data as data

# fgan
from networks.Ensemble_models import Ensemble
from networks.SA_model import SA
from networks.DWT_model import DWT
from networks.FGAN import Discriminator as D

from logger import setup_logger
from tools.color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from tools.metrics_compute import compute_metrics, compute_psnr, prepare_lpips
from tools.tool import *
from torchvision import transforms



if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('--iter_num', default=100, type=float, help='Training Iterations Numer')
    config_parser.add_argument('--perturb_wt', default=10, type=float, help='Perturbation Weight')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('--loss_type', default='l2', type=str, help='Loss Type')
    config_parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
    config_parser.add_argument('--eposilon', default=0.02, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--show_iter', default=1, type=int, help='Show Results After Every Iters')

    config_parser.add_argument('--dataset_path', default='/home/lab/workspace/dataset/CelebAMask-HQ/CelebA-HQ-img/', type=str, help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path', default='/home/lab/workspace/dataset/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt', type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], type=list, help='Attribute Selection')


    opts = config_parser.parse_args()

    print(opts)
    perturb_wt = opts.perturb_wt
    batch_size = opts.batch_size
    loss_type = opts.loss_type
    lr = opts.lr
    eposilon = opts.eposilon
    img_size = opts.img_size
    show_iter = opts.show_iter


    path_isexists()

    logger = setup_logger(logger_save_path, 'result.log', 'train_logger')
    logger.info(f'======Proactive Defense Against Facial Editing in DWT=======')
    logger.info(f'Loading model.')

    ######  data set prepare  ######
    train_dataset = CelebA(opts.dataset_path, opts.attribute_txt_path,
                           img_size, 'train', attrs,
                           opts.selected_attrs)
    val_dataset = CelebA(opts.dataset_path, opts.attribute_txt_path,
                         img_size, 'val', attrs,
                         opts.selected_attrs)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, num_workers=4,
        shuffle=True
    )
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=1, num_workers=4,
        shuffle=False
    )

    ###### net prepare ######
    lpips_model, lpips_model2 = prepare_lpips()
    Models = Ensemble(opts)
    SA_net = SA()  # saliency detection model
    DWT_net = DWT(opts)  # dwt

    # get the number of images in the dataset.
    print('The number of train Iterations = %d' % len(train_dataloader))
    print('The number of val Iterations = %d' % len(val_dataloader))

    optim_list = [(SGD, {'lr': lr}), (Adam, {'lr': lr})]
    my_optim = optim_list[1]
    optimizer = my_optim[0]
    optim_args = my_optim[1]
    y_mask = Y_mask(opts)

    PG = PG_Model(input_nc, output_nc, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    netDisc = D().to(device)

    model_optim = optimizer(params=list(PG.parameters()), **optim_args)
    disc_optim = optimizer(netDisc.parameters(), lr=lr)

    best_loss = float("inf")

    ######## 设置可学习权重 ######
    weights = torch.Tensor([0.3, 0.4, 0.3])
    # 设置更新步长
    delta = 0.05
    # 设置平衡因子
    balance_factor = 0.1

    performance_m1 = 20
    performance_m2 = 20
    performance_m3 = 20

    max_psnr_fgan, max_psnr_attgan, max_psnr_hisd = 50.0, 50.0, 50.0

    for epoch in range(1, opts.iter_num):
        train_imgs_fgan, val_imgs_fgan = [], []
        train_imgs_attgan, val_imgs_attgan = [], []
        train_imgs_hisd, val_imgs_hisd = [], []
        train_current_loss = {'D/loss_real': 0., 'D/loss_fake': 0., 'D/loss_gp': 0.,'G/loss_fake': 0., 'G/loss_attack': 0.}
        psnr_value, ssim_value, lpips_alexs, lpips_vggs = 0.0, 0.0, 0.0, 0.0
        # psnr_sm = 0.0
        psnr_fgan, psnr_attgan, psnr_hisd = 0.0, 0.0, 0.0
        for idx, (img_a, att_a, c_org, filename) in enumerate(tqdm(train_dataloader, desc='')):
            weights_softmax = weights
            c_trg_list = Models.create_labels(c_org)
            b_trg_list = Models.create_labels_attgan(att_a) # attgan

            x_real = img_a.to(device).clone().detach()
            x_ori = tensor2numpy(x_real)

            # compute saliency mask
            sa_mask = SA_net.compute_mask(x_real)
            # convert RGB to YCbCr
            x_ycbcr = rgb2ycbcr_np(x_ori)
            x_y = ycbcr_to_tensor(x_ycbcr).to(device)
            # DWT transform
            x_dwt = DWT_net.dwt(x_y)
            x_LL, x_HL, x_LH, x_HH = DWT_net.get_subbands(x_dwt)
            reshape_img = DWT_net.dwt_to_whole(x_LL, x_HL, x_LH, x_HH) #[1,3,256,256]

            if epoch % show_iter == 0 and idx < 10:
                ori_outs_fgan, ori_outs_attgan, ori_outs_hisd = Models.ensemble_models_out(x_real, c_trg_list, b_trg_list)

            adv_noise = PG(reshape_img) * y_mask
            adv_noise = torch.clamp(adv_noise, -eposilon, eposilon)
            adv_noise = noise_clamp(adv_noise, img_size, sa_mask)


            x_L_adv = reshape_img + adv_noise
            x_adv_dwt = DWT_net.whole_to_dwt(x_L_adv)
            x_adv = ycbcr_to_rgb(DWT_net.iwt(x_adv_dwt))
            adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

            ####### optimize D #######
            # compute loss with real images.
            out_src, out_cls = netDisc(x_real)
            d_loss_real = - torch.mean(out_src)

            # compute loss with fake images.
            out_src, _ = netDisc(adv_A.detach_())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
            x_hat = (alpha * x_real.data + (1 - alpha) * adv_A.data).requires_grad_(True)  # No Attack
            out_src, _ = netDisc(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat)

            d_loss = d_loss_real + d_loss_fake + 10 * d_loss_gp

            disc_optim.zero_grad()
            model_optim.zero_grad()
            d_loss.backward()
            disc_optim.step()

            train_current_loss['D/loss_real'] += d_loss_real.item()
            train_current_loss['D/loss_fake'] += d_loss_fake.item()
            train_current_loss['D/loss_gp'] += d_loss_gp.item()

            ####### optimize G #######
            adv_noise = PG(reshape_img) * y_mask
            adv_noise = torch.clamp(adv_noise, -eposilon, eposilon)
            adv_noise = noise_clamp(adv_noise, img_size, sa_mask)

            x_L_adv = reshape_img + adv_noise
            x_adv_dwt = DWT_net.whole_to_dwt(x_L_adv)
            x_adv = ycbcr_to_rgb(DWT_net.iwt(x_adv_dwt))
            adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

            out_src, _ = netDisc(adv_A)
            g_loss_fake = - torch.mean(out_src)

            loss_adv = Models.ensemble_compute_loss(adv_A, x_real, c_trg_list, weights_softmax)

            g_loss = g_loss_fake + 10 * loss_adv
            disc_optim.zero_grad()
            model_optim.zero_grad()
            g_loss.backward()
            model_optim.step()


            if idx < 10 and epoch % show_iter == 0:
                adv_outs_fgan, adv_outs_attgan, adv_outs_hisd = Models.ensemble_models_out(adv_A, c_trg_list, b_trg_list)
                # adv_outs = SM_net.SM_out(adv_A, c_trg_list)

            train_current_loss['G/loss_attack'] += loss_adv.item()
            train_current_loss['G/loss_fake'] += g_loss_fake.item()
            # train_current_loss['disc_gan'] += loss_G_fake

            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
            psnr_value += psnr_temp
            ssim_value += ssim_temp
            lpips_alexs += lpips_alex
            lpips_vggs += lpips_vgg

            for i in range(len(opts.selected_attrs)):
                psnr_fgan += compute_psnr(ori_outs_fgan[i+1], adv_outs_fgan[i+1])
                psnr_attgan += compute_psnr(ori_outs_attgan[i + 1], adv_outs_attgan[i + 1])
                psnr_hisd += compute_psnr(ori_outs_hisd[i + 1], adv_outs_hisd[i + 1])

            if epoch % show_iter == 0 and idx < 10:
                train_imgs_fgan.append(torch.cat(ori_outs_fgan, dim=0))
                train_imgs_fgan.append(torch.cat(adv_outs_fgan, dim=0))
                train_imgs_attgan.append(torch.cat(ori_outs_attgan, dim=0))
                train_imgs_attgan.append(torch.cat(adv_outs_attgan, dim=0))
                train_imgs_hisd.append(torch.cat(ori_outs_hisd, dim=0))
                train_imgs_hisd.append(torch.cat(adv_outs_hisd, dim=0))

        if train_imgs_fgan and train_imgs_attgan and train_imgs_hisd:
            save_grid_img(train_imgs_fgan, save_train_path_fgan, epoch)
            save_grid_img(train_imgs_attgan, save_train_path_attgan, epoch)
            save_grid_img(train_imgs_hisd, save_train_path_hisd, epoch)


        psnr_value /= len(train_dataloader)
        ssim_value /= len(train_dataloader)
        lpips_alexs /= len(train_dataloader)
        lpips_vggs /= len(train_dataloader)

        psnr_fgan /= (len(train_dataloader) * 5)
        psnr_attgan /= (len(train_dataloader) * 5)
        psnr_hisd /= (len(train_dataloader) * 5)

        if epoch % 5 == 0:
            if psnr_fgan > performance_m1:
                weights_softmax[0] += delta  # 0.05
                weights_softmax[1] -= delta * balance_factor # 0.05 * 0.1
                weights_softmax[2] -= delta * balance_factor
            elif psnr_attgan > performance_m2:
                weights_softmax[1] += delta
                weights_softmax[0] -= delta * balance_factor
                weights_softmax[2] -= delta * balance_factor
            elif psnr_hisd > performance_m3:
                weights_softmax[2] += delta
                weights_softmax[0] -= delta * balance_factor
                weights_softmax[1] -= delta * balance_factor
            # 平衡权重，确保他们之和为1
            total_weight = torch.sum(weights_softmax)
            weights_softmax[0] /= total_weight
            weights_softmax[1] /= total_weight
            weights_softmax[2] /= total_weight

        train_current_loss['D/loss_real'] /= len(train_dataloader)
        train_current_loss['D/loss_fake'] /= len(train_dataloader)
        train_current_loss['D/loss_gp'] /= len(train_dataloader)
        train_current_loss['G/loss_fake'] /= len(train_dataloader)
        train_current_loss['G/loss_attack'] /= len(train_dataloader)


        val_current_loss = {'G/loss_attack': 0.}
        val_psnr_value, val_ssim_value, val_lpips_alexs, val_lpips_vggs = 0.0, 0.0, 0.0, 0.0
        for idx, (img_a, att_a, c_org, filename) in enumerate(tqdm(val_dataloader, desc='')):
            with torch.no_grad():
                x_real = img_a.to(device).clone().detach()
                c_trg_list = Models.create_labels(c_org)
                b_trg_list = Models.create_labels_attgan(att_a)
                x_ori = tensor2numpy(x_real)
                sa_mask = SA_net.compute_mask(x_real)

                x_ycbcr = rgb2ycbcr_np(x_ori)
                x_y = ycbcr_to_tensor(x_ycbcr).cuda()
                x_dwt = DWT_net.dwt(x_y)
                x_LL, x_HL, x_LH, x_HH = DWT_net.get_subbands(x_dwt)
                reshape_img = DWT_net.dwt_to_whole(x_LL, x_HL, x_LH, x_HH)

                if epoch % show_iter == 0 and idx < 10:
                    ori_outs_fgan, ori_outs_attgan, ori_outs_hisd = Models.ensemble_models_out(x_real, c_trg_list, b_trg_list)


                adv_noise = PG(reshape_img) * y_mask
                adv_noise = torch.clamp(adv_noise, -eposilon, eposilon)
                adv_noise = noise_clamp(adv_noise, img_size, sa_mask)

                x_L_adv = reshape_img + adv_noise
                x_adv_dwt = DWT_net.whole_to_dwt(x_L_adv)
                x_adv = ycbcr_to_rgb(DWT_net.iwt(x_adv_dwt))
                adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

                #### compute loss ####
                loss = Models.ensemble_compute_loss(adv_A, x_real, c_trg_list, weights_softmax)

                if idx < 10 and epoch % show_iter == 0:
                    adv_outs_fgan, adv_outs_attgan, adv_outs_hisd = Models.ensemble_models_out(adv_A, c_trg_list, b_trg_list)

                if epoch % show_iter == 0 and idx < 10:
                    val_imgs_fgan.append(torch.cat(ori_outs_fgan, dim=0))
                    val_imgs_fgan.append(torch.cat(adv_outs_fgan, dim=0))
                    val_imgs_attgan.append(torch.cat(ori_outs_attgan, dim=0))
                    val_imgs_attgan.append(torch.cat(adv_outs_attgan, dim=0))
                    val_imgs_hisd.append(torch.cat(ori_outs_hisd, dim=0))
                    val_imgs_hisd.append(torch.cat(adv_outs_hisd, dim=0))

                val_current_loss['G/loss_attack'] += loss.item()

                psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
                val_psnr_value += psnr_temp
                val_ssim_value += ssim_temp
                val_lpips_alexs += lpips_alex
                val_lpips_vggs += lpips_vgg

        if val_imgs_fgan and val_imgs_attgan and val_imgs_hisd:
            save_grid_img(val_imgs_fgan, save_val_path_fgan, epoch)
            save_grid_img(val_imgs_attgan, save_val_path_attgan, epoch)
            save_grid_img(val_imgs_hisd, save_val_path_hisd, epoch)

        val_current_loss['G/loss_attack'] /= len(val_dataloader)

        val_psnr_value /= len(val_dataloader)
        val_ssim_value /= len(val_dataloader)
        val_lpips_alexs /= len(val_dataloader)
        val_lpips_vggs /= len(val_dataloader)

        log_message = ''
        for tag, value in train_current_loss.items():
            log_message += ", {}: {:.3f}".format(tag, value)

        log_message += f', psnr: {psnr_value:.3f}'
        log_message += f', ssim: {ssim_value:.4f}'
        log_message += f', lpips: {(lpips_alexs + lpips_vggs) / 2:.5f}'
        log_message += f", val_loss: {val_current_loss['G/loss_attack']:.3f}"
        log_message += f', val_psnr: {val_psnr_value:.3f}'
        log_message += f', val_ssim: {val_ssim_value:.4f}'
        log_message += f', val_lpips: {(val_lpips_alexs + val_lpips_vggs) / 2:.5f}'
        log_message += f', psnr_fgan: {psnr_fgan:.4f}'
        log_message += f', psnr_attgan: {psnr_attgan:.4f}'
        log_message += f', psnr_hisd: {psnr_hisd:.4f}'

        print(log_message)
        if logger:
            logger.debug(f'Step: {epoch:05d}, '
                          f'lr: {lr:.2e}, '
                         f'e: {eposilon:.2e},'
                          f'{log_message}')

        if psnr_fgan < max_psnr_fgan and psnr_attgan < max_psnr_attgan and psnr_hisd < max_psnr_hisd:
            max_psnr_fgan = psnr_fgan
            max_psnr_attgan = psnr_attgan
            max_psnr_hisd = psnr_hisd

            save_filename_model = 'perturb_%s.pth' % (epoch)
            save_path = os.path.join(weight_save_path, save_filename_model)
            print('Updating the noise model')
            torch.save({"protection_net": PG.state_dict()}, save_path)
            best_loss = val_current_loss['G/loss_attack']

        print(
            'Epoch {} / {} \t Train Loss: {:.3f} \t Val Loss: {:.3f}'.format(epoch, opts.iter_num,
                                                                            train_current_loss['G/loss_attack'],
                                                                            val_current_loss['G/loss_attack']))
        save_filename_model = 'perturb_latest.pth'
        save_path = os.path.join(weight_save_path, save_filename_model)
        torch.save({"protection_net": PG.state_dict()}, save_path)

        if psnr_fgan < 20 and psnr_hisd < 20 and psnr_attgan < 20:
            break
