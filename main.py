from tqdm import tqdm
from torch.optim import SGD, Adam
from argparse import ArgumentParser

from networks.PG_network import define_G as PG_Model
from config import no_dropout, init_type, init_gain, ngf, net_noise, norm, device, input_nc, output_nc, max_psnr

from data_loader import get_loader

# fgan
from networks.SM_model import SM
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
    config_parser.add_argument('--eposilon', default=0.01, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--show_iter', default=1, type=int, help='Show Results After Every Iters')

    config_parser.add_argument('--dataset_path', default='/home/as/hh/data/CelebA-HQ-img/', type=str, help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path', default='/home/as/hh/data/CelebAMask-HQ-attribute-anno.txt', type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], type=list, help='Attribute Selection')

    config_parser.add_argument('--SM_path', default="./checkpoints/200000-G.ckpt", type=str, help='SM Weight Path')
    config_parser.add_argument('--mask_model_path', default="./checkpoints/FAN/best-model_epoch-204_mae-0.0505_loss-0.1370.pth", type=str, help='Saliency Detection Model Weight Path')

    config_parser.add_argument('--train_save_path',default='/data1/hh/Proactive_results/train', type=str, help='Train Result Path')
    config_parser.add_argument('--val_save_path',default='/data1/hh/Proactive_results/val', type=str, help='Val Result Path')
    config_parser.add_argument('--weight_save_path',default='/data1/hh/Proactive_results/weight', type=str, help='PG Weights Save Path')
    config_parser.add_argument('--logger_save_path',default='/data1/hh/Proactive_results/', type=str, help='Logger Save Path')
    opts = config_parser.parse_args()

    print(opts)
    perturb_wt = opts.perturb_wt
    batch_size = opts.batch_size
    loss_type = opts.loss_type
    lr = opts.lr
    eposilon = opts.eposilon
    img_size = opts.img_size
    show_iter = opts.show_iter

    path_isexists(opts)

    logger = setup_logger(opts.logger_save_path, 'result.log', 'train_logger')
    logger.info(f'======Proactive Defense Against Facial Editing in DWT=======')
    logger.info(f'Loading model.')

    train_dataloader = get_loader(opts.dataset_path, opts.attribute_txt_path, opts.selected_attrs, batch_size=batch_size, mode='train')
    val_dataloader = get_loader(opts.dataset_path, opts.attribute_txt_path, opts.selected_attrs, batch_size=batch_size, mode='val')

    # net prepare
    lpips_model, lpips_model2 = prepare_lpips()
    SM_net = SM(opts) # suggorate model
    SA_net = SA(opts) # saliency detection model
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



    for epoch in range(1, opts.iter_num):
        train_imgs = []
        val_imgs = []
        train_current_loss = {'D/loss_real': 0., 'D/loss_fake': 0., 'D/loss_gp': 0.,'G/loss_fake': 0., 'G/loss_attack': 0.}

        psnr_value, ssim_value, lpips_alexs, lpips_vggs = 0.0, 0.0, 0.0, 0.0
        psnr_sm = 0.0
        for idx, (img_a, c_org) in enumerate(tqdm(train_dataloader, desc='')):
            c_trg_list = SM_net.create_labels(c_org)
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
                ori_outs = SM_net.SM_out(x_real, c_trg_list)

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

            loss_adv = SM_net.compute_loss(adv_A, x_real, c_trg_list)

            g_loss = g_loss_fake + 10 * loss_adv
            disc_optim.zero_grad()
            model_optim.zero_grad()
            g_loss.backward()
            model_optim.step()


            if idx < 10 and epoch % show_iter == 0:
                adv_outs = SM_net.SM_out(adv_A, c_trg_list)

            train_current_loss['G/loss_attack'] += loss_adv.item()
            train_current_loss['G/loss_fake'] += g_loss_fake.item()
            # train_current_loss['disc_gan'] += loss_G_fake

            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
            psnr_value += psnr_temp
            ssim_value += ssim_temp
            lpips_alexs += lpips_alex
            lpips_vggs += lpips_vgg

            for i in range(len(opts.selected_attrs)):
                psnr_sm += compute_psnr(ori_outs[i+1], adv_outs[i+1])



            if epoch % show_iter == 0 and idx < 10:
                train_imgs.append(torch.cat(ori_outs, dim=0))
                train_imgs.append(torch.cat(adv_outs, dim=0))

        if train_imgs:
            save_grid_img(train_imgs, opts.train_save_path, epoch)


        psnr_value /= len(train_dataloader)
        ssim_value /= len(train_dataloader)
        lpips_alexs /= len(train_dataloader)
        lpips_vggs /= len(train_dataloader)

        psnr_sm /= (len(train_dataloader) * 5)

        train_current_loss['D/loss_real'] /= len(train_dataloader)
        train_current_loss['D/loss_fake'] /= len(train_dataloader)
        train_current_loss['D/loss_gp'] /= len(train_dataloader)
        train_current_loss['G/loss_fake'] /= len(train_dataloader)
        train_current_loss['G/loss_attack'] /= len(train_dataloader)


        val_current_loss = {'G/loss_attack': 0.}
        val_psnr_value, val_ssim_value, val_lpips_alexs, val_lpips_vggs = 0.0, 0.0, 0.0, 0.0
        for idx, (img_a, c_org) in enumerate(tqdm(val_dataloader, desc='')):
            with torch.no_grad():
                x_real = img_a.to(device).clone().detach()
                c_trg_list = SM_net.create_labels(c_org)
                x_ori = tensor2numpy(x_real)
                sa_mask = SA_net.compute_mask(x_real)

                x_ycbcr = rgb2ycbcr_np(x_ori)
                x_y = ycbcr_to_tensor(x_ycbcr).cuda()
                x_dwt = DWT_net.dwt(x_y)
                x_LL, x_HL, x_LH, x_HH = DWT_net.get_subbands(x_dwt)
                reshape_img = DWT_net.dwt_to_whole(x_LL, x_HL, x_LH, x_HH)

                if epoch % show_iter == 0 and idx < 10:
                    ori_outs = SM_net.SM_out(x_real, c_trg_list)

                adv_noise = PG(reshape_img) * y_mask
                adv_noise = torch.clamp(adv_noise, -eposilon, eposilon)
                adv_noise = noise_clamp(adv_noise, img_size, sa_mask)

                x_L_adv = reshape_img + adv_noise
                x_adv_dwt = DWT_net.whole_to_dwt(x_L_adv)
                x_adv = ycbcr_to_rgb(DWT_net.iwt(x_adv_dwt))
                adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

                #### compute loss ####
                loss = SM_net.compute_loss(adv_A, x_real, c_trg_list)

                if idx < 10 and epoch % show_iter == 0:
                    adv_outs = SM_net.SM_out(adv_A, c_trg_list)

                if epoch % show_iter == 0 and idx < 10:
                    val_imgs.append(torch.cat(ori_outs, dim=0))
                    val_imgs.append(torch.cat(adv_outs, dim=0))

                val_current_loss['G/loss_attack'] += loss.item()

                psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
                val_psnr_value += psnr_temp
                val_ssim_value += ssim_temp
                val_lpips_alexs += lpips_alex
                val_lpips_vggs += lpips_vgg
        if val_imgs:
            save_grid_img(val_imgs, opts.val_save_path, epoch)

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
        log_message += f', psnr_sm: {psnr_sm:.4f}'

        print(log_message)
        if logger:
            logger.debug(f'Step: {epoch:05d}, '
                          f'lr: {lr:.2e}, '
                         f'e: {eposilon:.2e},'
                          f'{log_message}')

        if psnr_sm < max_psnr:
            max_psnr = psnr_sm
            save_filename_model = 'perturb_%s.pth' % (epoch)
            save_path = os.path.join(opts.weight_save_path, save_filename_model)
            print('Updating the noise model')
            torch.save({"protection_net": PG.state_dict()}, save_path)
            best_loss = val_current_loss['G/loss_attack']

        print(
            'Epoch {} / {} \t Train Loss: {:.3f} \t Val Loss: {:.3f}'.format(epoch, opts.iter_num,
                                                                            train_current_loss['G/loss_attack'],
                                                                            val_current_loss['G/loss_attack']))
        save_filename_model = 'perturb_latest.pth'
        save_path = os.path.join(opts.weight_save_path, save_filename_model)
        torch.save({"protection_net": PG.state_dict()}, save_path)
