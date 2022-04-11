# coding=utf-8
"""
Entroformer with hyperprior and context model.
"""
import argparse, os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Compose, ToTensor
from torchvision.utils import save_image

from dataset import DatasetFromFolder
from module import *
from module import ssim
from util import *
from ac.util import ArithmeticEncoder, ArithmeticDecoder
from criterion import *

import warnings
warnings.filterwarnings("ignore")


def train(epoch):
    # Loss: MSE
    criterion_mse = nn.MSELoss()
    criterion_msssim = ssim.MS_SSIM(nonnegative_ssim=True)

    # Set module training
    encode_model.train()
    decode_model.train()
    quant_noise.train()
    quant_ste.train()
    cit_he.train()
    cit_hd.train()
    cit_ar.train()
    cit_pn.train()
    prob_model.train()

    num_pixels = opt.patchSize ** 2
    train_size = len(training_data_loader)
    
    # Training iteration
    for iteration, batch in enumerate(training_data_loader, 1):
        # batch
        image, _ = batch
        image = image.to(device)
        n, c, h, w = image.shape
        
        # Updata lr
        lr_scheduler.update_lr(batch_size=n)
        current_lr = lr_scheduler.get_lr()
        for param_group in base_optimizer.param_groups:
            param_group['lr'] = current_lr
        for param_group in entropy_optimizer.param_groups:
            param_group['lr'] = current_lr

        # Encoder
        y = encode_model(image*2 - 1)
        y_tilde = quant_noise(y)
        y_tilde2 = quant_ste(y)   # quant_ste(y), y_tilde
        # Hyperprio Transformer Entropy Model
        z = cit_he(y)
        z_tilde = quant_noise(z)
        feat_hyper = cit_hd(z_tilde)
        # Auto-regressive Transformer Entropy Model
        feat_ar = cit_ar(y_tilde)
        # Merge 2 features and Parameter Network
        feat_merge = torch.cat([feat_hyper,feat_ar], 1)
        predicted_param = cit_pn(feat_merge)      
        # Decoder
        x_tilde = decode_model(y_tilde2)
        x_tilde = x_tilde / 2 + 0.5
        x_tilde = torch.clamp(x_tilde, 0., 1.)
        
        # Distortion Loss 
        loss_mse = criterion_mse(x_tilde, image) * 255 * 255
        loss_ms_ssim = 1 - criterion_msssim(x_tilde*255., image*255.)
        if opt.loss_type=='mse':
            loss_distortion = loss_mse
        elif opt.loss_type=='msssim':
            loss_distortion = loss_ms_ssim
        else:
            raise ValueError("No such loss type")
            
        # Calculate bpp of y & z
        z_prob = prob_model(z_tilde)
        loss_rate_z = - torch.log2(z_prob + 1e-10).sum() / num_pixels / opt.batchSize
        loss_rate_y = criterion_entropy(y_tilde, predicted_param).sum() / np.log(2) / num_pixels / opt.batchSize
        total_loss = loss_distortion * opt.alpha + loss_rate_y + loss_rate_z
        
        # Zero Gradient
        base_optimizer.zero_grad()
        entropy_optimizer.zero_grad()
        # Backward and Update modules
        total_loss.backward()
        # Gradient Clipping
        if opt.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(encode_model.parameters(), opt.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(decode_model.parameters(), opt.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(cit_he.parameters(), opt.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(cit_hd.parameters(), opt.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(cit_ar.parameters(), opt.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(cit_pn.parameters(), opt.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(prob_model.parameters(), opt.grad_norm_clip)
        # Update
        base_optimizer.step()
        entropy_optimizer.step()
        
        if(iteration%20 == 0):
            print_fmt = "- Epoch[{}]({}/{}) - MSE:{:.2f}, MS-SSIM:{:.3f}, bpp y:{:.3f}, bpp z:{:.3f}, lr:{:.1e}/{:.1e}"
            log.logger.info(print_fmt.format(epoch, iteration, train_size,
                                   loss_mse.item(), 1-loss_ms_ssim.item(),
                                   loss_rate_y.item(), loss_rate_z.item(),
                                   base_optimizer.param_groups[0]['lr'], entropy_optimizer.param_groups[0]['lr']))
    log.logger.info("--- Epoch {} Complete.".format(epoch))

def test(epoch=0, shape_num=64):
    criterion_mse = nn.MSELoss()
    criterion_msssim = ssim.MS_SSIM()
        
    # Set module testing
    encode_model.eval()
    decode_model.eval()
    cit_he.eval()
    cit_hd.eval()
    cit_ar.eval()
    cit_pn.eval()
    prob_model.eval()
    quant_noise.eval()

    results = np.zeros((len(testing_data_loader), 6))
    with torch.no_grad():
        for iteration, sample in enumerate(testing_data_loader, 1):        
            image, img_path = sample
            img_name = img_path[0].split('/')[-1].split('.png')[0]
            image = image.to(device)
            n, c, h, w = image.shape
            num_pixels = h * w
            
            # image padding
            image_padded = img_pad(image, shape_num)
            # Encoder
            y = encode_model(image_padded*2 - 1)
            y_hat = quant_noise(y)
            # Hyperprio Transformer Entropy Model
            z = cit_he(y)
            z_hat = quant_noise(z)
            feat_hyper = cit_hd(z_hat)
            # Auto-regressive Transformer Entropy Model
            feat_ar = cit_ar(y_hat)
            # Merge 2 features and Parameter Network
            feat_merge = torch.cat([feat_hyper,feat_ar], 1)
            predicted_param = cit_pn(feat_merge)
            # Decoder
            x_hat = decode_model(y_hat)
            x_hat = x_hat / 2 + 0.5
            x_hat = torch.clamp(x_hat, 0., 1.)
            # image de-pad
            pad_up = ((shape_num - h % shape_num) % shape_num ) // 2
            pad_left = ((shape_num - w % shape_num) % shape_num ) // 2
            x_hat = x_hat[:, :, pad_up:pad_up+h, pad_left:pad_left+w]

            # predicted_param -> probability
            y_predicted_logits = criterion_entropy(y_hat, predicted_param)
            y_prob = (- y_predicted_logits).exp_()
            z_prob = prob_model(z_hat)

            # All the metric in np.array format
            mse = criterion_mse(x_hat*255., image*255.)
            psnr = 20. * np.log10(255.) - 10 * np.log10(mse.item())
            msssim = criterion_msssim(image*255, x_hat*255).item()
            bpp_y = - torch.log2(y_prob).sum().item() / num_pixels
            bpp_z = - torch.log2(z_prob).sum().item() / num_pixels
            log.logger.info("%s - PSNR:%.2f, MS-SSIM:%.5f, bpp:%.4f/%.4f"%(img_name,psnr,msssim,bpp_y,bpp_z))
            results[iteration-1] = [mse.item()*num_pixels*3, psnr, msssim, h, w, (bpp_y+bpp_z)*num_pixels]

    npixels_ = np.multiply(results[:,3], results[:,4])
    length_ = results[:, 5].sum()
    mse_ = results[:, 0].sum() / npixels_.sum() / 3
    psnr_, msssim_ = results[:,1].mean(), results[:,2].mean()
    bpp_ = length_ / npixels_.sum()
    format_print = "* Avg. PSNR:%.2f, MS-SSIM:%.5f, bpp:%.4f" % (psnr_,msssim_,bpp_)
    log.logger.info(format_print)

def compress(shape_num=64):
    if not os.path.exists('./compressed'):
        os.mkdir('./compressed')

    criterion_mse = nn.MSELoss()
    criterion_msssim = ssim.MS_SSIM()
    
    # Set module testing
    encode_model.eval()
    decode_model.eval()
    cit_he.eval()
    cit_hd.eval()
    cit_ar.eval()
    cit_pn.eval()
    prob_model.eval()
    quant_noise.eval()

    # Tables to CDF of channels
    tables = torch.range(-opt.table_range, opt.table_range-1)

    with torch.no_grad():
        image = Image.open(opt.input_file).convert('RGB')
        image = ToTensor()(image).unsqueeze(0)        
        image = image.to(device)
        img_name = opt.input_file.split('/')[-1].split('.png')[0]
        _, c, h, w = image.shape
        num_pixels = h * w
        
        # image padding
        image_padded = img_pad(image, shape_num)
        # Encoder
        y = encode_model(image_padded*2 - 1)
        y_hat = quant_noise(y)
        # Hyperprio Transformer Entropy Model
        z = cit_he(y)
        z_hat = quant_noise(z)
        feat_hyper = cit_hd(z_hat)
        # Auto-regressive Transformer Entropy Model
        feat_ar = cit_ar(y_hat)
        # Merge 2 features and Parameter Network
        feat_merge = torch.cat([feat_hyper,feat_ar], 1)
        predicted_param = cit_pn(feat_merge)
        # Decoder
        x_hat = decode_model(y_hat)
        x_hat = x_hat / 2 + 0.5
        x_hat = torch.clamp(x_hat, 0., 1.)
        # image de-pad
        pad_up = ((shape_num - h % shape_num) % shape_num ) // 2
        pad_left = ((shape_num - w % shape_num) % shape_num ) // 2
        x_hat = x_hat[:, :, pad_up:pad_up+h, pad_left:pad_left+w]

        # predicted_param -> probability
        y_predicted_logits = criterion_entropy(y_hat, predicted_param)
        y_prob = (- y_predicted_logits).exp_()
        z_prob = prob_model(z_hat)

        mse = criterion_mse(x_hat*255., image*255.)
        psnr = 20. * np.log10(255.) - 10 * np.log10(mse.item())
        msssim = criterion_msssim(image*255, x_hat*255)
        bpp_z = - torch.log2(z_prob).sum().item() / num_pixels
        bpp_y = - torch.log2(y_prob).sum().item() / num_pixels
        log.logger.info("%s - PSNR:%.2f, MS-SSIM:%.5f, bpp:%.4f/%.4f"%(img_name,psnr,msssim,bpp_y,bpp_z))

        # Compress
        yh, yw = y.shape[2:]
        zh, zw = z.shape[2:]

        # Compress z_hat
        tables_z = tables.repeat(1, opt.hyper_channels, 1, 1).to(device)
        z_symbol = z_hat.type(torch.int16).cpu() + opt.table_range
        pmf_z = prob_model(tables_z).unsqueeze(-2).cpu()
        cdf_z = torch.cumsum(torch.clip(pmf_z, 1e-9, None), dim=-1)
        cdf_z = torch.roll(cdf_z, shifts=1, dims=-1)
        cdf_z[...,0] = 0
        cdf_z = cdf_z.repeat(1,1,zh,zw,1).clip(min=0, max=1)
        
        # Compress y_hat
        # [L, C, H, W]
        tables_y = tables.repeat(opt.last_channels, yh, yw, 1).to(device).permute(3,0,1,2)
        # [1, H, W, C]
        y_symbol = y_hat.type(torch.int16).cpu().permute(0,2,3,1) + opt.table_range
        pmf_y_logit = criterion_entropy(tables_y, predicted_param.repeat(opt.table_range*2, 1, 1, 1))
        pmf_y = (-pmf_y_logit).exp_().cpu()
        # [1, H, W, C, L]
        pmf_y = pmf_y.permute(2,3,1,0).unsqueeze(0)
        cdf_y = torch.cumsum(pmf_y , dim=-1)
        cdf_y = torch.roll(cdf_y, shifts=1, dims=-1)
        cdf_y[...,0] = 0
        cdf_y = cdf_y.clip(min=0, max=1)

        # Write to binary file
        ac_encoder = ArithmeticEncoder("compressed/%s.bin" % img_name)
        ac_encoder.write_int([h,w,yh,yw,zh,zw])  # write shape of image and feature

        if opt.na == 'unidirectional':
            cdf = torch.cat([cdf_z.view(-1, cdf_z.size(-1)), cdf_y.view(-1, cdf_y.size(-1))], dim=0)
            symbol = torch.cat([z_symbol.flatten(), y_symbol.flatten()], dim=0)
            ac_encoder.encode(cdf, symbol)
        else:
            L = opt.table_range*2
            _, _, _, mask = cit_ar.get_mask(1, yh, yw)

            y1_slice_idx = torch.where(mask[0,0].flatten() == False)[0]
            y1_slice_idx = y1_slice_idx.view(1, y1_slice_idx.size(0), 1).repeat(1, 1, opt.last_channels)
            y1_symbol_slice = torch.gather(y_symbol.view(1, -1, opt.last_channels), dim=1, index=y1_slice_idx)
            y1_slice_idx = y1_slice_idx.unsqueeze(-1).repeat(1,1,1,L)
            cdf_y1_slice = torch.gather(cdf_y.view(1, -1, cdf_y.size(-2), cdf_y.size(-1)), dim=1, index=y1_slice_idx)

            y2_slice_idx = torch.where(mask[0,0].flatten() == True)[0]
            y2_slice_idx = y2_slice_idx.unsqueeze(-1).repeat(1, opt.last_channels).unsqueeze(0)
            y2_symbol_slice = torch.gather(y_symbol.view(1, -1, opt.last_channels), dim=1, index=y2_slice_idx)
            y2_slice_idx = y2_slice_idx.unsqueeze(-1).repeat(1,1,1,L)
            cdf_y2_slice = torch.gather(cdf_y.view(1, -1, cdf_y.size(-2), cdf_y.size(-1)), dim=1, index=y2_slice_idx)

            cdf = [cdf_z.view(-1, L), cdf_y1_slice.view(-1, L), cdf_y2_slice.view(-1, L)]
            cdf = torch.cat(cdf, dim=0)
            symbol = [z_symbol.flatten(), y1_symbol_slice.flatten(), y2_symbol_slice.flatten()]
            symbol = torch.cat(symbol, dim=0)
            ac_encoder.encode(cdf, symbol)
        ac_encoder.close()

def decompress(shape_num=64):
    if not os.path.exists('./decompressed'):
        os.mkdir('./decompressed')

    # Set module testing
    decode_model.eval()
    cit_hd.eval()
    cit_ar.eval()
    cit_pn.eval()
    prob_model.eval()
    quant_noise.eval()

    # Tables to CDF of channels
    tables = torch.range(-opt.table_range, opt.table_range-1)

    with torch.no_grad():
        # Read from binary file
        ac_decoder = ArithmeticDecoder(opt.input_file, opt.table_range*2)
        h,w,yh,yw,zh,zw = ac_decoder.read_head(6)
        ac_decoder.construct(zh*zw + yh*yw)

        # decompress z_hat
        tables_z = tables.repeat(1, opt.hyper_channels, 1, 1).to(device)
        pmf_z = prob_model(tables_z).unsqueeze(-2).cpu()
        cdf_z = torch.cumsum(torch.clip(pmf_z, 1e-9, None) , dim=-1)
        cdf_z = torch.roll(cdf_z, shifts=1, dims=-1)
        cdf_z[...,0] = 0
        cdf_z = cdf_z.repeat(1,1,zh,zw,1).clip(min=0, max=1)
        z_symbol = ac_decoder.decode(cdf_z)
        z_hat = torch.Tensor(z_symbol).type(torch.float32).to(device) - opt.table_range

        # hyperprior decoder
        feat_hyper = cit_hd(z_hat)
        
        if opt.na == 'unidirectional':    
            y_hat = torch.zeros((1, opt.last_channels, yh, yw)).to(device)
            tables_y = tables.repeat(opt.last_channels, 1, 1, 1).to(device).permute(3,0,1,2)
            for i in range(yh):
                print("Row: %d"%i)
                for j in range(yw):
                    feat_ar = cit_ar(y_hat)
                    feat_merge = torch.cat([feat_hyper,feat_ar], 1)
                    predicted_param = cit_pn(feat_merge)[:,:,i:i+1,j:j+1]

                    pmf_y_logit = criterion_entropy(tables_y, predicted_param.repeat(opt.table_range*2, 1, 1, 1))
                    pmf_y = (-pmf_y_logit).exp_().cpu()
                    # [1, H, W, C, L]
                    pmf_y = pmf_y.permute(2,3,1,0).unsqueeze(0)
                    cdf_y = torch.cumsum(pmf_y , dim=-1)
                    cdf_y = torch.roll(cdf_y, shifts=1, dims=-1)
                    cdf_y[...,0] = 0
                    cdf_y = cdf_y.clip(min=0, max=1)
                    
                    y_symbol = ac_decoder.decode(cdf_y)
                    # [1, H, W, C] -> [1, C, H, W]
                    y_symbol = torch.Tensor(y_symbol).permute(0,3,1,2).to(device)
                    y_hat[0,:,i,j] = y_symbol[0,:,0,0] - opt.table_range
        else:
            y_hat = torch.zeros((1, opt.last_channels, yh, yw)).to(device)
            tables_y = tables.repeat(opt.last_channels, yh, yw, 1).to(device).permute(3,0,1,2)

            L = opt.table_range*2
            _, _, _, mask = cit_ar.get_mask(1, yh, yw)
            
            ## y1 slice
            feat_ar = cit_ar(y_hat)
            feat_merge = torch.cat([feat_hyper,feat_ar], 1)
            predicted_param = cit_pn(feat_merge)

            pmf_y_logit = criterion_entropy(tables_y, predicted_param.repeat(opt.table_range*2, 1, 1, 1))
            pmf_y = (-pmf_y_logit).exp_().cpu()
            pmf_y = pmf_y.permute(2,3,1,0).unsqueeze(0)  # [1, H, W, C, L]
            cdf_y = torch.cumsum(pmf_y , dim=-1)
            cdf_y = torch.roll(cdf_y, shifts=1, dims=-1)
            cdf_y[...,0] = 0
            cdf_y = cdf_y.clip(min=0, max=1)
            
            y1_slice_idx = torch.where(mask[0,0].flatten() == False)[0]
            y1_slice_idx = y1_slice_idx.view(1, y1_slice_idx.size(0), 1, 1).repeat(1, 1, opt.last_channels, L)
            cdf_y1_slice = torch.gather(cdf_y.view(1, -1, cdf_y.size(-2), cdf_y.size(-1)), dim=1, index=y1_slice_idx)

            y1_symbol_slice = ac_decoder.decode(cdf_y1_slice)
            y1_hat_slice = torch.Tensor(y1_symbol_slice).permute(0,2,1).to(device) - opt.table_range
            y1_slice_idx = y1_slice_idx[:,:,:,0].permute(0,2,1).to(device)
            y_hat = y_hat.view(1, opt.last_channels, yh*yw).scatter(dim=2, index=y1_slice_idx, src=y1_hat_slice)
            y_hat = y_hat.view(1, opt.last_channels, yh, yw)

            ## y2 slice
            feat_ar = cit_ar(y_hat)
            feat_merge = torch.cat([feat_hyper,feat_ar], 1)
            predicted_param = cit_pn(feat_merge)

            pmf_y_logit = criterion_entropy(tables_y, predicted_param.repeat(opt.table_range*2, 1, 1, 1))
            pmf_y = (-pmf_y_logit).exp_().cpu()
            pmf_y = pmf_y.permute(2,3,1,0).unsqueeze(0)  # [1, H, W, C, L]
            cdf_y = torch.cumsum(pmf_y , dim=-1)
            cdf_y = torch.roll(cdf_y, shifts=1, dims=-1)
            cdf_y[...,0] = 0
            cdf_y = cdf_y.clip(min=0, max=1)
            
            y2_slice_idx = torch.where(mask[0,0].flatten() == True)[0]
            y2_slice_idx = y2_slice_idx.view(1, y2_slice_idx.size(0), 1, 1).repeat(1, 1, opt.last_channels, L)
            cdf_y2_slice = torch.gather(cdf_y.view(1, -1, cdf_y.size(-2), cdf_y.size(-1)), dim=1, index=y2_slice_idx)

            y2_symbol_slice = ac_decoder.decode(cdf_y2_slice)
            y2_hat_slice = torch.Tensor(y2_symbol_slice).permute(0,2,1).to(device) - opt.table_range
            y2_slice_idx = y2_slice_idx[:,:,:,0].permute(0,2,1).to(device)
            y_hat = y_hat.view(1, opt.last_channels, yh*yw).scatter(dim=2, index=y2_slice_idx, src=y2_hat_slice)
            y_hat = y_hat.view(1, opt.last_channels, yh, yw)

        ac_decoder.close()
        
        # Decoder
        x_hat = decode_model(y_hat)
        x_hat = x_hat / 2 + 0.5
        x_hat = torch.clamp(x_hat, 0., 1.)
        
        # image de-pad
        pad_up = ((shape_num - h % shape_num) % shape_num ) // 2
        pad_left = ((shape_num - w % shape_num) % shape_num ) // 2
        x_hat = x_hat[:, :, pad_up:pad_up+h, pad_left:pad_left+w]
        
        # Save image
        img_name = opt.input_file.split('/')[-1].split('.bin')[0]
        decompress_img = "decompressed/%s.png" % img_name
        save_image(x_hat[0].clone(), decompress_img)

def checkpoint(epoch, model_prefix='checkpoint/'):
    if not os.path.exists(model_prefix):
        os.mkdir(model_prefix)
    model_out_path = os.path.join( model_prefix , "model_epoch_{}.pth".format(epoch) )
    if isinstance(encode_model, torch.nn.DataParallel):
        state = {'encode':encode_model.module, 
                 'decode': decode_model.module,
                 'cit_he':cit_he.module,
                 'cit_hd':cit_hd.module,
                 'prob': prob_model.module,
                 'cit_ar':cit_ar.module,
                 'cit_pn':cit_pn.module,
                 }
    else:
        state = {'encode':encode_model, 
                 'decode': decode_model, 
                 'cit_he':cit_he,
                 'cit_hd':cit_hd,
                 'prob': prob_model,
                 'cit_ar':cit_ar,
                 'cit_pn':cit_pn,
                 }
    torch.save(state, model_out_path)
    log.logger.info("Checkpoint saved to {}".format(model_out_path))

def restore(model_pretrained):
    log.logger.info("===> Loading pre-trained model: %s" % model_pretrained)
    state = torch.load(model_pretrained, map_location=torch.device('cpu'))

    encode_model.load_state_dict(state['encode'].state_dict())
    decode_model.load_state_dict(state['decode'].state_dict())
    log.logger.info('Load main AE model.')

    cit_he.load_state_dict(state['cit_he'].state_dict())
    cit_hd.load_state_dict(state['cit_hd'].state_dict())
    prob_model.load_state_dict(state['prob'].state_dict())
    cit_ar.load_state_dict(state['cit_ar'].state_dict(), strict=False)
    cit_pn.load_state_dict(state['cit_pn'].state_dict())
    log.logger.info('Load Transformer entropy model.')


if __name__ == "__main__":
    # Arg settings
    parser = get_parser()
    opt = parser.parse_args()

    # create log
    log_file = '%s.log' % opt.mode
    log = Logger(filename=os.path.join(opt.model_prefix, log_file), 
                 level='info', 
                 fmt="%(asctime)s - %(message)s")
    log.logger.info(opt)

    # Environment setting
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda else "cpu")

    ## Main Auto-encoder model
    log.logger.info('===> Building model')
    encode_model = Balle2Encoder(opt.channels, opt.last_channels, opt.norm)
    decode_model = Balle2Decoder(opt.channels, opt.last_channels, opt.norm)
    # Quantize Mode
    quant_noise = NoiseQuant(table_range=opt.table_range)
    quant_ste = SteQuant(table_range=opt.table_range)
    # Probability model of hyperprior information
    prob_model = Entropy(opt.hyper_channels)
    # Hyperprior Transformer Entropy Model
    cit_he = TransHyperScale(cin=opt.last_channels, cout=opt.hyper_channels, scale=opt.scale, down=True, opt=opt)
    cit_hd = TransHyperScale(cin=opt.hyper_channels, scale=opt.scale, down=False, opt=opt)
    # AR Transformer Entropy Model and PN module.
    if(opt.na == 'unidirectional'):
        cit_ar = TransDecoder(cin=opt.last_channels, opt=opt)
    elif(opt.na == 'bidirectional'):
        TransDecoder2.train_scan_mode = 'random' if opt.mask_ratio > 0 else 'default'
        cit_ar = TransDecoder2(cin=opt.last_channels, opt=opt)
    else:
        raise ValueError("No such na.")

    # Parameter Network
    cit_pn = torch.nn.Sequential(
        nn.Conv2d(opt.dim_embed*2, opt.dim_embed*opt.mlp_ratio, 1, 1, 0),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(opt.dim_embed*opt.mlp_ratio, opt.last_channels*opt.K*opt.num_parameter, 1, 1, 0),
    )

    # Construct Criterion
    criterion_entropy = DiscretizedMixGaussLoss(rgb_scale=False, x_min=-opt.table_range, x_max=opt.table_range-1,
                                                num_p=opt.num_parameter, L=opt.table_range*2)
   
    # Init modules
    encode_model.apply(xavier_uniform_init)
    decode_model.apply(xavier_uniform_init)
    cit_he.apply(vit2_init)
    cit_hd.apply(vit2_init)
    cit_ar.apply(vit2_init)
    cit_pn.apply(xavier_uniform_init) 

    log.logger.info(encode_model)
    log.logger.info(decode_model)
    log.logger.info(cit_he)
    log.logger.info(cit_hd)
    log.logger.info(cit_ar)
    log.logger.info(cit_pn)

    # Load pre-trained model
    if(opt.model_pretrained != ""):
        restore(opt.model_pretrained)

    # GPU setting
    if torch.cuda.device_count() > 1:
        encode_model = nn.DataParallel(encode_model)
        decode_model = nn.DataParallel(decode_model)
        cit_he = nn.DataParallel(cit_he)
        cit_hd = nn.DataParallel(cit_hd)
        cit_ar = nn.DataParallel(cit_ar)
        cit_pn = nn.DataParallel(cit_pn)
        prob_model = nn.DataParallel(prob_model)
        criterion_entropy = nn.DataParallel(criterion_entropy)
    encode_model.to(device)
    decode_model.to(device)
    quant_noise.to(device)
    quant_ste.to(device)
    cit_he.to(device)
    cit_hd.to(device)
    cit_ar.to(device)
    cit_pn.to(device)
    prob_model.to(device)

    if(opt.mode == "test"):
        test_set = DatasetFromFolder(opt.test_dir, input_transform=Compose([ToTensor()]))
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
        test()
    elif(opt.mode == "compress"):
        compress()
    elif(opt.mode == "decompress"):
        decompress()
    elif(opt.mode == "train"):
        transform = Compose([RandomCrop(opt.patchSize), RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])   
        train_set = DatasetFromFolder(opt.train_dir, input_transform=transform, cache=False)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
        test_set = DatasetFromFolder(opt.test_dir, input_transform=Compose([ToTensor()]))
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
        
        # Optimizer and lr scheduler
        base_optimizer = torch.optim.AdamW([
                             {'params':encode_model.parameters(), 'lr':opt.lr},
                             {'params':decode_model.parameters(), 'lr':opt.lr},
                             ], eps=1e-8, weight_decay=opt.wd)
        entropy_optimizer = torch.optim.AdamW([
                                {'params':cit_he.parameters(), 'lr':opt.lr},
                                {'params':cit_hd.parameters(), 'lr':opt.lr},
                                {'params':cit_ar.parameters(), 'lr':opt.lr},
                                {'params':cit_pn.parameters(), 'lr':opt.lr},
                                {'params':prob_model.parameters(), 'lr':opt.lr},
                                ], eps=1e-8, weight_decay=opt.wd)
        lr_step = list(np.linspace(opt.epoch_pretrained, opt.nEpochs, 6, dtype=int))[1:]
        lr_scheduler = LearningRateScheduler(mode='stagedecay',
                                             lr=opt.lr,
                                             num_training_instances=len(train_set),
                                             stop_epoch=opt.nEpochs,
                                             warmup_epoch=opt.nEpochs*opt.warmup,
                                             stage_list=lr_step,
                                             stage_decay=opt.lr_decay)

        lr_scheduler.update_lr(opt.epoch_pretrained*len(train_set))
        log.logger.info("LR change in:")
        log.logger.info(lr_step)

        ckpt_stage = list(np.linspace(opt.epoch_pretrained, opt.nEpochs, 6, dtype=int))[1:]
        log.logger.info("Save checkpoint in:")
        log.logger.info(ckpt_stage)
        test(0)
        for epoch in range(opt.epoch_pretrained+1, opt.nEpochs+1):
            train(epoch)
            if epoch%1==0:
                test(epoch)
            if epoch in ckpt_stage:
                checkpoint(epoch, opt.model_prefix)
    else:
        raise ValueError("No such mode!")


    