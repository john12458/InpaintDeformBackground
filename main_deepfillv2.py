#!/usr/bin/env python
# coding: utf-8

import os
from get_args import get_args
""" Setting """
wandb_prefix_name = "deep_fillv2"
image_size = (256,256)
seed = 5
test_size = 0.1
device = "cuda"
# weight_cliping_limit = 0.01
know_args = None
opt = get_args(know_args)
if opt.multi_gpu == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import models.deepfillv2.network as network
import deepfillv2_utils
from utils import (
    seed_everything,
    checkallData,
    check_create_dir,
    create_guassian_blur_f,
    to_pillow_f,
)
from warp_dataset import WarppedDataset
from natsort import natsorted
from sklearn.model_selection import train_test_split
import timm

# 雖然寫WGAN 但我看起來像LSGAN...
def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark
    """ Wandb"""
    if opt.wandb:
        import wandb
        wandb.init(project="warp_inpaint", entity='kycj')
        opt.log_dir = f"{opt.log_dir}/{wandb.run.id}/"
        wandb.config.update(opt)
        wandb.run.name = f"{wandb_prefix_name}_{wandb.run.name}"
        
    print(vars(opt))

    """ result dir setting """
    checkpoint_dir = opt.log_dir + "/ckpts/"
    check_create_dir(checkpoint_dir)
    check_create_dir(checkpoint_dir+"/best/")

    sample_dir = opt.log_dir + "/samples/"
    check_create_dir(sample_dir)

    """ Models """
    # Generator
    generator = network.GatedGenerator(opt)
    print('Generator is created!')
    network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize generator with %s type' % opt.init_type)
    # Discrimnator
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    # percepturalnet
    perceptualnet = network.PerceptualNet()
    print('Perceptual network is created!')

    """ Loss """
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    """ Optimizers """
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(checkpoint_dir, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # Save the dicriminator model
    def save_model_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(checkpoint_dir, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # load the model
    def load_model(net, epoch, opt, type='G'):
        """Save the model at "checkpoint_interval" and its multiple"""
        if type == 'G':
            model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(checkpoint_dir, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(generator, opt.resume_epoch, opt, type='G')
        load_model(discriminator, opt.resume_epoch, opt, type='D')
        print('--------------------Pretrained Models are Loaded--------------------')
        
    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------
    """ Data """
    """ Train Val Split """
    d_dir = f"{opt.data_dir}/{opt.mask_type}/"
    origin_dir = f"{d_dir}/origin/"
    print(origin_dir)
    image_names = natsorted(os.listdir(origin_dir))
    image_id_list = list(map(lambda s: s.split('.')[0], image_names))
    print(len(image_id_list))

    checkallData(d_dir,image_id_list)
    print("Seed:",seed)
    train_ids, valid_ids = train_test_split(image_id_list , test_size=test_size, random_state=seed)
    print("Total train data:",len(train_ids))
    print("Total valid data:", len(valid_ids))
    guassian_blur_f = False
    if opt.guassian_blur:
        guassian_blur_f = create_guassian_blur_f(opt.guassian_ksize,opt.guassian_sigma)
        print(f"Guassian_Blur ksize:{opt.guassian_ksize}, sigma:{opt.guassian_sigma}")
    trainset = WarppedDataset(
                 opt.data_dir,
                 train_ids,
                 opt.mask_type,
                 opt.varmap_type,
                 opt.varmap_threshold,
                 guassian_blur_f=guassian_blur_f,
                 transform=None, 
                 return_mesh=True,
                 checkExist=False,
                 debug=False)
    print("Total train len:",len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, 
                                          batch_size= opt.batch_size,
                                          shuffle=True,
                                          drop_last=True, 
                                          num_workers=16
                                          )

    validset = WarppedDataset(
                 opt.data_dir,
                 valid_ids,
                 opt.mask_type,
                 opt.varmap_type,
                 opt.varmap_threshold,
                 guassian_blur_f=guassian_blur_f,
                 transform=None, 
                 return_mesh=True,
                 checkExist=False,
                 debug=False)
    print("Total valid len:",len(validset))
    val_loader = torch.utils.data.DataLoader( 
                                            validset, 
                                            batch_size= opt.batch_size,
                                            shuffle=True,
                                            drop_last=True, 
                                            num_workers=16
                                            )                        
    val_batch_num = 5 # len(val_loader)  # 要用幾個batch來驗證 ,  len(val_loader) 個batch 的話就是全部的資料
    
    
    # ----------------------------------------
    #            Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            #  (img, height, width) = batch_data
            # img = img.cuda()
            # set the same free form masks for each batch
            # mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
            # for i in range(opt.batch_size):
                # mask[i] = torch.from_numpy(train_dataset.InpaintDataset.random_ff_mask(
                                                # shape=(height[0], width[0])).astype(np.float32)).cuda()
            
            origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks = batch_data
            masks = masks.permute(0,3,1,2)
            origin_imgs, warpped_imgs = origin_imgs.to(device), warpped_imgs.to(device)
            masks = masks.float().to(device)
            # deepfillv2 0:unmask 1:mask, 跟原本 mask 0:mask, 1: unmask不一樣
            img, mask = origin_imgs, (1 - masks)

            # LSGAN vectors
            valid = Tensor(np.ones((img.shape[0], 1, opt.imgsize//32, opt.imgsize//32)))
            fake = Tensor(np.zeros((img.shape[0], 1, opt.imgsize//32, opt.imgsize//32)))
            zero = Tensor(np.zeros((img.shape[0], 1, opt.imgsize//32, opt.imgsize//32)))
            # valid = Tensor(np.ones((img.shape[0], 1, height[0]//32, width[0]//32)))
            # fake = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))
            # zero = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))

            ### Train Discriminator
            optimizer_d.zero_grad()

            # Generator output
            first_out, second_out = generator(warpped_imgs, mask)

            # forward propagation
            first_out_wholeimg = warpped_imgs * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = warpped_imgs * (1 - mask) + second_out * mask      # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
            # True samples
            true_scalar = discriminator(origin_imgs, mask)

            
            # Loss and optimize
            loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
            loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            optimizer_g.zero_grad()

            # L1 Loss
            first_L1Loss = (first_out - origin_imgs).abs().mean()
            second_L1Loss = (second_out - origin_imgs).abs().mean()
            
            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = -torch.mean(fake_scalar)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            origin_img_featuremaps = perceptualnet(origin_imgs)                          # feature maps
            second_out_featuremaps = perceptualnet(second_out)
            second_PerceptualLoss = L1Loss(second_out_featuremaps, origin_img_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * first_L1Loss + opt.lambda_l1 * second_L1Loss + \
                   opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + batch_idx
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            """ LOG """
            log_dict = {
                "train":{
                    "first_Mask_L1_Loss":  first_L1Loss.item(),
                    "second_Mask_L1_Loss": second_L1Loss.item(),
                    "Perceptual_Loss": second_PerceptualLoss.item(),
                    "epoch": epoch 
                },
                "val":{}
            }
                


            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(train_loader), first_L1Loss.item(), second_L1Loss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
                (loss_D.item(), GAN_Loss.item(), second_PerceptualLoss.item(), time_left))
            
            masked_warpped = warpped_imgs * (1 - mask) + mask
            mask = torch.cat((mask, mask, mask), 1)
            """ Validation """
            if (batch_idx + 1) % 40 == 0:
                """ NO GRAD AREA """
                with torch.no_grad():
                    origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks = None, None, None, None, None
                    first_L1Loss_list = []
                    second_L1Loss_list = []
                    second_PerceptualLoss_list = []
                    for idx, batch_data in enumerate(val_loader):
                        if idx == val_batch_num:
                            break
                        origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks = batch_data
                        masks = masks.permute(0,3,1,2)
                        origin_imgs, warpped_imgs = origin_imgs.to(device), warpped_imgs.to(device)
                        masks = masks.float().to(device)
                        img, mask = origin_imgs, (1 - masks)  # deepfillv2 0:unmask 1:mask, 跟原本 mask 0:mask, 1: unmask不一樣

                        # Generator output
                        first_out, second_out = generator(warpped_imgs, mask)

                        # forward propagation
                        first_out_wholeimg = warpped_imgs * (1 - mask) + first_out * mask        # in range [0, 1]
                        second_out_wholeimg = warpped_imgs * (1 - mask) + second_out * mask      # in range [0, 1]

                        # L1 Loss
                        first_L1Loss = (first_out - origin_imgs).abs().mean()
                        second_L1Loss = (second_out - origin_imgs).abs().mean()
                        
                        # GAN Loss
                        # fake_scalar = discriminator(second_out_wholeimg, mask)
                        # GAN_Loss = -torch.mean(fake_scalar)

                        # Get the deep semantic feature maps, and compute Perceptual Loss
                        origin_img_featuremaps = perceptualnet(origin_imgs)                          # feature maps
                        second_out_featuremaps = perceptualnet(second_out)
                        second_PerceptualLoss = L1Loss(second_out_featuremaps, origin_img_featuremaps)


                        first_L1Loss_list.append(first_L1Loss.item())
                        second_L1Loss_list.append(second_L1Loss.item())
                        second_PerceptualLoss_list.append(second_PerceptualLoss.item())
                        

                        # Compute losses
                        # loss = opt.lambda_l1 * first_L1Loss + opt.lambda_l1 * second_L1Loss + \
                            # opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
                    
                    log_dict["val"] = {
                            "first_Mask_L1_Loss":  np.array(first_L1Loss_list).mean(),
                            "second_Mask_L1_Loss":  np.array(second_L1Loss_list).mean(),
                            "Perceptual_Loss": np.array(second_PerceptualLoss_list).mean(),
                            "epoch": epoch 
                    }
                    print("[VALIDATION]",log_dict["val"])

                    masked_warpped = warpped_imgs * (1 - mask) + mask
                    mask = torch.cat((mask, mask, mask), 1)    
                    img_list = [origin_imgs, warpped_imgs, mask, masked_warpped, first_out, second_out]
                    name_list = ['origin','warpped', 'mask', 'masked_warpped', 'first_out', 'second_out']
                    img_path =  deepfillv2_utils.save_sample_png(sample_folder = sample_dir, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
                    # print(img_path)
                    if opt.wandb:
                        wandb.log({"Sample_Image": wandb.Image(img_path)} , commit=False)
            
            if opt.wandb:
                wandb.log(log_dict, commit=True)

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model_generator(generator, (epoch + 1), opt)
        save_model_discriminator(discriminator, (epoch + 1), opt)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [origin_imgs, warpped_imgs, mask, masked_warpped, first_out, second_out]
            name_list = ['origin','warpped', 'mask', 'masked_warpped', 'first_out', 'second_out']
            deepfillv2_utils.save_sample_png(sample_folder = sample_dir, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

if __name__ == "__main__":
    WGAN_trainer(opt)