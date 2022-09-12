#!/usr/bin/env python
# coding: utf-8
import os
""" Setting """
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
wandb_prefix_name = "warp_mask"
know_args = ['--note',"Poly_varmap_only_epsilon",
             "--log_dir",f"/workspace/inpaint_mask/log/{wandb_prefix_name}/",
             "--data_dir","/workspace/inpaint_mask/data/warpData/celeba/",
             # "--data_dir", "/workspace/inpaint_mask/data/warpData/CIHP/Training/",
             # "--data_dir", "/workspace/inpaint_mask/data/warpData/Celeb-reID-light/train/",
             '--mask_type', "tri",
             '--varmap_type', "small_grid",
             '--varmap_threshold',"-1",
             
             "--mask_weight","1",
             
             "--batch_size","16",

             '--guassian_ksize','17',
             '--guassian_sigma','0.0',
             '--guassian_blur',
            #  "--in_out_area_split",
             "--wandb"
            ]
# image_size = (256,128)
image_size = (256,256)
# image_size = (512,512)
seed = 5
test_size = 0.1
val_batch_num = 6
device = "cuda"
weight_cliping_limit = 0.01

# assert args.D_iter > args.G_iter,print("WGAN Need D_iter > G_iter") # wgan parameters

# from losses.SegLoss.losses_pytorch.ND_Crossentropy import CrossentropyND
# from losses.SegLoss.losses_pytorch.dice_loss import SoftDiceLoss

# from losses.bl1_loss import BalancedL1Loss
# from losses.bmse_loss import BMCLossMD
from metric import BinaryMetrics 
from losses.poly_loss import PolyBCELoss 
import torch
import numpy as np
from tqdm.auto import tqdm
from get_args import get_args
from utils import (
    seed_everything,
    checkallData,
    check_create_dir,
    create_guassian_blur_f,
    to_pillow_f,
)
from loss_utils import (
    calculate_mask_loss_with_split,
    calc_gradient_penalty
)
from warp_dataset import WarppedDataset
from sklearn.model_selection import train_test_split
import timm
from models import Generator,Discriminator
from natsort import natsorted
import matplotlib.pyplot as plt

seed_everything(seed)
args = get_args(know_args)
assert len(timm.list_models(args.backbone,pretrained=True)) !=0, print(f"no such backbone {args.backbone} ")
args.image_size = image_size
print(vars(args))


""" Train Val Split """
d_dir = f"{args.data_dir}/{args.mask_type}/"
origin_dir = f"{d_dir}/origin/"
image_names = natsorted(os.listdir(origin_dir))
image_id_list = list(map(lambda s: s.split('.')[0], image_names))
print(len(image_id_list))

checkallData(d_dir,image_id_list)
print("Seed:",seed)
train_ids, valid_ids = train_test_split(image_id_list , test_size=test_size, random_state=seed)
print("Total train data:",len(train_ids))
print("Total valid data:", len(valid_ids))

""" Wandb"""
if args.wandb:
    import wandb
    wandb.init(project="warp_inpaint", entity='kycj')
    args.log_dir = f"{args.log_dir}/{wandb.run.id}/"
    wandb.config.update(args)
    wandb.run.name = f"{wandb_prefix_name}_{wandb.run.name}"
    
    print(vars(args))
    
""" result dir setting """
checkpoint_dir = args.log_dir + "./ckpts/"
check_create_dir(checkpoint_dir)
check_create_dir(checkpoint_dir+"/best/")
sample_dir = args.log_dir + "./samples/"
check_create_dir(sample_dir)

print("Log dir:",args.log_dir)


""" Data """
guassian_blur_f = False
if args.guassian_blur:
    guassian_blur_f = create_guassian_blur_f(args.guassian_ksize,args.guassian_sigma)
    print(f"Guassian_Blur ksize:{args.guassian_ksize}, sigma:{args.guassian_sigma}")
    
trainset = WarppedDataset(
                 args.data_dir,
                 train_ids,
                 args.mask_type,
                 args.varmap_type,
                 args.varmap_threshold,
                 guassian_blur_f=guassian_blur_f,
                 transform=None, 
                 return_mesh=True,
                 checkExist=False,
                 debug=False)
print("Total train len:",len(trainset))
train_loader = torch.utils.data.DataLoader(trainset, 
                                          batch_size= args.D_iter* args.batch_size,
                                          shuffle=True,
                                          drop_last=True, 
                                          num_workers=16
                                          )

validset = WarppedDataset(
                 args.data_dir,
                 valid_ids,
                 args.mask_type,
                 args.varmap_type,
                 args.varmap_threshold,
                 guassian_blur_f=guassian_blur_f,
                 transform=None, 
                 return_mesh=True,
                 checkExist=False,
                 debug=False)
print("Total valid len:",len(validset))
val_loader = torch.utils.data.DataLoader( 
                                          validset, 
                                          batch_size= args.batch_size,
                                          shuffle=True,
                                          drop_last=True, 
                                          num_workers=16
                                         )
val_batch_num = min(len(val_loader),val_batch_num) if val_batch_num> 0 else len(val_loader)  # 要用幾個batch來驗證 ,  len(val_loader) 個batch 的話就是全部的資料
# print("val_loader",len(val_loader), "src:", args.val_dir)
print(f"val_batch_num : {val_batch_num}/{len(val_loader)}")
print("num data per valid:",val_batch_num* args.batch_size)


""" Model """
G = Generator(image_size = image_size, backbone = args.backbone)
D = Discriminator(image_size = image_size)
G = G.to(device)
D = D.to(device)
""" Optimizer """
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr,betas=(0.5,0.99))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr,betas=(0.5,0.99))

""" Loss function """
l1_loss_f = torch.nn.L1Loss()
# smoothL1Loss = torch.nn.SmoothL1Loss()
# init_noise_sigma = 1.5
# bmc_loss = BMCLossMD(init_noise_sigma)
# bmc_loss.to(device)
# optimizer_G.add_param_group({'params': bmc_loss.parameters(), 'lr': 0.01})
poly_bce_loss = PolyBCELoss()
# dice_f = SoftDiceLoss()
# ce_f = CrossentropyND()
# mask_loss_f = lambda x,y : dice_f(x,y)+ ce_f(x,y)
# mask_loss_f = lambda x,y,var : poly_loss(x,y,var)
# balanced_l1_loss = BalancedL1Loss()
mask_loss_f = lambda pred,gt,var : poly_bce_loss(pred,gt,var)
# mask_loss_f = PolyLoss()
metric_f = BinaryMetrics(activation= None) # mask-estimator had use sigmoid



# In[54]:


import gc
print("train..")
step = 0
total_steps = args.epoch * len(train_loader)
with tqdm(total= total_steps) as pgbars:
    for epoch in range(args.epoch):
        for idx, batch_data in enumerate(train_loader):
            pgbars.set_description(f"Epoch:{epoch}/{args.epoch}-b{idx}/{len(train_loader)}")
            G.train()
            D.train()
            
            # origin_imgs, warpped_imgs = batch_data
            origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks, varmap = batch_data
            
            varmap = varmap.permute(0,3,1,2)
            masks = masks.permute(0,3,1,2)

            origin_imgs, warpped_imgs = origin_imgs.to(device), warpped_imgs.to(device)
            masks = masks.float().to(device)
            varmap = varmap.float().to(device)


            origin_list = origin_imgs.reshape((args.D_iter,-1,3,image_size[0],image_size[1]))
            warpped_list = warpped_imgs.reshape((args.D_iter,-1,3,image_size[0],image_size[1]))
            masks_list = masks.reshape((args.D_iter,-1,1,image_size[0],image_size[1]))
            varmap_list = varmap.reshape((args.D_iter,-1,1,image_size[0],image_size[1]))
            
            
            
            # WGAN - Training discriminator more iterations than generator
            """ UPDATE Discriminator """
            for p in D.parameters():  # reset requires_grad
                p.requires_grad = True
            
            for j in range(args.D_iter):
                optimizer_D.zero_grad()
                origin,warpped = origin_list[j], warpped_list[j]
                assert origin.shape == (args.batch_size,3,image_size[0],image_size[1])
                assert warpped.shape == (args.batch_size,3,image_size[0],image_size[1])
                
                # fake loss
                fake_images,_ = G(warpped)
                # fake_images = fake_images.detach()
                fake_D_logits = D(fake_images)
                fake_D_loss = fake_D_logits.mean(0).view(1)

                # real loss
                real_D_logits = D(origin)
                real_D_loss = real_D_logits.mean(0).view(1)

                # Train with gradient penalty
                gradient_penalty = calc_gradient_penalty(D, origin, fake_images, device, args)
                # print("gradient_penalty",gradient_penalty)

                d_loss = fake_D_loss - real_D_loss + gradient_penalty * args.gp_lambda 
                Wasserstein_D =  real_D_loss -  fake_D_loss

                d_loss.backward()
                torch.nn.utils.clip_grad_value_(D.parameters(), weight_cliping_limit)
                optimizer_D.step()

            """ UPDATE Generator """
            for p in D.parameters():
                p.requires_grad = False  # to avoid computation
                
            for _ in range(args.G_iter):
                optimizer_G.zero_grad()
                # origin, warpped
                i = np.random.randint(0, args.D_iter)
                origin,warpped = origin_list[i], warpped_list[i]
                gt_masks = masks_list[i]
                assert origin.shape == (args.batch_size,3,image_size[0],image_size[1])
                assert warpped.shape == (args.batch_size,3,image_size[0],image_size[1])
                assert gt_masks.shape == (args.batch_size,1,image_size[0],image_size[1])
                gt_varmap = varmap_list[i]
                assert gt_varmap.shape == (args.batch_size,1,image_size[0],image_size[1])
                
                # fake_loss
                fake_images, fake_masks = G(warpped)
                fake_logits = D(fake_images)
                fake_loss = fake_logits.mean(0).view(1)
                
                # l1_loss
                l1_loss = l1_loss_f(fake_images, origin).mean() 
                
                # matt_loss
                matt_loss = l1_loss_f(fake_masks * warpped, fake_masks * origin).mean()
                
                # mask_loss 
                mask_loss, in_area_mask_loss, out_area_mask_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1) 
                mask_loss = mask_loss_f(fake_masks, gt_masks, gt_varmap)
                # if args.in_out_area_split:
                #     mask_loss, in_area_mask_loss, out_area_mask_loss = calculate_mask_loss_with_split(gt_masks, fake_masks, args.in_area_weight, args.out_area_weight, l1_loss_f)
                # else:
                #     mask_loss = l1_loss_f(fake_masks, gt_masks).mean()

                # genreator loss
                # g_loss = - fake_loss + l1_loss * abs(fake_loss) + matt_loss + mask_loss
                g_loss = - fake_loss + ( l1_loss + 100*matt_loss + mask_loss) * abs(fake_loss) 
                g_loss.backward()
                optimizer_G.step()

                # metric
                pixel_acc, dice, precision, specificity, recall = metric_f(gt_masks, fake_masks)


            """ LOG """
            log_dict = {
                "train":{
                    "d_loss":d_loss.item(),
                    "d_gradient_penalty":gradient_penalty.item(),
                    "g_loss":g_loss.item(),
                    "g_fake_loss": fake_loss.item(),
                    "g_l1_loss": l1_loss.item(),
                    "matt_loss":  matt_loss.item(),
                    "mask_loss": mask_loss.item(),
                    "in_area_mask_loss": in_area_mask_loss.item(),
                    "out_area_mask_loss": out_area_mask_loss.item(),
                    "Wasserstein_D":Wasserstein_D.item(),
                    "metric":{
                        "pixel_acc":pixel_acc,  
                        "dice":dice, 
                        "precision":precision, 
                        "specificity":specificity,
                        "recall":recall
                    }
                    
                },
                # "val":{}
            }
            
            del d_loss
            del gradient_penalty
            del g_loss
            del fake_loss
            del l1_loss
            del matt_loss
            del mask_loss
            del Wasserstein_D
            del batch_data 
            del origin
            del warpped
            del fake_images
            del fake_masks
            del gt_masks
            collected = gc.collect()
            # print(f"Garbage collector: collected {collected} objects " )
            
            """ NO GRAD AREA """
            with torch.no_grad():
                G.eval()
                D.eval()
                
                origin,warpped,fake_images,fake_masks,gt_masks = None,None,None,None,None
                """ Validation """
                if np.mod(step, 100) == 1:
                    val_metrics = []
                    val_g_loss = []
                    val_fake_loss = []
                    val_matt_loss = []
                    val_l1_loss = []
                    val_mask_loss = []
                    val_in_area_mask_loss = []
                    val_out_area_mask_loss = []
                    # origin,warpped,fake_images,fake_masks,gt_masks = None,None,None,None,None
                    
                    for idx, batch_data in enumerate(val_loader):
                        if idx == val_batch_num:
                            break
                        # print(f"{idx}/{len(val_loader)}")
                    # for _ in range(1):
                        # batch_data = next(iter(val_loader))
                    
                        origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks, varmap = batch_data
            
                        varmap = varmap.permute(0,3,1,2)
                        masks = masks.permute(0,3,1,2)

                        origin_imgs, warpped_imgs = origin_imgs.to(device), warpped_imgs.to(device)
                        masks = masks.float().to(device)
                        varmap = varmap.float().to(device)

                        
                        origin,warpped = origin_imgs, warpped_imgs
                        gt_masks = masks
                        assert origin.shape == (args.batch_size,3,image_size[0],image_size[1])
                        assert warpped.shape == (args.batch_size,3,image_size[0],image_size[1])
                        assert gt_masks.shape == (args.batch_size,1,image_size[0],image_size[1])
                        gt_varmap = varmap
                        assert gt_varmap.shape == (args.batch_size,1,image_size[0],image_size[1])
                        
                        # fake_loss
                        fake_images, fake_masks = G(warpped)
                        fake_logits = D(fake_images)
                        fake_loss = fake_logits.mean(0).view(1)

                        # l1_loss
                        l1_loss = l1_loss_f(fake_images, origin).mean() 

                        # matt_loss
                        matt_loss = l1_loss_f(fake_masks * warpped, fake_masks * origin).mean()

                        # mask_loss 
                        mask_loss, in_area_mask_loss, out_area_mask_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1) 
                        mask_loss = mask_loss_f(fake_masks, gt_masks, gt_varmap)
                        # if args.in_out_area_split:
                        #     mask_loss, in_area_mask_loss, out_area_mask_loss = calculate_mask_loss_with_split(gt_masks, fake_masks, args.in_area_weight, args.out_area_weight, l1_loss_f)
                        # else:
                        #     mask_loss = l1_loss_f(fake_masks, gt_masks).mean()

                        # genreator loss
                        # g_loss = - fake_loss + l1_loss * abs(fake_loss) + matt_loss + mask_loss
                        g_loss = (- 1 + l1_loss + 100*matt_loss + args.mask_weight*mask_loss) * abs(fake_loss) 
                        
                        
                        # val_mask_l1_metric.append( l1_loss_f(fake_masks, gt_masks).mean().item() )
                        val_metric = metric_f(gt_masks, fake_masks)
                        val_metrics.append(val_metric)

                        val_fake_loss.append(fake_loss.item())
                        val_l1_loss.append(l1_loss.item())
                        val_matt_loss.append(matt_loss.item())
                        val_mask_loss.append(mask_loss.item())
                        val_in_area_mask_loss.append(in_area_mask_loss.item())
                        val_out_area_mask_loss.append(out_area_mask_loss.item())
                        val_g_loss.append(g_loss.item())

                    val_metrics = np.array(val_metrics).mean(axis = 0)
                    pixel_acc, dice, precision, specificity, recall = val_metrics

                    # print("np.array(val_mask_loss)",np.array(val_mask_loss).shape)
                    log_dict["val"] ={
                        "g_loss": np.array(val_g_loss).mean(),
                        "g_fake_loss": np.array(val_fake_loss).mean(),
                        "g_l1_loss": np.array(val_l1_loss).mean(),
                        "matt_loss":  np.array(val_matt_loss).mean(),
                        "mask_loss": np.array(val_mask_loss).mean(),
                        "in_area_mask_loss": np.array(val_in_area_mask_loss).mean(),
                        "out_area_mask_loss": np.array(val_out_area_mask_loss).mean(),
                        "metric":{
                            "pixel_acc":pixel_acc,  
                            "dice":dice, 
                            "precision":precision, 
                            "specificity":specificity,
                            "recall":recall
                        }
                    }
                    
                    # print("log_dict[val]",log_dict["val"])

                    k = np.random.randint(0, len(origin))
                    img_path = f"{sample_dir}/sample_{step}_{epoch}.jpg"
                    # to_pillow_f(fake_images[k]).save(img_path)

                    # plot result
                    fig, axs = plt.subplots(1, 7, figsize=(32,8))
                    axs[0].set_title('origin')
                    axs[0].imshow( to_pillow_f(origin[k]) )
                    
                    axs[1].set_title('warpped')
                    axs[1].imshow( to_pillow_f(warpped[k]) )
                    
                    axs[2].set_title('fakeImage')
                    axs[2].imshow( to_pillow_f(fake_images[k]) )

                    axs[3].set_title('fakeMask')
                    # 變成pillow之後 vmax應該改成255 而不是 1
                    axs[3].imshow( to_pillow_f(fake_masks[k]),vmin=0, vmax=255,cmap='gray' )
                    
                    axs[4].set_title('GTMask')
                    # 變成pillow之後 vmax應該改成255 而不是 1
                    axs[4].imshow( to_pillow_f(gt_masks[k]),vmin=0, vmax=255,cmap='gray' )

                    axs[5].set_title('GTMask on warpped')
                    axs[5].imshow( to_pillow_f(gt_masks[k]*warpped[k]))

                    axs[6].set_title('inv_varmap')
                    axs[6].imshow( to_pillow_f(1 - varmap[k]),vmin=0, vmax=255, cmap='gray' )

                    fig.savefig(img_path) 
                    plt.close(fig)
                    if args.wandb:
                        wandb.log({"Sample_Image": wandb.Image(img_path)}, step = step)
                    
                        

                """ MODEL SAVE """
                if np.mod(step, 1000) == 1:
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save({
                        'G_state_dict': G.state_dict(),
                        'D_state_dict':D.state_dict(),
                    }, 
                    f'{checkpoint_dir}/ckpt_{step}_{epoch}.pt'
                )
                    
            pgbars.set_postfix(log_dict)
            if args.wandb:
                wandb.log(log_dict,step=step)
            step = step + 1
            pgbars.update(1)