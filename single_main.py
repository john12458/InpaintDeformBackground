#!/usr/bin/env python
# coding: utf-8
import os
""" Setting """
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
wandb_prefix_name = "warp_mask_SINGLE"
know_args = ['--note',"",
             "--log_dir",f"/workspace/inpaint_mask/log/{wandb_prefix_name}/",
            #  "--data_dir","/workspace/inpaint_mask/data/warpData/celeba/",
            #  "--data_dir","/workspace/inpaint_mask/data/warpData/fashionLandmarkDetectionBenchmark/",
             "--data_dir", "/workspace/inpaint_mask/data/warpData/CIHP/Training/",
            #  "--data_dir", "/workspace/inpaint_mask/data/warpData/Celeb-reID-light/train/",
            # "--ckpt_path",'/workspace/inpaint_mask/log/warp_mask_SINGLE/2e9ztqt2/ckpts/ckpt_14001_2.pt',

            '--mask_type', "tps_dgrid_2_origin_true",
            #  '--mask_type', "tps_dgrid_2",  #"--mask_threshold", "0.9",
            #  '--mask_type', "tri"
            # "--lr","0.00006",
            "--regularzation_weight","0.0",

            '--varmap_type', "small_grid",
            #  '--varmap_type', "notuse", "--maskloss_type", "l1",
            #  '--varmap_type', "notuse",
            
             '--varmap_threshold',"-1",
            #  "--backbone","swinv2_base_window12to16_192to256_22kft1k",
            #  "--backbone","vqvae", '--use_attention',
             "--mask_weight","1",
            
             "--batch_size","16","--no_warp_ratio", "0.0625",
            #  "--batch_size","10","--no_warp_ratio", "0.1",

             '--guassian_ksize','17',
             '--guassian_sigma','0.0',
            #  '--guassian_blur',
             "--no_mesh", 
            #  '--use_hieratical',
             
            #  '--mask_inverse',
            #  "--in_out_area_split",
            #  "--wandb"
            ]
image_size = (256,256)
# image_size = (256,128)
# image_size = (512,512)
seed = 5
test_size = 0.1
val_batch_num = 6
device = "cuda"
weight_cliping_limit = 0.01

# assert args.D_iter > args.G_iter,print("WGAN Need D_iter > G_iter") # wgan parameters
import torchvision.transforms as transforms
from test import test_one
from losses.bl1_loss import BalancedL1Loss
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
    visualize
)
from loss_utils import (
    calculate_mask_loss_with_split,
    calc_gradient_penalty
)
from warp_dataset import WarppedDataset
from sklearn.model_selection import train_test_split
import timm
from models.generators.mask_estimator import MaskEstimator
from natsort import natsorted
import matplotlib.pyplot as plt

seed_everything(seed)
args = get_args(know_args)
if (args.backbone != "vqvae"):
    assert len(timm.list_models(args.backbone,pretrained=True)) !=0, print(f"no such backbone {args.backbone} ")
args.image_size = image_size
print(vars(args))


mask_img_f = lambda mask,img: img*(mask) 
if args.mask_inverse:
    mask_img_f = lambda mask,img: img*(1. - mask) + mask
else:
    mask_img_f = lambda mask,img: img*(mask) 

""" Train Val Split """
d_dir = f"{args.data_dir}/{args.mask_type}/"
warpped_dir = f"{d_dir}/warpped/"
image_names = natsorted(os.listdir(warpped_dir))
image_id_list = list(map(lambda s: s.split('.')[0], image_names))
print(len(image_id_list))

# checkallData(d_dir,image_id_list)
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
                #  transform=None, 
                 transform=transforms.Resize(size = image_size), 
                 return_mesh=True,
                 checkExist=False,
                 debug=False,
                 inverse = args.mask_inverse,
                 no_mesh = args.no_mesh,
                 mask_threshold = args.mask_threshold)
print("Total train len:",len(trainset))
train_loader = torch.utils.data.DataLoader(trainset, 
                                          batch_size= args.batch_size,
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
                 transform=transforms.Resize(size = image_size), 
                 return_mesh=True,
                 checkExist=False,
                 debug=False,
                 inverse = args.mask_inverse,
                 no_mesh = args.no_mesh,
                 mask_threshold = args.mask_threshold)
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
G = MaskEstimator(image_size = image_size, backbone = args.backbone, use_attention= args.use_attention, use_hieratical=args.use_hieratical)
G = G.to(device)
# load model if args.ckpt_path
if args.ckpt_path != '':
    print("Load weight from:",args.ckpt_path)
    G.load_state_dict(torch.load(args.ckpt_path)['G_state_dict'])
""" Optimizer """
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr,betas=(0.5,0.99))

""" Loss function """
l1_loss_f = torch.nn.L1Loss()

mask_loss_f = None
regularzation_term_f = None
if args.maskloss_type == "balancel1":
    balanced_l1_loss = BalancedL1Loss()
    regularzation_term_f = lambda pred: torch.mean(
                        torch.log(0.1 + pred.view(pred.size(0), -1)) +
                        torch.log(0.1 + 1. - pred.view(pred.size(0), -1)) - -2.20727, dim=-1).mean()
    mask_loss_f = lambda pred,gt,var : balanced_l1_loss(pred,gt)
elif args.maskloss_type == "l1":
    regularzation_term_f = lambda pred: torch.Tensor([0.0]).to(pred.device)
    mask_loss_f = lambda pred,gt,var : l1_loss_f(pred,gt)
else:
    raise NotImplementedError(f"maskloss_type {args.mask_type} not implemented!")

metric_f = BinaryMetrics(activation= None) # mask-estimator had use sigmoid

""" Setting """
no_warp_num = int(args.batch_size * args.no_warp_ratio)
print("no_warp_num:",no_warp_num)

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

            
            # origin_imgs, warpped_imgs = batch_data
            origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks, varmap = batch_data
            
            # varmap = varmap.permute(0,3,1,2)
            # masks = masks.permute(0,3,1,2)

            origin_imgs, warpped_imgs = origin_imgs.to(device), warpped_imgs.to(device)
            masks = masks.float().to(device)
            varmap = varmap.float().to(device)

            origin,warpped,gt_masks = origin_imgs, warpped_imgs, masks
            gt_varmap = varmap
                
            optimizer_G.zero_grad()

            assert origin.shape == (args.batch_size,3,image_size[0],image_size[1])
            assert warpped.shape == (args.batch_size,3,image_size[0],image_size[1])
            assert gt_masks.shape == (args.batch_size,1,image_size[0],image_size[1])
            assert gt_varmap.shape == (args.batch_size,1,image_size[0],image_size[1])

            if no_warp_num > 0:
                sample_idxs = torch.randint(0, args.batch_size, (no_warp_num,))
                warpped[sample_idxs] = origin[sample_idxs].clone()
                gt_masks[sample_idxs] = torch.ones_like(gt_masks[sample_idxs])
            
            fake_masks = G(warpped)
            
            # matt_loss
            matt_loss = l1_loss_f(mask_img_f(fake_masks,warpped), mask_img_f(fake_masks,origin)).mean()
            # mask_loss 
            mask_loss = mask_loss_f(fake_masks, gt_masks, gt_varmap)
            # regularzation
            regularzation_term_loss = regularzation_term_f(fake_masks)

            # total
            g_loss = args.matt_weight * matt_loss + \
                args.mask_weight * mask_loss + \
                args.regularzation_weight * regularzation_term_loss

            g_loss.backward()
     
            optimizer_G.step()

            # metric
            pixel_acc, dice, precision, specificity, recall = metric_f(gt_masks, fake_masks)


            """ LOG """
            log_dict = {
                "train":{
                    "g_loss":g_loss.item(),
                    "matt_loss":  matt_loss.item(),
                    "mask_loss": mask_loss.item(),
                    "regularzation_term_loss": regularzation_term_loss.item(),
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
            
            del g_loss
            del matt_loss
            del mask_loss
            del batch_data 
            del origin
            del warpped
            del fake_masks
            del gt_masks
            collected = gc.collect()
            # print(f"Garbage collector: collected {collected} objects " )
            
            """ NO GRAD AREA """
            with torch.no_grad():
                G.eval()
                
                origin,warpped,fake_images,fake_masks,gt_masks = None,None,None,None,None
                """ Validation """
                if np.mod(step, 100) == 1:
                    val_metrics = []
                    val_g_loss = []
                    val_matt_loss = []
                    val_mask_loss = []
                    # origin,warpped,fake_images,fake_masks,gt_masks = None,None,None,None,None
                    
                    for idx, batch_data in enumerate(val_loader):
                        if idx == val_batch_num:
                            break
                        # print(f"{idx}/{len(val_loader)}")
                    # for _ in range(1):
                        # batch_data = next(iter(val_loader))
                    
                        origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks, varmap = batch_data
            
                        # varmap = varmap.permute(0,3,1,2)
                        # masks = masks.permute(0,3,1,2)

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
                        
                        fake_masks = G(warpped)

                        # matt_loss
                        matt_loss = l1_loss_f(mask_img_f(fake_masks,warpped), mask_img_f(fake_masks,origin)).mean()

                        # mask_loss 
                        mask_loss = mask_loss_f(fake_masks, gt_masks, gt_varmap)
                        
                         # total
                        g_loss = args.matt_weight * matt_loss + \
                            args.mask_weight * mask_loss + \
                            args.regularzation_weight * regularzation_term_loss
                        
                        
                        # val_mask_l1_metric.append( l1_loss_f(fake_masks, gt_masks).mean().item() )
                        val_metric = metric_f(gt_masks, fake_masks)
                        val_metrics.append(val_metric)

                        val_matt_loss.append(matt_loss.item())
                        val_mask_loss.append(mask_loss.item())
                        val_g_loss.append(g_loss.item())

                    val_metrics = np.array(val_metrics).mean(axis = 0)
                    pixel_acc, dice, precision, specificity, recall = val_metrics

                    # print("np.array(val_mask_loss)",np.array(val_mask_loss).shape)
                    log_dict["val"] ={
                        "g_loss": np.array(val_g_loss).mean(),
                        "matt_loss":  np.array(val_matt_loss).mean(),
                        "mask_loss": np.array(val_mask_loss).mean(),
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

                    visual_dict= {
                        'origin': {
                            'X':to_pillow_f(origin[k])
                        }, 
                        'warpped': {
                            'X':to_pillow_f(warpped[k])
                        },
                        'fakeMask': {
                            'X':to_pillow_f(fake_masks[k]),
                            'vmin':0, 
                            'vmax':255,
                            'cmap':'gray'
                        },
                        'GTMask':{
                            'X':to_pillow_f(gt_masks[k]),
                            'vmin':0, 
                            'vmax':255,
                            'cmap':'gray'
                        },
                        'fakeMask on warpped':{
                            'X':to_pillow_f(mask_img_f(fake_masks[k],warpped[k]))
                        },
                        'GTMask on warpped':{
                            'X':to_pillow_f(mask_img_f(gt_masks[k],warpped[k])) 
                        },
                        # 'inv_varmap': {
                        #     'X':to_pillow_f(1 - varmap[k]),
                        #     'vmin':0, 
                        #     'vmax':255,
                        #     'cmap':'gray'
                        # },
                    }
                    """ Test """
                    test_dict = {}
                    test_dir = '/workspace/inpaint_mask/data/test/warpped/'
                    test_files = ['05.jpeg','03.jpeg']
                    for test_name in test_files:
                        fake_mask, fake_masks_on_img  = test_one(f'{test_dir}/{test_name}',G,image_size, mask_img_f, device)
                        test_dict.update({
                            f'fakeMask on {test_name}':{
                                'X': to_pillow_f(fake_masks_on_img)
                            },
                            f'fakeMask_{test_name}':{
                                'X': to_pillow_f(fake_mask),
                                'vmin':0, 
                                'vmax':255,
                                'cmap':'gray'
                            }
                     })

                    """ Visualize """
                    # visual_dict.update(test_dict)
                    visualize(visual_dict,img_path)

                    test_img_path = f"{sample_dir}/test_{step}_{epoch}.jpg"
                    visualize(test_dict,test_img_path)

                    if args.wandb:
                        wandb.log({
                            "Sample_Image": wandb.Image(img_path),
                            "Test_Image": wandb.Image(test_img_path)
                        }, step = step)
                    
                        

                """ MODEL SAVE """
                if np.mod(step, 1000) == 1 :
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save({
                        'G_state_dict': G.state_dict(),
                        # 'D_state_dict':D.state_dict(),
                    }, 
                    f'{checkpoint_dir}/ckpt_{step}_{epoch}.pt'
                )
                    
            pgbars.set_postfix(log_dict)
            if args.wandb:
                wandb.log(log_dict,step=step)
            step = step + 1
            pgbars.update(1)