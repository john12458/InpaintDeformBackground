#!/usr/bin/env python
# coding: utf-8
# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Setting:
# In[3]:
wandb_prefix_name = "warp_mask"
know_args = ["--log_dir",f"/workspace/inpaint_mask/log/{wandb_prefix_name}/",
             # "--data_dir","/workspace/inpaint_mask/data/warpData/celeba/",
             # "--data_dir", "/workspace/inpaint_mask/data/warpData/CIHP/Training/",
             "--data_dir", "/workspace/inpaint_mask/data/warpData/Celeb-reID-light/train/",
             '--mask_type', "tri",
             '--varmap_type', "var(warp)",
             '--varmap_threshold',"-1",
             
             "--mask_weight","1",
             
             "--batch_size","16",
             "--wandb"
            ]
image_size = (256,128)
# image_size = (256,256)
# image_size = (512,512)
seed = 5
test_size = 0.1

# In[4]:
import argparse
def get_args(know_args=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--note', dest='note', type=str, default="", help='note what you want')
    # Mask Setting
    parser.add_argument('--mask_type', dest='mask_type', type=str, default="grid", help='grid, tri')
    parser.add_argument('--varmap_type', dest='varmap_type', type=str, default="notuse", help='notuse, var(warp), warp(var)')
    parser.add_argument('--varmap_threshold', dest='varmap_threshold', type=float, default=0.7, help='0 to 1 , if -1: not use')
    
    # train Setting
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for optimizer')
    
    # Model Setting
    parser.add_argument('--backbone', dest='backbone', type=str, default="convnext_base_in22k", help='models in timm')
    
    parser.add_argument('--D_iter', dest='D_iter', type=int, default=5, help='d iter per batch')
    parser.add_argument('--G_iter', dest='G_iter', type=int, default=1, help='g iter per batch')
    parser.add_argument('--type', dest='type', type=str, default="wgangp", help='GAN LOSS TYPE')
    parser.add_argument('--gp_lambda', dest='gp_lambda', type=int, default=10, help='Gradient penalty lambda hyperparameter')
    
    parser.add_argument('--mask_weight', dest='mask_weight', type=float, default=1.0, help='weight of mask_loss')

    # Dir
    parser.add_argument('--data_dir',dest='data_dir',type=str, help="warppeData dir")
    # parser.add_argument('--train_dir',dest='train_dir',type=str,default="./data/celebAHQ/train_data",help="train dataset dir")
    # parser.add_argument('--val_dir',dest='val_dir',type=str,default="./data/celebAHQ/val_data",help="val dataset dir")
    parser.add_argument('--log_dir', dest='log_dir', default='./log/', help='log dir')
   
    # Other
    parser.add_argument('--wandb',default=False, action="store_true")

    args = parser.parse_args(know_args) if know_args != None else parser.parse_args()
    return args


# In[2]:
import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2
import timm

import random
import scipy.spatial as spatial
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
#
import data_utils

# Seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True






# In[5]:
args = get_args(know_args)
assert len(timm.list_models(args.backbone,pretrained=True)) !=0, print(f"no such backbone {args.backbone} ")
args.image_size = image_size
print(vars(args))


# # Dataset

# In[46]:


def checkallData(data_dir,image_id_list):
    print("Check all data exist")
    origin_dir = f"{data_dir}/origin/"
    warpped_dir = f"{data_dir}/warpped/"
    mask_dir = f"{data_dir}/mask/"
    mesh_dir = f"{data_dir}/mesh/"

    for select_image_id in tqdm(image_id_list):
        if os.path.exists(f"{origin_dir}/{select_image_id}.jpg")             and os.path.exists(f"{warpped_dir}/{select_image_id}.jpg")             and os.path.exists(f"{mask_dir}/{select_image_id}.npy")             and os.path.exists(f"{mesh_dir}/{select_image_id}.npz") :
                continue
        else:
            raise FileNotFoundError(f"{select_image_id} is broken")
            
class WarppedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,
                 image_ids,
                 mask_type,
                 varmap_type,
                 varmap_threshold,
                 transform=None, 
                 return_mesh=False,
                 checkExist=True,
                 debug = False):   
        
        self.image_ids = image_ids
        self.mask_type = mask_type
        self.varmap_type = varmap_type
        self.varmap_threshold = varmap_threshold
          
        self.basic_transform = transforms.Compose(
        [
             transforms.ToTensor(),
             transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transform = transform 
        self.return_mesh = return_mesh
        
        d_dir = f"{data_dir}/{mask_type}/"
        self.origin_dir = f"{d_dir}/origin/"
        self.warpped_dir = f"{d_dir}/warpped/"
        self.mask_dir = f"{d_dir}/mask/"
        self.mesh_dir = f"{d_dir}/mesh/"
        
        self.debug = debug
       
        if checkExist:
            self._check_all_data_exist()
    
    def __len__(self): 
        return len(self.image_ids)
    
    def _check_all_data_exist(self):
        print("Check length:", len(self.image_ids))
        print("Check Data exist")
        for idx in tqdm(range(len( self.image_ids))):
            select_image_id = self.image_ids[idx]
            if os.path.exists(f"{origin_dir}/{select_image_id}.jpg")                 and os.path.exists(f"{warpped_dir}/{select_image_id}.jpg")                 and os.path.exists(f"{mask_dir}/{select_image_id}.npy")                 and os.path.exists(f"{mesh_dir}/{select_image_id}.npz") :
                    continue
            else:
                raise FileNotFoundError(f"{select_image_id} is broken")
                
    def _varmap_selectior(self,origin,warpped,mesh_pts,mesh_tran_pts):
        if self.varmap_type == "notuse":
            return None
        
        if self.mask_type == "grid":
            raise NotImplementedError(f"varmap in grid  not implemented!")
                
        elif self.mask_type == "tri":
            if self.varmap_type == "var(warp)":
                return data_utils.get_tri_varmap(np.array(warpped),mesh_pts)

            elif self.varmap_type == "warp(var)":
                src_image = np.array(origin)
                image_size = (src_image.shape[0], src_image[1])
                tri_varmap = data_utils.get_tri_varmap(src_image, mesh_pts)
                return data_utils.warp_image(tri_varmap, mesh_pts, mesh_tran_pts, image_size)

            else:
                raise NotImplementedError(f"varmap_type {self.varmap_type} not implemented!")
        else:
            raise NotImplementedError(f"mask_type {self.mask_type} not implemented!")
        
        
    
    def __getitem__(self, idx):
        select_image_id = self.image_ids[idx]
        # Get the path to the image 
        origin = Image.open(f"{self.origin_dir}/{select_image_id}.jpg")
        warpped = Image.open(f"{self.warpped_dir}/{select_image_id}.jpg")
        mask = np.load(f"{self.mask_dir}/{select_image_id}.npy")
        mesh_and_mesh_tran = np.load(f"{self.mesh_dir}/{select_image_id}.npz")
        mesh_pts = mesh_and_mesh_tran["mesh"]
        mesh_tran_pts = mesh_and_mesh_tran["mesh_tran"]
        
        if self.transform:
            origin = self.transfrom(origin)
            warpped = self.transfrom(warpped)
            mask = self.transfrom(mask)
        
        varmap = self._varmap_selectior(origin,warpped,mesh_pts,mesh_tran_pts)
        if self.debug:
            if varmap is not None:
                plt.imshow(varmap)
        
        mask = data_utils.mix_mask_var(mask,varmap,threshold=self.varmap_threshold)                 if varmap is not None else mask 
        
        
        origin = self.basic_transform(origin)
        warpped = self.basic_transform(warpped)
        
      
        
        if self.return_mesh:
            return origin, warpped, mesh_pts, mesh_tran_pts, mask
        else:
            return origin, warpped, mask


# # Split Data to train and valid

# In[7]:



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


# # Models

# In[8]:


import torch.nn as nn
import torch.nn.functional as F

class MaskEstimator(nn.Module):
    
        
    def __init__(self,image_size,backbone):
        super().__init__()
        self.image_size = image_size
       
        self.encoder = timm.create_model(backbone, pretrained=True)
        
        
        md_channels = [1024,512,256,128,64,1]
        print("md_channels",md_channels)
        print([ (in_d*2,out_d) for in_d,out_d in zip(md_channels[:-2],md_channels[1:-1])])
        mask_decoder_list = [self._deconv_block(in_d, out_d, drop_rate=0.5) for in_d,out_d in zip(md_channels[:-2],md_channels[1:-1])]
        mask_decoder_list.append(
            nn.ConvTranspose2d(
                in_channels=md_channels[-2],
                out_channels=md_channels[-1],
                kernel_size=4, 
                stride=2,
                padding=1,
            ),
        )
        mask_decoder_list.append(nn.Sigmoid())
        self.mask_decoder = nn.ModuleList(mask_decoder_list)
        
        
        
    def _deconv_block(self,in_dim,out_dim,drop_rate=0):
         return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=4, 
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(drop_rate),
                nn.ReLU()
         )
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        hiddens = []
        # encoder
        latent_code = self.encoder.forward_features(x)
        
        # decoder
        y =  latent_code
        for idx,layer in enumerate(self.mask_decoder):
            y=layer(y)
            # print(y.shape)
        mask = y 
        
        assert mask.shape == (batch_size,1,self.image_size[0],self.image_size[1]), print(mask.shape)
        return mask
        # return reconstruct, mask 


# In[9]:


import torch.nn as nn
import torch.nn.functional as F

class InpaintGenerator(nn.Module):
    
        
    def __init__(self,image_size):
        super().__init__()
        self.image_size = image_size
        channels = [3,64,128,256,512]
        print("channels",channels)
        print([ (in_d,out_d) for in_d,out_d in zip(channels[:-2],channels[1:-1])])
        encoder_list = [ self._conv_block(in_d,out_d) for in_d,out_d in zip(channels[:-2],channels[1:-1])]
        encoder_list.append( 
            nn.Conv2d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=4,
                stride=2,
                padding=1,
        ))
        self.encoder = nn.ModuleList(encoder_list)
        
        
        
        d_channels = channels[::-1]  
        print("d_channels",d_channels)
        print([ (in_d*2,out_d) for in_d,out_d in zip(d_channels[:-2],d_channels[1:-1])])
        decoder_list = [self._deconv_block(in_d*2, out_d, drop_rate=0.5) for in_d,out_d in zip(d_channels[:-2],d_channels[1:-1])]
        decoder_list.append(
            nn.ConvTranspose2d(
                in_channels=d_channels[-2]*2,
                out_channels=d_channels[-1],
                kernel_size=4, 
                stride=2,
                padding=1,
        ))
        self.decoder = nn.ModuleList(decoder_list)
        self.tanh = nn.Tanh()
         

        
    def _deconv_block(self,in_dim,out_dim,drop_rate=0):
         return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=4, 
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(drop_rate),
                nn.ReLU()
         )
    
    def _conv_block(self,in_dim,out_dim,drop_rate=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ), 
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2)
        )  
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        hiddens = []
        # encoder
        y = x 
        for layer in self.encoder:
            y = layer(y)
            hiddens.append(y)
        latent_code = y.clone()
        
        # inpaint decoder
        # y = latent_code.clone()
        reversed_hiddens = hiddens[::-1]
        for idx,layer in enumerate(self.decoder):
            if idx < len(reversed_hiddens):
                y = torch.cat((y,reversed_hiddens[idx]),dim=1)
            y=layer(y)
        reconstruct =self.tanh(y)
        assert reconstruct.shape == (batch_size,3,self.image_size[0],self.image_size[1])
            
        return reconstruct


# In[10]:


import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    
        
    def __init__(self,image_size,backbone):
        super().__init__()
        self.image_size = image_size
        self.MG = MaskEstimator(image_size = image_size, backbone = backbone)
        self.IG = InpaintGenerator(image_size = image_size)
   
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        
        fake_masks = self.MG(x)
        reconstructs = self.IG(x * fake_masks)
        
        
            
        return reconstructs,fake_masks


# In[11]:


class Discriminator(nn.Module):
    def __init__(self,image_size):
        super().__init__()
        self.image_size = image_size
        channels = [3,64,128,256,512]
        print("channels",channels)
        print([ (in_d,out_d) for in_d,out_d in zip(channels[:-2],channels[1:-1])])
        encoder_list = [ self._conv_block(in_d,out_d) for in_d,out_d in zip(channels[:-2],channels[1:-1])]
        encoder_list.append( 
            nn.Conv2d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=4,
                stride=1,
                padding=1,
        ))
        self.encoder = nn.ModuleList(encoder_list)
        
        predict_in_dim = self._calculate_predictor_input_dim()
        print("predict_in_dim",predict_in_dim)
        self.predictor = nn.Sequential(
            nn.Linear(  predict_in_dim, 1, bias=True), 
        )
        
    def _calculate_predictor_input_dim(self):
        batch_size = 1
        x = torch.randn((batch_size,3,self.image_size[0],self.image_size[1]))
        y = x 
        for layer in self.encoder:
            y = layer(y)
        out = y.flatten(start_dim = 1)
        
        self.zero_grad()
        return out.shape[1]
        
    def _conv_block(self,in_dim,out_dim,drop_rate=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ), 
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2)
        )  

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        
        y = x 
        for layer in self.encoder:
            y = layer(y)
        
        out = y.flatten(start_dim = 1)

        assert out.shape[0] == batch_size
        out = self.predictor(out) 
        return out



# # Utils function

# In[32]:


def check_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[33]:


to_pillow_f = torchvision.transforms.ToPILImage()


# In[34]:


def calc_gradient_penalty(D, real_images, fake_images, device ,args):
    
    batch_size = fake_images.shape[0]
    # cuda = bool(device == "cuda")

    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)
  

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    interpolated = interpolated.to(device)
   

    # define it to calculate gradient
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)
   
    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return grad_penalty


# # trainning

# In[35]:


if args.wandb:
    import wandb
    wandb.init(project="warp_inpaint", entity='kycj')
    args.log_dir = f"{args.log_dir}/{wandb.run.id}/"
    wandb.config.update(args)
    wandb.run.name = f"{wandb_prefix_name}_{wandb.run.name}"
    
    print(vars(args))
    


# In[36]:


""" result dir setting """
checkpoint_dir = args.log_dir + "./ckpts/"
check_create_dir(checkpoint_dir)
check_create_dir(checkpoint_dir+"/best/")
sample_dir = args.log_dir + "./samples/"
check_create_dir(sample_dir)


# In[52]:


""" Split train valid Data """ 
# move forward block

# d_dir = f"{args.data_dir}/{args.mask_type}/"
# origin_dir = f"{d_dir}/origin/"
# image_names = natsorted(os.listdir(origin_dir))
# image_id_list = list(map(lambda s: s.split('.')[0], image_names))
# print(len(image_id_list))

# # checkallData(d_dir,image_id_list)
# print("Seed:",seed)
# train_ids, valid_ids = train_test_split(image_id_list , test_size=0.25, random_state=seed)
# print("Total train data:",len(train_ids))
# print("Total valid data:", len(valid_ids))

""" Data """
trainset = WarppedDataset(
                 args.data_dir,
                 train_ids,
                 args.mask_type,
                 args.varmap_type,
                 args.varmap_threshold,
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
val_batch_num = len(val_loader) # 要用幾個batch來驗證 ,  len(val_loader) 個batch 的話就是全部的資料
# print("val_loader",len(val_loader), "src:", args.val_dir)
print("num data per valid:",val_batch_num* args.batch_size)


# In[53]:


from tqdm import tqdm
# def train(G,D,train_loader,val_dataloader,args):

device = "cuda"

D_iter = args.D_iter
G_iter = args.G_iter

# wgan parameters
assert D_iter > G_iter,print("WGAN Need D_iter > G_iter")
weight_cliping_limit = 0.01

""" Model """
G = Generator(image_size = image_size, backbone = args.backbone)
D = Discriminator(image_size = image_size)
G = G.to(device)
D = D.to(device)
""" Optimizer """
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr,betas=(0.5,0.99))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr,betas=(0.5,0.99))

""" Loss function """
# adversarial_loss= nn.BCELoss()
l1_loss_f = torch.nn.L1Loss()


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
            origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks = batch_data
            masks = masks.permute(0,3,1,2)
            # print("masks",masks.shape)
            origin_imgs, warpped_imgs = origin_imgs.to(device), warpped_imgs.to(device)
            masks = masks.to(device)

            origin_list = origin_imgs.reshape((args.D_iter,-1,3,image_size[0],image_size[1]))
            warpped_list = warpped_imgs.reshape((args.D_iter,-1,3,image_size[0],image_size[1]))
            masks_list = masks.reshape((args.D_iter,-1,1,image_size[0],image_size[1]))
            
            
            
            # WGAN - Training discriminator more iterations than generator
            """ UPDATE Discriminator """
            for p in D.parameters():  # reset requires_grad
                p.requires_grad = True
            
            for j in range(D_iter):
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
                
            for _ in range(G_iter):
                optimizer_G.zero_grad()
                # origin, warpped
                i = np.random.randint(0, args.D_iter)
                origin,warpped = origin_list[i], warpped_list[i]
                gt_masks = masks_list[i]
                assert origin.shape == (args.batch_size,3,image_size[0],image_size[1])
                assert warpped.shape == (args.batch_size,3,image_size[0],image_size[1])
                assert gt_masks.shape == (args.batch_size,1,image_size[0],image_size[1])
                
                # fake_loss
                fake_images, fake_masks = G(warpped)
                fake_logits = D(fake_images)
                fake_loss = fake_logits.mean(0).view(1)
                
                # l1_loss
                l1_loss = l1_loss_f(fake_images, origin).mean() 
                
                # matt_loss
                matt_loss = l1_loss_f(fake_masks * warpped, fake_masks * origin).mean()
                
                # mask_loss 
                mask_loss = l1_loss_f(fake_masks, gt_masks).mean()

                # genreator loss
                # g_loss = - fake_loss + l1_loss * abs(fake_loss) + matt_loss + mask_loss
                g_loss = - fake_loss + ( l1_loss + 100*matt_loss + mask_loss) * abs(fake_loss) 
                g_loss.backward()
                optimizer_G.step()


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
                    "Wasserstein_D":Wasserstein_D.item()
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
                    val_g_loss = []
                    val_fake_loss = []
                    val_matt_loss = []
                    val_l1_loss = []
                    val_mask_loss = []
                    # origin,warpped,fake_images,fake_masks,gt_masks = None,None,None,None,None
                    
                    for idx, batch_data in enumerate(val_loader):
                        if idx == val_batch_num:
                            break
                        # print(f"{idx}/{len(val_loader)}")
                    # for _ in range(1):
                        # batch_data = next(iter(val_loader))
                    
                        origin_imgs, warpped_imgs, origin_meshes, warpped_meshes, masks = batch_data
                        masks = masks.permute(0,3,1,2)
                        origin_imgs, warpped_imgs = origin_imgs.to(device), warpped_imgs.to(device)
                        masks = masks.to(device)
                        
                        origin,warpped = origin_imgs, warpped_imgs
                        gt_masks = masks
                        assert origin.shape == (args.batch_size,3,image_size[0],image_size[1])
                        assert warpped.shape == (args.batch_size,3,image_size[0],image_size[1])
                        assert gt_masks.shape == (args.batch_size,1,image_size[0],image_size[1])
                        
                        # fake_loss
                        fake_images, fake_masks = G(warpped)
                        fake_logits = D(fake_images)
                        fake_loss = fake_logits.mean(0).view(1)

                        # l1_loss
                        l1_loss = l1_loss_f(fake_images, origin).mean() 

                        # matt_loss
                        matt_loss = l1_loss_f(fake_masks * warpped, fake_masks * origin).mean()

                        # mask_loss 
                        mask_loss = l1_loss_f(fake_masks, gt_masks).mean()

                        # genreator loss
                        # g_loss = - fake_loss + l1_loss * abs(fake_loss) + matt_loss + mask_loss
                        g_loss = (- 1 + l1_loss + 100*matt_loss + args.mask_weight*mask_loss) * abs(fake_loss) 
                        
                        
                        val_fake_loss.append(fake_loss.item())
                        val_l1_loss.append(l1_loss.item())
                        val_matt_loss.append(matt_loss.item())
                        val_mask_loss.append(mask_loss.item())
                        val_g_loss.append(g_loss.item())
                    
                    # print("np.array(val_mask_loss)",np.array(val_mask_loss).shape)
                    log_dict["val"] ={
                        "g_loss": np.array(val_g_loss).mean(),
                        "g_fake_loss": np.array(val_fake_loss).mean(),
                        "g_l1_loss": np.array(val_l1_loss).mean(),
                        "matt_loss":  np.array(val_matt_loss).mean(),
                        "mask_loss": np.array(val_mask_loss).mean()
                    }
                    
                    # print("log_dict[val]",log_dict["val"])

                    k = np.random.randint(0, len(origin))
                    img_path = f"{sample_dir}/sample_{step}_{epoch}.jpg"
                    # to_pillow_f(fake_images[k]).save(img_path)

                    # plot result
                    fig, axs = plt.subplots(1, 6, figsize=(32,8))
                    axs[0].set_title('origin')
                    axs[0].imshow( to_pillow_f(origin[k]) )
                    
                    axs[1].set_title('warpped')
                    axs[1].imshow( to_pillow_f(warpped[k]) )
                    
                    axs[2].set_title('fakeImage')
                    axs[2].imshow( to_pillow_f(fake_images[k]) )

                    axs[3].set_title('fakeMask')
                    axs[3].imshow( to_pillow_f(fake_masks[k]),vmin=0, vmax=1,cmap='gray' )
                    
                    axs[4].set_title('GTMask')
                    axs[4].imshow( to_pillow_f(gt_masks[k]),vmin=0, vmax=1,cmap='gray' )
                    
                    axs[5].set_title('GTMask on warpped')
                    axs[5].imshow( to_pillow_f(gt_masks[k]*warpped[k]))

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