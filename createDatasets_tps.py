#!/usr/bin/env python
# coding: utf-8

# In[27]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# # Classes

# %%


import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2


# %%


#. https://github.com/alyssaq/face_morpher/blob/dlib/facemorpher/warper.py
import numpy as np
import scipy.spatial as spatial



import thinplate as tps_pytotch
def tps_warp(pillow_img,src_pts,target_pts):
    img = np.array(pillow_img)
        
    dshape = img.shape[:2]
    theta = tps_pytotch.tps_theta_from_points(src_pts, target_pts, reduced=True)
    grid, delta_grid = tps_pytotch.tps_grid(theta, target_pts, dshape)
    # print(z.shape)
    mapx, mapy = tps_pytotch.tps_grid_to_remap(grid, img.shape)
    warped =cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
    warped_delta_grid = cv2.remap(delta_grid, mapx, mapy, cv2.INTER_CUBIC)
    # warped_pillow = Image.fromarray(warped)
    warped_pillow = Image.fromarray(np.clip(warped, 0.0, 255.0).astype(np.uint8))
    return warped_pillow, grid, delta_grid, warped_delta_grid, mapx, mapy

def get_neighbor_diff_guassian(input_tensor, kernel_size= 9, sigma = 1, channels = 2):

    import math

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # gaussian_kernel[kernel_size//2,kernel_size//2] =  0
    gaussian_kernel *= -1
    gaussian_kernel[kernel_size//2,kernel_size//2] =  gaussian_kernel.sum().abs()
    

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    
    padding = math.ceil((kernel_size - 1) /2)
    dx = torch.nn.functional.conv2d(input_tensor[:,0,:,:].unsqueeze(0), gaussian_kernel, bias=None,stride=1, padding =padding)
    dy = torch.nn.functional.conv2d(input_tensor[:,1,:,:].unsqueeze(0), gaussian_kernel, bias=None,stride=1, padding =padding)
    result = torch.stack([dx[:,0,:,:], dy[:,0,:,:] ],dim=1)
    return result

class TPSWarp:
    
    def __init__(self, 
                mesh_size = 18 ,
                num_vertex_wanted_to_move=10, 
                warp_min_factor=1,
                warp_max_factor=3,
                bounding_sample=1,
                use_neighbor=True,
                use_normlize = True
            ):
        self.mesh_size = mesh_size
        self.num_vertex_wanted_to_move = num_vertex_wanted_to_move
        self.warp_min_factor = warp_min_factor
        self.warp_max_factor = warp_max_factor
        self.use_neighbor = use_neighbor
        self.bounding_sample = bounding_sample
        self.use_normlize = use_normlize
        
    def __call__(self, pillow_img):
        src_img = np.array(pillow_img)
        image_size = (src_img.shape[0],src_img.shape[1])

        # Create Mesh
        x = np.linspace(0,  image_size[0], image_size[0]//self.mesh_size)
        y = np.linspace(0,  image_size[1], image_size[1]//self.mesh_size)
        xv, yv = np.meshgrid( y,x)
        mesh = np.concatenate( (xv[np.newaxis,:], yv[np.newaxis,:]) )
        mesh = np.int32(mesh)
        mesh_pts = mesh.reshape(2,-1).T
    
        # Create Deform Mesh
        sample_pts_mesh_xy = np.concatenate(
            (
                np.random.randint( self.bounding_sample,mesh.shape[1]-self.bounding_sample, size=(self.num_vertex_wanted_to_move, 1)),
                np.random.randint( self.bounding_sample,mesh.shape[2]-self.bounding_sample, size=(self.num_vertex_wanted_to_move, 1))
            ),axis=1
        )
        
        src_pts = []
        target_pts = []
        mesh_tran = mesh.copy()
        sample_pts = []
        for x, y in sample_pts_mesh_xy:     
            effect = 0.1 * np.random.randint(self.warp_min_factor,self.warp_max_factor,size = 2)
            sign = 2 * np.random.randint(2,size = 2) -1 
            shift_vector = np.ones(2)
            shift_vector[0] *=  self.mesh_size * effect[0] * sign[0]
            shift_vector[1] *=  self.mesh_size * effect[1] * sign[1]
            mesh_tran[:,x,y] += np.int32(shift_vector)
            sample_pts.append(mesh_tran[:,x,y])
        sample_pts = np.stack(sample_pts)
        sample_pts = np.array(sample_pts, dtype=np.int32)
        
        mesh_tran = np.array(mesh_tran, dtype= np.int32)
        mesh_tran_pts =mesh_tran.reshape(2,-1).T
        
        for m, mt in zip(mesh_pts, mesh_tran_pts):
            src_pts.append(m)
            target_pts.append(mt)
        
        src_pts = np.array(np.array(src_pts,dtype=np.int32) / image_size[0], dtype=np.float64)
        target_pts = np.array(np.array(target_pts,dtype=np.int32) / image_size[0], dtype=np.float64)
        
        warpped_img, grid, delta_grid, warpped_delta_grid, mapx, mapy = tps_warp(pillow_img,src_pts,target_pts)
        identity_warp =  tps_warp(pillow_img,src_pts,src_pts)[0]
        
        mask = None
        # print("warpped_delta_grid",warpped_delta_grid.shape)
        if self.use_neighbor:
            warpped_delta_grid_tensor = torch.from_numpy(warpped_delta_grid).unsqueeze(0).permute(0,3,1,2).to(torch.float32)
            neighbor_warpped_delta_grid = get_neighbor_diff_guassian(warpped_delta_grid_tensor, kernel_size= self.mesh_size-1).permute(0,2,3,1).float().numpy()
            neighbor_warpped_delta_grid = neighbor_warpped_delta_grid[0]
            mask = np.linalg.norm(neighbor_warpped_delta_grid, axis=2)[...,np.newaxis]
            # print("mask",mask.shape)
        else:
            mask = np.linalg.norm(warped_delta_grid, axis=2)[...,np.newaxis]
        
        if self.use_normlize:
            min_max_norm_f = lambda x : (x - x.min()) / (x.max() - x.min())
            mask = min_max_norm_f(mask)

        mask = 1 - mask
        
        return warpped_img, mesh_pts, mesh_tran_pts, mask, warpped_delta_grid
        # return warpped_img, src_pts, target_pts, warpped_delta_grid, neighbor_warpped_delta_grid, mapx, mapy,identity_warp
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

# %%


def check_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# %%


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, warp_f,
                 root_dir, 
                 image_size=(256,256),
                 transform=None, 
                 return_mesh=False,
                 savePath=False):
        """
        Args:
        root_dir (string): Directory with all the images
        transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        # self.resize = transforms.Resize(size= image_size)
        self.image_size = image_size
        self.centerCrop = transforms.CenterCrop(size= image_size)
        self.basic_transform = transforms.Compose(
        [
             transforms.ToTensor(),
             transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])
        
        self.transform = transform 
        self.image_names = natsorted(image_names)
        
        self.warp_f = warp_f
        
        self.return_mesh = return_mesh
        
        self.savePath = savePath
        self.origin_dir = f"{savePath}/origin/"
        check_create_dir(self.origin_dir)
        self.warpped_dir = f"{savePath}/warpped/"
        check_create_dir(self.warpped_dir)
        self.mask_dir = f"{savePath}/mask/"
        check_create_dir(self.mask_dir)
        self.mesh_dir = f"{savePath}/mesh/"
        check_create_dir(self.mesh_dir)
        
        self.warp_dgrid_dir = f"{savePath}/warp_dgrid/"
        check_create_dir(self.warp_dgrid_dir)
        
        
        self._test_for_open_img()
        
        
        
    def __len__(self): 
        return len(self.image_names)
    
    def _test_for_open_img(self):
        img_path = os.path.join(self.root_dir, self.image_names[np.random.randint(len(self.image_names))])
        img_pillow_origin = Image.open(img_path).convert('RGB')
        img_pillow_array = np.array(img_pillow_origin)
        print("[test_for_open_img]",img_pillow_array.shape)
        
    
    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        
        # Load image and convert it to RGB
        img_pillow_origin = Image.open(img_path).convert('RGB')        
        if img_pillow_origin.size[0] < self.image_size[0]  and img_pillow_origin.size[1] < self.image_size[1]:
            origin_size = max(img_pillow_origin.size[0],img_pillow_origin.size[1])
            scale_factor = min(self.image_size[0],self.image_size[1]) / origin_size
            scale_size = [ int(img_pillow_origin.size[1] * scale_factor) ,int(img_pillow_origin.size[0] * scale_factor) ]
            # print("img_pillow_origin.size",img_pillow_origin.size)
            # print("scale_factor",scale_factor, "scale_size ",scale_size )
            img_pillow_origin = torchvision.transforms.Resize(size = scale_size)(img_pillow_origin)
        # plt.imshow(img_pillow_origin)
        # print("img_pillow_origin",img_pillow_origin.size)
            
        img_pillow = img_pillow_origin
        if self.transform:
            img_pillow = self.transfrom(img_pillow)
        
        img_pillow = self.centerCrop(img_pillow)
        padding_mask_pillow = self.centerCrop(Image.fromarray(np.ones_like(img_pillow_origin)*255))
        padding_mask = (np.array(padding_mask_pillow)/255)[...,0][...,np.newaxis]
        
        
            
        if self.return_mesh:
            warpped_img, mesh_no_last_row, mesh_trans_no_last_row,mask, warpped_delta_grid = self.warp_f(img_pillow)
           
            
            
            if self.savePath:
                # img_pillow = self.centerCrop(img_pillow)
                origin_path = f"{self.origin_dir}/{self.image_names[idx]}"
                img_pillow.save(origin_path)
                
                # warpped_img = self.centerCrop(warpped_img)
                warpped_img = Image.fromarray(np.uint8(np.array(warpped_img) * padding_mask))
                
                warpped_path = f"{self.warpped_dir}/{self.image_names[idx]}"
                warpped_img.save(warpped_path )
                
                # mask = (np.array( \
                #         self.centerCrop( \
                #             Image.fromarray(np.uint8(mask * 255)[:,:,0]) \
                #         ) \
                #     )/255)[...,np.newaxis]
                
                inv_mask = np.abs(mask - 1)
                inv_padded_mask = inv_mask * padding_mask 
                mask = padded_mask = np.abs(inv_padded_mask - 1)
                
                mask_path = f"{self.mask_dir}/{self.image_names[idx].split('.')[0]}"
                mask = np.float32(mask)
                np.save(mask_path,mask)
                
                
                warpped_delta_grid_path = f"{self.warp_dgrid_dir}/{self.image_names[idx].split('.')[0]}"
                warpped_delta_grid = np.int32(warpped_delta_grid*255)
                np.save( warpped_delta_grid_path, warpped_delta_grid)
                
                # mesh_pts,mesh_tran_pts
                # mesh_path = f"{self.mesh_dir}/{self.image_names[idx].split('.')[0]}.npz"
                # np.savez(mesh_path, mesh=mesh_no_last_row, mesh_tran=mesh_trans_no_last_row)
                
                
                print("origin:",origin_path)
                print("warpped:", warpped_path)
                print("mask_path", mask_path)
                print("warpped_delta_grid_path:",warpped_delta_grid_path)
                # print("mesh_path", mesh_path)
                print("---")
            
            img = self.basic_transform(img_pillow) 
            warpped_img = self.basic_transform(warpped_img)
            
            
               
                
            return img
        else:
            warpped_img = self.warp_f(img_pillow)
            warpped_img = self.basic_transform(warpped_img) 
            return img


# # Run

# %%
image_size = (512,512)
args = type('', (), {})()
args.mask_type = "tps_dgrid"
args.varmap_type = "notuse"
args.varmap_threshold = -1

src_data_dir = "/workspace/inpaint_mask/data/CIHP/instance-level_human_parsing/Training/Images/"
target_data_dir = f"/workspace/inpaint_mask/data/warpData/CIHP/Training/{args.mask_type}/"

# src_data_dir = "/workspace/inpaint_mask/data/fashionLandmarkDetectionBenchmark/"
# target_data_dir = f"/workspace/inpaint_mask/data/warpData/fashionLandmarkDetectionBenchmark/{args.mask_type}/"

# check_create_dir(f"{target_data_dir}")
# origin_dir = f"{target_data_dir}/origin"
# os.symlink(src_data_dir, f"{target_data_dir}/origin")


# %%
class RandTPS():
    def __init__(self):
        self.warp_f_list=[
            TPSWarp(mesh_size = 64, warp_max_factor=3,num_vertex_wanted_to_move=3,bounding_sample=2,use_normlize=True),
            TPSWarp(mesh_size = 48, warp_max_factor=4,num_vertex_wanted_to_move=4,bounding_sample=2,use_normlize=True),
            TPSWarp(mesh_size = 32, warp_max_factor=6,num_vertex_wanted_to_move=6,bounding_sample=3,use_normlize=True)
            # TPSWarp(mesh_size = 32, warp_max_factor=3,num_vertex_wanted_to_move=3,bounding_sample=2),
            # TPSWarp(mesh_size = 24, warp_max_factor=4,num_vertex_wanted_to_move=4,bounding_sample=2),
            # TPSWarp(mesh_size = 16, warp_max_factor=6,num_vertex_wanted_to_move=5,bounding_sample=3)
        ]
    def __call__(self, pillow_img):
        idx = np.random.randint(3)
        return self.warp_f_list[idx](pillow_img)


# In[28]:


batch_size = 16
num_workers= 16
warp_f = RandTPS()
dataset = CelebADataset(warp_f,
                        root_dir=src_data_dir,
                        image_size=image_size,
                        return_mesh=True,
                        savePath=target_data_dir )
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=num_workers)


# In[2]:


# %%
print("start to warp")
cnt = 0
for data in data_loader:
    cnt += data.shape[0]
    # break
print("total length:",cnt)

# %%


# In[37]:


# In[3]:


sample_id = "img_00000001"

origin = Image.open(f"{target_data_dir}/origin/{sample_id}.jpg")
warpped = Image.open(f"{target_data_dir}/warpped/{sample_id}.jpg")
mask = np.load(f"{target_data_dir}/mask/{sample_id}.npy")
wdgrid = np.load(f"{target_data_dir}/warp_dgrid/{sample_id}.npy")
# mesh_tran_pts = np.load(f"{target_data_dir}/mesh/{sample_id}.npz")["mesh_tran"]


# In[38]:


# wdgrid.shape


# In[40]:


# plt.imshow(wdgrid[...,0],cmap='gray')
# plt.colorbar()


# In[41]:


# plt.imshow(wdgrid[...,1],cmap='gray')
# plt.colorbar()


# In[35]:


# plt.imshow(mask,cmap='gray')
# plt.colorbar()


# In[32]:


# In[5]:


# %%


# (np.array(
#     Image.fromarray(np.uint8(mask * 255)[:,:,0]))
#     /255)[...,np.newaxis].shape


# %%


fig, axs = plt.subplots(2, 3, figsize=(16,8))
axs[0,0].imshow(origin)
axs[0,1].imshow(warpped)
axs[0,2].imshow(mask ,cmap="gray")

axs[1,0].imshow( Image.fromarray(np.uint8(np.array(origin) * mask)))
axs[1,1].imshow( Image.fromarray(np.uint8(np.array(warpped) * mask)))
# axs[1,2].imshow(np.ones_like(origin)*255)
axs[1,2].imshow(mask ,cmap="gray")
# axs[1,2].triplot(mesh_tran_pts[:,0], mesh_tran_pts[:,1], spatial.Delaunay(mesh_tran_pts).simplices.copy())
fig.savefig(f"{target_data_dir}/sample.jpg")


# %%


# center_crop_f = transforms.CenterCrop(size= (512,512))
# img_pillow_origin = Image.open('/workspace/inpaint_mask/data/CIHP/instance-level_human_parsing/Training/Images/0000006.jpg')
# img_pillow = center_crop_f(img_pillow_origin)
# padding_mask_pillow= center_crop_f(Image.fromarray(np.ones_like(img_pillow_origin)*255))
# padding_mask = (np.array(padding_mask_pillow)/255)[...,0][...,np.newaxis]
# Image.fromarray(np.uint8(np.ones_like(img_pillow)*255 * padding_mask))
# # img_pillow


# %%


# origin.size


# %%


# In[ ]:


# In[ ]:




