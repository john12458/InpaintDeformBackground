import data_utils
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class WarppedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,
                 image_ids,
                 mask_type,
                 varmap_type,
                 varmap_threshold,
                 guassian_blur_f,
                 transform=None, 
                 return_mesh=False,
                 checkExist=True,
                 debug = False,
                 inverse = False):   
        self.inverse = inverse
        self.image_ids = image_ids
        self.mask_type = mask_type
        self.varmap_type = varmap_type
        self.varmap_threshold = varmap_threshold
        self.guassian_blur_f = guassian_blur_f
          
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
            if os.path.exists(f"{self.origin_dir}/{select_image_id}.jpg") \
                and os.path.exists(f"{self.warpped_dir}/{select_image_id}.jpg") \
                    and os.path.exists(f"{self.mask_dir}/{select_image_id}.npy") \
                        and os.path.exists(f"{self.mesh_dir}/{select_image_id}.npz") :
                    continue
            else:
                raise FileNotFoundError(f"{select_image_id} is broken")
                
    def _varmap_selector(self,origin,warpped,mesh_pts,mesh_tran_pts):
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
            elif self.varmap_type == "small_grid":
                mesh_size_for_varmap = 8 
                src_image = np.array(warpped)
                image_size = (src_image.shape[0], src_image.shape[1])
                mesh_for_varmap = data_utils.create_mesh(image_size= image_size, mesh_size = mesh_size_for_varmap)
                small_grid_varmap = data_utils.get_var_map(src_image,mesh_for_varmap)
                return small_grid_varmap
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
        
        varmap = self._varmap_selector(origin,warpped,mesh_pts,mesh_tran_pts)
        if self.debug:
            if varmap is not None:
                plt.imshow(varmap)
                plt.savefig('./varmap_sample.jpg')

        mask = data_utils.mix_mask_var(mask,varmap,threshold=self.varmap_threshold)   if varmap is not None else mask 
        if self.guassian_blur_f:
            mask = self.guassian_blur_f(mask)
        
        origin = self.basic_transform(origin)
        warpped = self.basic_transform(warpped)
        
        if self.inverse:
            mask = 1. - mask  # 原本的code 是 0 為mask 區域, Platte 則是 1 為mask區域
      
        
        if self.return_mesh:
            return origin, warpped, mesh_pts, mesh_tran_pts, mask, varmap
        else:
            return origin, warpped, mask
