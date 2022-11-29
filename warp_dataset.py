import data_utils
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from custom_transforms import ComposeWithMultiTensor,RandomApplyWithMultiTensor,RandomResizedCropWithMultiTensor,RandomHorizontalFlipWithMultiTensor,RandomVerticalFlipWithMultiTensor,RandomGrayscaleWithMultiTensor
basic_transform = transforms.Compose(
[
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
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
                 inverse = False,
                 no_mesh = False,
                 mask_threshold= -1,
                 use_resize_crop=False,
                 use_custom_transform=False,
                 lpips_threshold = 0.0,
                 use_dct=False):   
        
        self.use_dct = use_dct

        unmask_value = 0 if inverse else 1
        
        self.lpips_filter = None
        if lpips_threshold > 0:
            self.lpips_filter = data_utils.LPIPS_filter(lpips_threshold,unmask_value = unmask_value)
        
        self.use_resize_crop = use_resize_crop

        self.use_mix = True if mask_type == "mix_tri_tps" else False

        self.use_mesh = not no_mesh      
        self.inverse = inverse
        self.image_ids = image_ids
        self.mask_type = mask_type
        self.varmap_type = varmap_type
        self.varmap_threshold = varmap_threshold
        self.mask_threshold = mask_threshold
        self.guassian_blur_f = guassian_blur_f
          
        self.basic_transform = basic_transform
        
        self.transform = transform 
        self.use_custom_transform = use_custom_transform
        print("self.use_custom_transform",self.use_custom_transform)

        self.return_mesh = return_mesh
        
        d_dir = f"{data_dir}/{mask_type}/"
        self.data_dir = data_dir
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
        # else         
        if self.varmap_type == "small_grid":
                mesh_size_for_varmap = 8 
                src_image = np.array(warpped)
                image_size = (src_image.shape[0], src_image.shape[1])
                mesh_for_varmap = data_utils.create_mesh(image_size= image_size, mesh_size = mesh_size_for_varmap)
                small_grid_varmap = data_utils.get_var_map(src_image,mesh_for_varmap)
                return small_grid_varmap
        else:
            raise NotImplementedError(f"varmap_type {self.varmap_type} not implemented!")
        
    
    def __getitem__(self, idx):
        select_image_id = self.image_ids[idx]
        if self.use_mix:
            mask_type_list = ['tri','tps_dgrid_p16']
            select_mask_type = mask_type_list[np.random.randint(len(mask_type_list))]
            d_dir = f"{self.data_dir}/{select_mask_type}/"
            self.origin_dir = f"{d_dir}/origin/"
            self.warpped_dir = f"{d_dir}/warpped/"
            self.mask_dir = f"{d_dir}/mask/"
            self.mesh_dir = f"{d_dir}/mesh/"
            self.mask_type = select_mask_type

        # Get the path to the image 
        origin_path = f"{self.origin_dir}/{select_image_id}.jpg"
        origin = Image.open(origin_path)
        warpped_path = f"{self.warpped_dir}/{select_image_id}.jpg"
        warpped = Image.open(warpped_path)
        mask = np.load(f"{self.mask_dir}/{select_image_id}.npy")

        if self.use_dct: # dct only y channel
            origin_dct_list, origin_qtables = data_utils.get_jpeg_info(origin_path)
            origin_dct = torch.from_numpy(origin_dct_list[0]).unsqueeze(0)
            origin_qtables = torch.from_numpy(origin_qtables[0]).unsqueeze(0)

            warpped_dct_list, warpped_qtables = data_utils.get_jpeg_info(warpped_path)
            warpped_dct = torch.from_numpy(warpped_dct_list[0]).unsqueeze(0)
            warpped_qtables = torch.from_numpy(warpped_qtables[0]).unsqueeze(0)
            
            
        
        if self.use_mesh:
            mesh_and_mesh_tran = np.load(f"{self.mesh_dir}/{select_image_id}.npz")
            mesh_pts = mesh_and_mesh_tran["mesh"]
            mesh_tran_pts = mesh_and_mesh_tran["mesh_tran"]
        else:
            mesh_pts, mesh_tran_pts = torch.zeros(1),torch.zeros(1)
      
        
        varmap = self._varmap_selector(origin,warpped,mesh_pts,mesh_tran_pts)
        if self.debug:
            if varmap is not None:
                plt.imshow(varmap)
                plt.savefig('./varmap_sample.jpg')

        # add threshold for mask
        if self.mask_threshold != -1:
            mask[mask >= self.mask_threshold] = 1
            mask[mask <  self.mask_threshold] = 0
        
        # mix mask and var
        mask = data_utils.mix_mask_var(mask,varmap,threshold=self.varmap_threshold)   if varmap is not None else mask 
        if self.guassian_blur_f:
            mask = self.guassian_blur_f(mask)
        if varmap is not None:
            varmap = torch.from_numpy(varmap).permute(2,0,1).to(dtype = torch.float32)
        mask = torch.from_numpy(mask).permute(2,0,1).to(dtype = torch.float32)
        
        origin = self.basic_transform(origin)
        warpped = self.basic_transform(warpped)
        
        if self.inverse:
            mask = 1. - mask  # 原本的code 是 0 為mask 區域, Platte 則是 1 為mask區域
            
        if self.lpips_filter:
            mask = self.lpips_filter(mask,origin,warpped)

        if self.transform:
            origin =self.transform(origin)
            # print("origin",origin.shape)
            warpped =self.transform(warpped)
            mask =self.transform(mask)
            # mesh_pts =self.transform(mesh_pts)
            # mesh_tran_pts =self.transform(mesh_tran_pts)

            if varmap is not None:
                varmap = self.transform(varmap)
            if self.use_dct: # dct only y channel
                origin_dct = self.transform(origin_dct)
                warpped_dct = self.transform(warpped_dct)
        
       
                
       

        if self.use_custom_transform:
            transforms_f = ComposeWithMultiTensor([
                RandomGrayscaleWithMultiTensor(p=0.5),
                RandomVerticalFlipWithMultiTensor(p=0.5),
                RandomHorizontalFlipWithMultiTensor(p=0.5),
                RandomApplyWithMultiTensor([RandomResizedCropWithMultiTensor(size = (origin.shape[-2],origin.shape[-1]))], p=0.5)
            ])
            if varmap is not None:
                if self.use_dct:
                    origin, warpped, mask, varmap, origin_dct, warpped_dct  = transforms_f([origin, warpped, mask, varmap, origin_dct, warpped_dct ])
                else:                        
                    origin, warpped, mask, varmap = transforms_f([origin, warpped, mask, varmap])
                    varmap = torch.clamp(varmap, min=0, max=1)

            else:
                if self.use_dct:
                    origin, warpped, mask, origin_dct, warpped_dct = transforms_f([origin, warpped, mask, origin_dct, warpped_dct])
                else:
                    origin, warpped, mask = transforms_f([origin, warpped, mask])
            
            mask = torch.clamp(mask, min=0, max=1)

        else:

            if self.use_resize_crop:
                crop_f = RandomResizedCropWithMultiTensor(size = (origin.shape[-2],origin.shape[-1]))
                if varmap is not None:
                    origin, warpped, mask, varmap = crop_f([origin, warpped, mask, varmap])
                    varmap = torch.clamp(varmap, min=0, max=1)
                else:
                    origin, warpped, mask = crop_f([origin, warpped, mask])
                
                mask = torch.clamp(mask, min=0, max=1)
                
                
      
            
         
            
        
        if self.return_mesh:
            if self.use_dct:
                warpped_dct = data_utils.to_dct_volume(warpped_dct)
                origin_dct = data_utils.to_dct_volume(origin_dct)
                return origin, warpped, mesh_pts, mesh_tran_pts, mask, torch.empty(mask.shape), origin_dct, warpped_dct , origin_qtables, warpped_qtables
            if varmap is not None:
                return origin, warpped, mesh_pts, mesh_tran_pts, mask, varmap
            else :
                return origin, warpped, mesh_pts, mesh_tran_pts, mask, torch.empty(mask.shape)
        else:
            return origin, warpped, mask
