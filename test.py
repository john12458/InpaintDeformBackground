import torch
from warp_dataset import basic_transform
from utils import to_pillow_f
from PIL import Image
import torchvision.transforms as  transforms
import torchvision.transforms.functional as  F

@torch.no_grad()
def test_one(path,G,image_size,mask_img_f,device, return_mask=False):
    centor_crop_f = transforms.CenterCrop(size=image_size)
    warpped_pillow = Image.open(path).convert('RGB')
    
    """ preprocess v1""" 
    origin_size = max(warpped_pillow.size[0],warpped_pillow.size[1])
    scale_factor = min(image_size[0],image_size[1]) / origin_size
    img_pillow_warpped = F.affine(warpped_pillow, scale = scale_factor, angle = 0.0, translate=(0,0),shear=0.0)            
    warpped_centor_cropped = centor_crop_f(img_pillow_warpped)
    
    # """ preprocess v2 """ 
    # warpped_centor_cropped = centor_crop_f(warpped_pillow)

    test_image_tensor = basic_transform(warpped_centor_cropped) 
    test_image_tensor = test_image_tensor.unsqueeze(0).to(device)

    fake_masks = G(test_image_tensor)
    if return_mask:
        return fake_masks
    else:
        return to_pillow_f(mask_img_f(fake_masks[0],test_image_tensor[0]))
