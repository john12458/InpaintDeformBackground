import torch
import torch.nn as nn
from .generators.mask_estimator import MaskEstimator
from .generators.inpaint_generator import InpaintGenerator
from .deepfillv2.network import GatedGenerator
class Generator_deepfillv2(nn.Module):
    
    def __init__(self, mask_process_f, image_size, opt):
        super().__init__()
        self.MG = MaskEstimator(image_size = image_size, backbone = opt.backbone)
        self.mask_process_f = mask_process_f
        self.IG = GatedGenerator(opt)
   
    def forward(self, x):
        batch_size = x.shape[0]
        fake_masks = self.MG(x)
        fake_masks = self.mask_process_f(fake_masks)
        first_out, second_out= self.IG(x, fake_masks)
        return first_out, second_out, fake_masks

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
