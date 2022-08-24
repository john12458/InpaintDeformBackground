import torch
import torch.nn as nn
from .generators.mask_estimator import MaskEstimator
from .generators.inpaint_generator import InpaintGenerator


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
