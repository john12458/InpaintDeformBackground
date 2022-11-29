import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
from timm.models.layers import LayerNorm2d
import torchvision.transforms as transforms
from .dct import get_pretrained_model

class MaskEstimator(nn.Module):
    
    def __init__(self,image_size,backbone, use_attention=False, use_hieratical = False, drop_rate=0.5, no_sigmoid=False):
        super().__init__()
        print("-- Model --")
        print("use dct")
        self.dct_model = get_pretrained_model()

        self.dct_decoder = nn.Sequential(
            self._deconv_block(in_dim=2,out_dim=2), 
            self._deconv_block(in_dim=2,out_dim=2), 
            nn.ConvTranspose2d(in_channels=2,out_channels=1,kernel_size=4,stride=2,padding=1)
        )

        self.rgb_dct_fuse_layer = nn.Conv2d(4,3,kernel_size=(1,1))

        self.image_size = image_size

        self.no_sigmoid = no_sigmoid
       
        self.backbone_name = backbone
        print("backbone", backbone)

        self.encoder = self._encoder_selector()

        if self.backbone_name == "hrnet_w48":
            md_channels = [2048,1024,512,256,128,1]
            md_input_channels = [2048,1024,512,256]
            md_output_channels = [1024,512,256,128]
            mask_decoder_list = [self._deconv_block(in_d, out_d, drop_rate=drop_rate) for in_d,out_d in zip(md_input_channels,md_output_channels)]
            mask_decoder_list.append(
                nn.ConvTranspose2d(
                    in_channels=md_channels[-2],
                    out_channels=md_channels[-1],
                    kernel_size=4, 
                    stride=2,
                    padding=1,
                ),
            )
            if self.no_sigmoid:
                pass
            else:
                mask_decoder_list.append(nn.Sigmoid())
            self.mask_decoder = nn.ModuleList(mask_decoder_list)

        else:
            self.idx_skip_connect_list = []
            md_channels = [1024,512,256,128,64,1]
            md_input_channels = [1024, 512, 256, 128]
            for i in self.idx_skip_connect_list:
                md_input_channels[i+1] *= 2 

            print("md_input_channels:",md_input_channels)
            # md_input_channels = [1024, 512, 256, 128]
            md_output_channels = [512, 256, 128, 64]
            
            mask_decoder_list = [self._deconv_block(in_d, out_d, drop_rate=drop_rate) for in_d,out_d in zip(md_input_channels,md_output_channels)]
            mask_decoder_list.append(
                nn.ConvTranspose2d(
                    in_channels=md_channels[-2],
                    out_channels=md_channels[-1],
                    kernel_size=4, 
                    stride=2,
                    padding=1,
                ),
            )

            if self.no_sigmoid:
                pass
            else:
                mask_decoder_list.append(nn.Sigmoid())
            self.mask_decoder = nn.ModuleList(mask_decoder_list)
        print("-- End --")

    
    def _encoder_selector(self):
        return timm.create_model(self.backbone_name, pretrained=True) 
        
    def _deconv_block(self,in_dim,out_dim,drop_rate=0):
         return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=4, 
                    stride=2,
                    padding=1,
                ),
                LayerNorm2d(num_channels=out_dim),
                # nn.Dropout(drop_rate),
                # nn.BatchNorm2d(out_dim),
                nn.ReLU()
        )

    def _encode(self,x):
        latent_code = self.encoder.forward_features(x)
        if self.backbone_name == "swinv2_base_window12to16_192to256_22kft1k":
            # [10, 64, 1024])
            latent_code = latent_code.permute(0,2,1)
            latent_code = latent_code.reshape((latent_code.shape[0],latent_code.shape[1],8,8))
        return latent_code
    
    def _encoder_FPN(self,x):
        feature_list = []        
        feat = self.encoder.stem(x)
        for m in self.encoder.stages:
            feat = m(feat)
            feature_list.append(feat)
        return feat, feature_list
    
    
    def forward(self, x, dct, qtables):

        batch_size = x.shape[0]
        
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        # assert dct.shape == (batch_size,1,self.image_size[0],self.image_size[1])

        # rgb_dct = torch.cat((x,dct),dim=1)
        # print("dct)vol",to_dct_volume(dct).shape)
        dct_feature = self.dct_model(dct,qtables) # B,2,32,32
        dct_mask = self.dct_decoder(dct_feature)
        

        # x = self.rgb_dct_fuse_layer(rgb_dct)

        # rgb branch
        if self.backbone_name == "hrnet_w48":
            # encoder
            latent_code = self._encode(x) 
            y =  latent_code
            # decoder
            for idx,layer in enumerate(self.mask_decoder):
                y=layer(y)
            rgb_mask = y 
        else:
            # encoder
            latent_code, feature_list = self._encoder_FPN(x)
            reverse_feature_list = feature_list[::-1][1:]
            y = latent_code
            #decoder
            for idx,layer in enumerate(self.mask_decoder):
                y=layer(y)
                if idx < len(reverse_feature_list) and idx in self.idx_skip_connect_list:
                    y = torch.cat((y,reverse_feature_list[idx]),dim=1)
            
            rgb_mask = y 

        mask = dct_mask + rgb_mask

        
        
        assert mask.shape == (batch_size,1,self.image_size[0],self.image_size[1]), print(mask.shape)
        return mask

if __name__ == '__main__':
    import numpy as np

    net = MaskEstimator(image_size=(256,256), backbone= "convnext_base_in22k")
    x = torch.rand((1,3,256,256))
    x1 = net(x)
  
  