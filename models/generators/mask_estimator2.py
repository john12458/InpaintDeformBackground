import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
# from timm.models.vision_transformer import PatchEmbed, Block
import torchvision.transforms as transforms

class MaskEstimator(nn.Module):
    
    def __init__(self,image_size,backbone, use_attention=False, use_hieratical = False, drop_rate=0.5, no_sigmoid=False):
        super().__init__()
        self.image_size = image_size

        self.no_sigmoid = no_sigmoid
       
        self.backbone_name = backbone
        self.encoder = self._encoder_selector()


        # classfication branch
        md_channels = [2048,1024,512,256,128,1]
        md_input_channels = [2048,1024,512,256]
        md_output_channels = [1024,512,256,128]

        # md_channels = [1024,512,256,128,64,1]
        # md_input_channels = [1024, 512+3, 256, 128+3]
        # md_output_channels = [512, 256, 128, 64]
        
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

        # detail branch
        # print("detail md_channels",md_channels)
        # print([ (in_d*2,out_d) for in_d,out_d in zip(md_channels[:-2],md_channels[1:-1])])
        # mask_decoder_list = [self._deconv_block(in_d, out_d, drop_rate=drop_rate) for in_d,out_d in zip(md_channels[:-2],md_channels[1:-1])]
        # mask_decoder_list.append(
        #     nn.ConvTranspose2d(
        #         in_channels=md_channels[-2],
        #         out_channels=md_channels[-1],
        #         kernel_size=4, 
        #         stride=2,
        #         padding=1,
        #     ),
        # )
        # if self.no_sigmoid:
        #     pass
        # else:
        #     mask_decoder_list.append(nn.Sigmoid())
        # self.mask_decoder = nn.ModuleList(mask_decoder_list)
    
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
                nn.Dropout(drop_rate),
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
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        hiddens = []
        # encoder
        latent_code = self._encode(x) 
        

        # decoder
        y =  latent_code
        # latent_code torch.Size([16, 2048, 8, 8]) hrnet
# latent_code torch.Size([16, 1024, 8, 8])
# y_h torch.Size([16, 3, 64, 64])
# decoder torch.Size([16, 512, 16, 16])
# decoder torch.Size([16, 256, 32, 32])
# decoder torch.Size([16, 128, 64, 64])
# decoder torch.Size([16, 64, 128, 128])
# decoder torch.Size([16, 1, 256, 256])

        
        # y_h16 = F.interpolate(x, scale_factor=0.0625, mode="bilinear") #16
        # y_h4 = F.interpolate(x, scale_factor=0.25, mode="bilinear") #64
        # print("y_h",y_h.shape)

        
        for idx,layer in enumerate(self.mask_decoder):
            y=layer(y)
            # if idx == 0:
            #     y = torch.cat((y,y_h16),dim=1)
            # if idx == 2:
            #     y = torch.cat((y,y_h4),dim=1)
                

        
        mask = y 
        # print("mask",mask.shape)
        
        assert mask.shape == (batch_size,1,self.image_size[0],self.image_size[1]), print(mask.shape)
        return mask
        # return reconstruct, mask 
