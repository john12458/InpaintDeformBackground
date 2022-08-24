import torch
import torch.nn as nn
import timm


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
