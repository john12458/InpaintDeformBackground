import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
from timm.models.layers import LayerNorm2d
import torchvision.transforms as transforms
def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray

class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class MaskEstimator(nn.Module):
    
    def __init__(self,image_size,backbone, use_attention=False, use_hieratical = False, drop_rate=0.5, no_sigmoid=False, use_bayar=False):
        super().__init__()
        print("-- Model --")
        self.use_bayar = use_bayar
        if self.use_bayar:
            self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)

        self.image_size = image_size
        self.no_sigmoid = no_sigmoid
        self.backbone_name = backbone
        print("backbone", backbone)

        self.encoder = self._encoder_selector()

        # classfication branch
        self.idx_skip_connect_list = []
        md_channels = [1024,512,256,128,64,1]
        md_input_channels = [1024, 512, 256, 128]
        for i in self.idx_skip_connect_list:
            md_input_channels[i+1] *= 2 

        print("md_input_channels:",md_input_channels)
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
                # nn.Dropout(drop_rate),
                nn.BatchNorm2d(out_dim),
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
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        if self.use_bayar:
            x = rgb2gray(x)
            x = self.constrain_conv(x)
        # encoder
        latent_code, feature_list = self._encoder_FPN(x)
        reverse_feature_list = feature_list[::-1][1:]
        y = latent_code
        
        # decoder
        for idx,layer in enumerate(self.mask_decoder):
            y=layer(y)
            if idx < len(reverse_feature_list) and idx in self.idx_skip_connect_list:
                y = torch.cat((y,reverse_feature_list[idx]),dim=1)
        
        mask = y 
        
        assert mask.shape == (batch_size,1,self.image_size[0],self.image_size[1]), print(mask.shape)
        return mask
        # return reconstruct, mask 

if __name__ == '__main__':
    import numpy as np

    net = MaskEstimator(image_size=(256,256), backbone= "convnext_base_in22k")
    x = torch.rand((1,3,256,256))
    x1 = net(x)
  
    # print(net.encoder)
    
    # #### HELPER FUNCTION FOR FEATURE EXTRACTION
    # def get_features(name):
    #     def hook(model, input, output):
    #         features[name] = output.detach()
    #     return hook
    # net.encoder.stem.register_forward_hook(get_features(f'stem'))
    # ##### REGISTER HOOK
    # for i in range(len(net.encoder.stages)):
    #     net.encoder.stages[i].register_forward_hook(get_features(f'feat_{i}'))
    # ##### FEATURE EXTRACTION LOOP

    # # placeholders
    # PREDS = []
    # FEATS = []

    # # placeholder for batch features
    # features = {}
    # out = net(x)
    # FEATS.append(np.concatenate([features[f'stem'].cpu().numpy()]))
    # for i in range(len(net.encoder.stages)):
    #     feat_i = []
    #     feat_i.append(features[f'feat_{i}'].cpu().numpy())
    #     feat_i = np.concatenate(feat_i)
    #     FEATS.append(feat_i)
    # for f in FEATS:
    #     print(f.shape) 
