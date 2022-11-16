import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
from timm.models.layers import LayerNorm2d
# from timm.models.vision_transformer import PatchEmbed, Block
import torchvision.transforms as transforms
# class DeConvNeXtBlock(nn.Module):
#     """ ConvNeXt Block
#     There are two equivalent implementations:
#       (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#       (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
#     choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
#     is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
#     Args:
#         in_chs (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """

#     def __init__(
#             self,
#             in_chs,
#             out_chs=None,
#             kernel_size=7,
#             stride=1,
#             dilation=1,
#             mlp_ratio=4,
#             conv_mlp=False,
#             conv_bias=True,
#             ls_init_value=1e-6,
#             act_layer='gelu',
#             norm_layer=None,
#             drop_path=0.,
#     ):
#         super().__init__()
#         out_chs = out_chs or in_chs
#         act_layer = get_act_layer(act_layer)
#         if not norm_layer:
#             norm_layer = LayerNorm2d if conv_mlp else LayerNorm
#         mlp_layer = ConvMlp if conv_mlp else Mlp
#         self.use_conv_mlp = conv_mlp

#         self.conv_dw = create_conv2d(
#             in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, depthwise=True, bias=conv_bias)
#         self.norm = norm_layer(out_chs)
#         self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
#         self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         shortcut = x
#         x = self.conv_dw(x)
#         if self.use_conv_mlp:
#             x = self.norm(x)
#             x = self.mlp(x)
#         else:
#             x = x.permute(0, 2, 3, 1)
#             x = self.norm(x)
#             x = self.mlp(x)
#             x = x.permute(0, 3, 1, 2)
#         if self.gamma is not None:
#             x = x.mul(self.gamma.reshape(1, -1, 1, 1))

#         x = self.drop_path(x) + shortcut
#         return x
class MaskEstimator(nn.Module):
    
    def __init__(self,image_size,backbone, use_attention=False, use_hieratical = False, drop_rate=0.5, no_sigmoid=False):
        super().__init__()
        print("-- Model --")
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
        # encoder
        latent_code, feature_list = self._encoder_FPN(x)
        reverse_feature_list = feature_list[::-1][1:]
        y = latent_code
        #decoder
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
