import torch
import torch.nn as nn
import timm
from .vqvae import VQVAE
from torch.nn import functional as F
from timm.models.vision_transformer import PatchEmbed, Block
class AttentionLayer(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=512, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        num_patches = 1024
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)


        
        decoder_blocks_list = [
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)]
        self.decoder_blocks = nn.ModuleList(decoder_blocks_list)
        self.decoder_norm = norm_layer(decoder_embed_dim)

    def forward(self, x):
        # print("input x",x.shape)
        origin_shape = x.shape
        x = x.reshape(x.shape[0],x.shape[1],-1).permute(0,2,1)

        x = self.decoder_embed(x)
        # print("input x",x.shape)
        
        # append cls token
        cls_token = self.cls_token + self.decoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
            print("x",x.shape)
        x = self.decoder_norm(x)
        x = x[:, 1:, :]
        # x = self.decoder_norm(x)
        x = x.permute(0,2,1)
        x = x.reshape(x.shape[0],x.shape[1],origin_shape[2],origin_shape[3])
        # print("last",x.shape)

        return x

class MaskEstimator(nn.Module):
   
    
    def __init__(self,image_size,backbone, use_attention=False):
        super().__init__()
        self.image_size = image_size
       
        self.backbone_name = backbone
        self.encoder = self._encoder_selector()


        if use_attention:
            self.attention = AttentionLayer()
        
        
        md_channels = [1024,512,256,128,64,1]
        if self.backbone_name == "vqvae":
            md_channels = [512,256,64,1]
            if use_attention:
                md_channels = [256,128,64,1]

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
    
    def _encoder_selector(self):
        if self.backbone_name == "vqvae":
            model = VQVAE()
            vqvae_ckpt_path = "/workspace/inpaint_mask/src/models/generators/vqvae_560.pt"
            model.load_state_dict(torch.load(vqvae_ckpt_path))
            print(f"load vqvae from : {vqvae_ckpt_path}")
            return model
        else:
            print(backbone)
        return timm.create_model(backbone, pretrained=True) 
        
        
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
    def _encode(self,x):
        if self.backbone_name == "vqvae":
            _, _, _, id_t, id_b = self.encoder.encode(x)
            return F.one_hot(id_t.detach(), 512).permute(0, 3, 1, 2).type_as(x)
        else:
            return self.encoder.forward_features(x)
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        hiddens = []
        # encoder
        latent_code = self._encode(x)
        # print("latent_code",latent_code.shape)
        if self.attention:
            latent_code = self.attention(latent_code)
        # latent_code = self.attention(latent_code)
        # print("atten_out",atten_out.shape)
        
        # decoder
        y =  latent_code
        for idx,layer in enumerate(self.mask_decoder):
            y=layer(y)
            # print(y.shape)
        
        mask = y 
        # print("mask",mask.shape)
        
        assert mask.shape == (batch_size,1,self.image_size[0],self.image_size[1]), print(mask.shape)
        return mask
        # return reconstruct, mask 
