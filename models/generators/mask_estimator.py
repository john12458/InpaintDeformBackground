import torch
import torch.nn as nn
import timm
from .vqvae import VQVAE
from torch.nn import functional as F
from timm.models.vision_transformer import PatchEmbed, Block
import torchvision.transforms as transforms
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
            # print("x",x.shape)
        x = self.decoder_norm(x)
        x = x[:, 1:, :]
        # x = self.decoder_norm(x)
        x = x.permute(0,2,1)
        x = x.reshape(x.shape[0],x.shape[1],origin_shape[2],origin_shape[3])
        # print("last",x.shape)

        return x

class Global_estimator(nn.Module):
    def __init__(self,image_size):
        super().__init__()
        self.image_size = image_size
        channels = [3,64,128,256,512,1024]
        print("channels",channels)
        print([ (in_d,out_d) for in_d,out_d in zip(channels[:-2],channels[1:-1])])
        encoder_list = [ self._conv_block(in_d,out_d) for in_d,out_d in zip(channels[:-2],channels[1:-1])]
        encoder_list.append( 
            nn.Conv2d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=4,
                stride=2,
                padding=1,
        ))
        self.encoder = nn.ModuleList(encoder_list)
        
        # d_channels = channels[::-1]  
        # print("d_channels",d_channels)
        # print([ (in_d*2,out_d) for in_d,out_d in zip(d_channels[:-2],d_channels[1:-1])])
        # decoder_list = [self._deconv_block(in_d*2, out_d, drop_rate=0.5) for in_d,out_d in zip(d_channels[:-2],d_channels[1:-1])]
        # decoder_list.append(
        #     nn.ConvTranspose2d(
        #         in_channels=d_channels[-2]*2,
        #         out_channels=d_channels[-1],
        #         kernel_size=4, 
        #         stride=2,
        #         padding=1,
        # ))
        # self.decoder = nn.ModuleList(decoder_list)
        # self.tanh = nn.Tanh()
         
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
    
    def _conv_block(self,in_dim,out_dim,drop_rate=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ), 
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2)
        )  
    
    def forward(self, x):
        batch_size = x.shape[0]
        hiddens = []
        # encoder
        y = x 
        for layer in self.encoder:
            y = layer(y) 
            hiddens.append(y)
        latent_code = y.clone()
        # print("g_latent_code",latent_code.shape)
        
        # reversed_hiddens = hiddens[::-1]
        # for idx,layer in enumerate(self.decoder):
        #     if idx < len(reversed_hiddens):
        #         y = torch.cat((y,reversed_hiddens[idx]),dim=1)
        #     y=layer(y)
        #     print("y",y.shape)
        # reconstruct =self.tanh(y)
            
        return latent_code





class MaskEstimator(nn.Module):
    
    def __init__(self,image_size,backbone, use_attention=False, use_hieratical = False, drop_rate=0.5):
        super().__init__()
        self.image_size = image_size
       
        self.backbone_name = backbone
        self.encoder = self._encoder_selector()


        self.attention = use_attention
        if use_attention:
            # self.attention = AttentionLayer(embed_dim=64,decoder_embed_dim=64,decoder_depth=2)
            self.attention = AttentionLayer(embed_dim=512,decoder_embed_dim=512,decoder_depth=4)
            self.attention.decoder_embed = nn.Identity()
        
        self.use_hieratical = use_hieratical
        if self.use_hieratical:
            scale_factor = 2
            sub_image_size = [ img_size//scale_factor for img_size in image_size]
            self.resize_0_5_f = transforms.Resize(size = sub_image_size )
            self.global_estimator = Global_estimator(sub_image_size )
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

            
            # self.attention = AttentionLayer(embed_dim=64+64,decoder_embed_dim=64,decoder_depth=2)
            
            # self.attention = AttentionLayer(embed_dim=512,decoder_embed_dim=512,decoder_depth=4)
            # self.attention.decoder_embed = nn.Identity()
        
        
        md_channels = [1024,512,256,128,64,1]
        if self.backbone_name == "vqvae":
            md_channels = [512,256,64,1]
            if use_attention:
                md_channels = [64,32,16,1]

        print("md_channels",md_channels)
        print([ (in_d*2,out_d) for in_d,out_d in zip(md_channels[:-2],md_channels[1:-1])])
        mask_decoder_list = [self._deconv_block(in_d, out_d, drop_rate=drop_rate) for in_d,out_d in zip(md_channels[:-2],md_channels[1:-1])]
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
            print(self.backbone_name)
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
                nn.ReLU()
         )
    def _encode(self,x):

        latent_code = None
        if self.backbone_name == "vqvae":
            quant_t, _, difft, id_t, id_b = self.encoder.encode(x)
            # print(quant_t.shape, difft.shape) # torch.Size([16, 64, 32, 32]) torch.Size([16, 32, 32, 64])
            # difft = difft.permute(0, 3, 1, 2).type_as(x)
            # cat_quant_t_diff = torch.cat((quant_t,difft),dim=1)
            # return cat_quant_t_diff
            # print("id_t",id_t.shape)

            print("id_t",id_t.shape) # B, 32, 32
            latent_codebook = F.one_hot(id_t, 512).permute(0, 3, 1, 2).type_as(x)  # B, 512, 32, 32
            print("latent_codebook",latent_codebook.shape)

            latent_code = quant_t
            if self.attention:
                # latent_code = self.attention(latent_code.detach())


                # print("latent_codebook",latent_codebook.shape)
                latent_codebook = self.attention(latent_codebook.detach())
                # print("latent_codebook_afer attention",latent_codebook.shape)
                latent_codebook = torch.argmax(latent_codebook, dim=1)
                # print("reverse one hot latent_codebook",latent_codebook.shape)
                latent_code = self.encoder.quantize_t.embed_code(latent_codebook) # idx_t transformer quant_t
                # print("latent_code",latent_code.shape)
                latent_code = latent_code.permute(0, 3, 1, 2)
                # print("latent_code",latent_code.shape)
        else:
            latent_code = self.encoder.forward_features(x)
            if self.backbone_name == "swinv2_base_window12to16_192to256_22kft1k":
                # [10, 64, 1024])
                latent_code = latent_code.permute(0,2,1)
                latent_code = latent_code.reshape((latent_code.shape[0],latent_code.shape[1],8,8))
            # [10, 64, 1024])

        if self.use_hieratical:
            # latent_code torch.Size([16, 1024, 8, 8]) 256
            g_latent_code = self.global_estimator(self.resize_0_5_f(x)) # g_latent_code torch.Size([16, 1024, 4, 4]
            g_latent_code = self.upsample(g_latent_code) 
            latent_code = latent_code + g_latent_code
        
        return latent_code
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        hiddens = []
        # encoder
        latent_code = self._encode(x) 

        # decoder
        y =  latent_code
        for idx,layer in enumerate(self.mask_decoder):
            y=layer(y)
            # print("decoder",y.shape)
        
        mask = y 
        # print("mask",mask.shape)
        
        assert mask.shape == (batch_size,1,self.image_size[0],self.image_size[1]), print(mask.shape)
        return mask
        # return reconstruct, mask 
