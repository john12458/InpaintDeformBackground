import torch
import torch.nn as nn

class InpaintGenerator(nn.Module):
    
        
    def __init__(self,image_size):
        super().__init__()
        self.image_size = image_size
        channels = [3,64,128,256,512]
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
        
        
        
        d_channels = channels[::-1]  
        print("d_channels",d_channels)
        print([ (in_d*2,out_d) for in_d,out_d in zip(d_channels[:-2],d_channels[1:-1])])
        decoder_list = [self._deconv_block(in_d*2, out_d, drop_rate=0.5) for in_d,out_d in zip(d_channels[:-2],d_channels[1:-1])]
        decoder_list.append(
            nn.ConvTranspose2d(
                in_channels=d_channels[-2]*2,
                out_channels=d_channels[-1],
                kernel_size=4, 
                stride=2,
                padding=1,
        ))
        self.decoder = nn.ModuleList(decoder_list)
        self.tanh = nn.Tanh()
         

        
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
        assert x.shape == (batch_size,3,self.image_size[0],self.image_size[1])
        hiddens = []
        # encoder
        y = x 
        for layer in self.encoder:
            y = layer(y)
            hiddens.append(y)
        latent_code = y.clone()
        
        # inpaint decoder
        # y = latent_code.clone()
        reversed_hiddens = hiddens[::-1]
        for idx,layer in enumerate(self.decoder):
            if idx < len(reversed_hiddens):
                y = torch.cat((y,reversed_hiddens[idx]),dim=1)
            y=layer(y)
        reconstruct =self.tanh(y)
        assert reconstruct.shape == (batch_size,3,self.image_size[0],self.image_size[1])
            
        return reconstruct

