import torch
import torch.nn as nn

class Discriminator(nn.Module):
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
                stride=1,
                padding=1,
        ))
        self.encoder = nn.ModuleList(encoder_list)
        
        predict_in_dim = self._calculate_predictor_input_dim()
        print("predict_in_dim",predict_in_dim)
        self.predictor = nn.Sequential(
            nn.Linear(  predict_in_dim, 1, bias=True), 
        )
        
    def _calculate_predictor_input_dim(self):
        batch_size = 1
        x = torch.randn((batch_size,3,self.image_size[0],self.image_size[1]))
        y = x 
        for layer in self.encoder:
            y = layer(y)
        out = y.flatten(start_dim = 1)
        
        self.zero_grad()
        return out.shape[1]
        
    def _conv_block(self,in_dim,out_dim,drop_rate=0):
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
        
        y = x 
        for layer in self.encoder:
            y = layer(y)
        
        out = y.flatten(start_dim = 1)

        assert out.shape[0] == batch_size
        out = self.predictor(out) 
        return out


