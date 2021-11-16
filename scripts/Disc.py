import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self,img_ch,feat=64):
        super(Discriminator,self).__init__()
        #256x256x9
        self.disc=nn.Sequential(
            nn.Conv2d(img_ch,feat,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            #128x128x64
            self.disc_block(feat,feat*2,4,2,1),#64x64x128
            self.disc_block(feat*2,feat*4,4,2,1),#32x32x256
            self.disc_block(feat*4,feat*8,4,1,1),#31x30x512
            self.disc_block(feat*8,feat*8,4,1,1),#30x30x512
            self.disc_block(feat*8,1,3,1,1),#30x30x1
            )
        
    def disc_block(self,in_channels,out_channels,kernel_size,stride,padding=1,padding_mode='reflect'):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))
    def forward(self,x):
        x=self.disc(x)
        # print(x.shape)
        return x
