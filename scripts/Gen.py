import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self,img_ch,feat):
        super(Generator,self).__init__()
#         self.gen=nn.Sequential(
        #256x256x6
        self.down1=self.gen_block_down(img_ch,feat,4,2)
        #128x128x64
        self.down2=self.gen_block_down(feat,feat,4,2)
        #64x64x64
        self.down3=self.gen_block_down(feat,feat*2,4,2)
        #32x32x128
        self.down4=self.gen_block_down(feat*2,feat*4,4,2)
        #16x16x256
        self.down5=self.gen_block_down(feat*4,feat*4,4,2)
        #8x8x256
        self.down6=self.gen_block_down(feat*4,feat*8,4,2)
        #4x4x512
        self.down7=self.gen_block_down(feat*8,feat*8,4,2)
        #2x2x512
        self.btnk=self.gen_block_down(feat*8,feat*16,4,2)
        self.up1=self.gen_block_up(feat*16,feat*8,4,2,padding=1)
        #2x2x512
        self.up2=self.gen_block_up(feat*16,feat*8,4,2)
        #4x4x512
        self.up3=self.gen_block_up(feat*16,feat*4,4,2)
        #8x8x256
        self.up4=self.gen_block_up(feat*8,feat*4,4,2)
        #16x16x256
        self.up5=self.gen_block_up(feat*8,feat*2,4,2)
        #32x32x128
        self.up6=self.gen_block_up(feat*4,feat,4,2)
        #64x64x64
        self.up7=self.gen_block_up(feat*2,feat,4,2)
        #128x128x64
        self.up8=nn.Sequential(
            nn.ConvTranspose2d(feat*2,3,4,2,padding=1,bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh())
        #256x256x3
    def gen_block_down(self,in_channels,out_channels,kernel_size,stride,padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))
    def gen_block_up(self,in_channels,out_channels,kernel_size,stride,padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))
    def forward(self,x):
            d1=self.down1(x)
#             print(f'D1: {d1.shape}')
            d2=self.down2(d1)
#             print(f'D2: {d2.shape}')
            d3=self.down3(d2)
#             print(f'D3: {d3.shape}')
            d4=self.down4(d3)
#             print(f'D4: {d4.shape}')
            d5=self.down5(d4)
#             print(f'D5: {d5.shape}')
            d6=self.down6(d5)
#             print(f'D6: {d6.shape}')
            d7=self.down7(d6)
#             print(f'D7: {d7.shape}')
            btnk=self.btnk(d7)
#             print(f'BTNK: {btnk.shape}')
            up1=self.up1(btnk)
#             print(f'UP1: {up1.shape}')
            up2=self.up2(torch.cat([up1,d7],1))
#             print(f'UP2: {up2.shape}')
            up3=self.up3(torch.cat([up2,d6],1))
#             print(f'UP3: {up3.shape}')
            up4=self.up4(torch.cat([up3,d5],1))
#             print(f'UP4: {up4.shape}')
            up5=self.up5(torch.cat([up4,d4],1))
#             print(f'UP5: {up5.shape}')
            up6=self.up6(torch.cat([up5,d3],1))
#             print(f'UP6: {up6.shape}')
            up7=self.up7(torch.cat([up6,d2],1))
#             print(f'UP7: {up7.shape}')
            up8=self.up8(torch.cat([up7,d1],1))
#             print(f'UP8: {up8.shape}')
            return up8
