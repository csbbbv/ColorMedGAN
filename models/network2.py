import torch
import torch.nn as nn
from spectral import SpectralNorm
from torch.nn import functional as F
from basic_modules import *
from collections import OrderedDict
class Baseline(nn.Module):
    def __init__(self, img_ch=1, num_classes=5, depth=2, use_deconv=False):
        super(Baseline, self).__init__()
        chs = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool2d(2, 2)
        # p1 encoder
        self.p1_enc1 = EncoderBlock(img_ch, chs[0], depth=depth)
        self.p1_enc2 = EncoderBlock(chs[0], chs[1], depth=depth)
        self.p1_enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.p1_enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.p1_cen = EncoderBlock(chs[3], chs[4], depth=depth)

        self.p1_dec4 = DecoderBlock(chs[4], chs[3], use_deconv=use_deconv)
        self.p1_decconv4 = EncoderBlock(chs[3] * 2, chs[3])

        self.p1_dec3 = DecoderBlock(chs[3], chs[2], use_deconv=use_deconv)
        self.p1_decconv3 = EncoderBlock(chs[2] * 2, chs[2])

        self.p1_dec2 = DecoderBlock(chs[2], chs[1], use_deconv=use_deconv)
        self.p1_decconv2 = EncoderBlock(chs[1] * 2, chs[1])

        self.p1_dec1 = DecoderBlock(chs[1], chs[0], use_deconv=use_deconv)
        self.p1_decconv1 = EncoderBlock(chs[0] * 2, chs[0])

        self.p1_conv_1x1 = nn.Conv2d(chs[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # p1 encoder
        p1_x1 = self.p1_enc1(x)#64*256*256
        p1_x2 = self.pool(p1_x1)#128*128*128
        p1_x2 = self.p1_enc2(p1_x2)
        p1_x3 = self.pool(p1_x2)#256*64*64
        p1_x3 = self.p1_enc3(p1_x3)
        p1_x4 = self.pool(p1_x3)#512*32*32
        p1_x4 = self.p1_enc4(p1_x4)
        p1_center = self.pool(p1_x4)#1024*16*16 
        p1_center = self.p1_cen(p1_center)

        """
          first path decoder
        """
        d4 = self.p1_dec4(p1_center)#512*32*32
        d4 = torch.cat((p1_x4, d4), dim=1)
        d4 = self.p1_decconv4(d4)

        d3 = self.p1_dec3(d4)#256*64*64
        d3 = torch.cat((p1_x3, d3), dim=1)
        d3 = self.p1_decconv3(d3)

        d2 = self.p1_dec2(d3)#128*128*128
        d2_cat = torch.cat((p1_x2, d2), dim=1) # 256
        d2 = self.p1_decconv2(d2_cat)# 128

        d1 = self.p1_dec1(d2)#64*256*256
        d1 = torch.cat((p1_x1, d1), dim=1)
        d1 = self.p1_decconv1(d1)

        p1_out = self.p1_conv_1x1(d1)#5*256*256

        return    p1_x2 , d2
        # return p1_out


class generator(torch.nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf

        self.downconv1 = nn.Sequential(#input H,W 3 output H,W 64  1
            SpectralNorm(nn.Conv2d(in_nc, nf, 5, 1, 2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )

        self.downconv2 = nn.Sequential(#input H,W 64 output H/2,W/2 128  2
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.downconv3 = nn.Sequential(#input H/2,W/2 128 output H/4,W/4 256  3
            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.downconv4 = nn.Sequential(# input H/4,W/4 256 output H/8,W/8 512  4
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )

        self.downconv5 = nn.Sequential(# input H/8,W/8 512 output H/8,W/8 512  5
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )


        self.upconv3 = nn.Sequential(# input H/8,W/8 1024 output H/4,W/4 256  6
            SpectralNorm(nn.ConvTranspose2d(nf * 16, nf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.upconv2 = nn.Sequential(# input H/4,W/4 512 output H/2,W/2 128  7
            SpectralNorm(nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.upconv1 = nn.Sequential(#input H/2,W/2 256 output H,W 3  8
            SpectralNorm(nn.ConvTranspose2d(nf * 4, nf, 4, 2, 1)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 5, 1, 2),
            nn.Tanh(),
        )

        # forward method
    def encoder(self,input):
        x1 = self.downconv1(input) #64 H,W
        x2 = self.downconv2(x1) #128 H/2,W/2
        x3 = self.downconv3(x2) #256 H/4,W/4
        x4 = self.downconv4(x3) #512 H/8,W/8
        x5 = self.downconv5(x4) #512 H/8,W/8
        return x2,x3,x4,x5
    
    def decoder(self,x2,x3,x4,x5):
        y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        output = y1
        return output
    def forward(self, input):

        x2,x3,x4,x5 = self.encoder(input)

        # y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        # y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        # y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        output = self.decoder(x2,x3,x4,x5)

        return output


class generator_upsample(torch.nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(generator_upsample, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf

        self.downconv1 = nn.Sequential(#input H,W 3 output H,W 64  1
            SpectralNorm(nn.Conv2d(in_nc, nf, 5, 1, 2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )

        self.downconv2 = nn.Sequential(#input H,W 64 output H/2,W/2 128  2
            SpectralNorm(nn.Conv2d(nf, nf * 2, 5, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.downconv3 = nn.Sequential(#input H/2,W/2 128 output H/4,W/4 256  3
            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 5, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.downconv4 = nn.Sequential(# input H/4,W/4 256 output H/8,W/8 512  4
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 5, 2, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )

        self.downconv5 = nn.Sequential(# input H/8,W/8 512 output H/8,W/8 512  5
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )


        self.upconv3 = nn.Sequential(# input H/8,W/8 1024 output H/4,W/4 256  6
            # SpectralNorm(nn.ConvTranspose2d(nf * 16, nf * 4, 4, 2, 1)),
            nn.Sample(256//4),
            SpectralNorm(nn.Conv2d(nf*16,nf*4,4,5,1,2)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.upconv2 = nn.Sequential(# input H/4,W/4 512 output H/2,W/2 128  7
            # SpectralNorm(nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)),
            nn.Sample(256//2),
            SpectralNorm(nn.Conv2d(nf*8,nf*2,5,1,2)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.upconv1 = nn.Sequential(#input H/2,W/2 256 output H,W 3  8
            # SpectralNorm(nn.ConvTranspose2d(nf * 4, nf, 4, 2, 1)),
            nn.Sample(256),
            SpectralNorm(nn.Conv2d(nf*4,nf*1,5,1,2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 5, 1, 2),
            nn.Tanh(),
        )

        # forward method
    def encoder(self,input):
        x1 = self.downconv1(input) #64 H,W
        x2 = self.downconv2(x1) #128 H/2,W/2
        x3 = self.downconv3(x2) #256 H/4,W/4
        x4 = self.downconv4(x3) #512 H/8,W/8
        x5 = self.downconv5(x4) #512 H/8,W/8
        return x2,x3,x4,x5
    
    def decoder(self,x2,x3,x4,x5):
        y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        output = y1
        return output
    def forward(self, input):

        x2,x3,x4,x5 = self.encoder(input)

        # y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        # y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        # y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        output = self.decoder(x2,x3,x4,x5)

        return output


class generator_seg_branch(nn.Module):
    def __init__(self,in_nc,out_nc,nf):
        super(generator_seg_branch,self).__init__()
        
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf

        self.downconv1 = nn.Sequential(#input H,W 3 output H,W 64  1
            SpectralNorm(nn.Conv2d(in_nc, nf, 5, 1, 2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )

        self.downconv2 = nn.Sequential(#input H,W 64 output H/2,W/2 128  2
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.downconv3 = nn.Sequential(#input H/2,W/2 128 output H/4,W/4 256  3
            SpectralNorm(nn.Conv2d(nf * 4, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.downconv4 = nn.Sequential(# input H/4,W/4 256 output H/8,W/8 512  4
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )

        self.downconv5 = nn.Sequential(# input H/8,W/8 512 output H/8,W/8 512  5
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )


        self.upconv3 = nn.Sequential(# input H/8,W/8 1024 output H/4,W/4 256  6
            SpectralNorm(nn.ConvTranspose2d(nf * 16, nf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.upconv2 = nn.Sequential(# input H/4,W/4 512 output H/2,W/2 128  7
            SpectralNorm(nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.upconv1 = nn.Sequential(#input H/2,W/2 256 output H,W 3  8
            SpectralNorm(nn.ConvTranspose2d(nf * 6, nf, 4, 2, 1)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 5, 1, 2),
            nn.Tanh(),
        )

        # forward method
    # def encoder(self,input):
    #     seg_ft1,seg_ft2 = self.seg_net(input)
    #     x1 = self.downconv1(input) #64 H,W
    #     x2 = self.downconv2(x1) #128 H/2,W/2
    #     # x2 = self.downconv2(torch.cat([x1,seg_ft1],dim=1))
    #     x3 = self.downconv3(torch.cat([x2,seg_ft1],dim=1)) #256 H/4,W/4
    #     x4 = self.downconv4(x3) #512 H/8,W/8
    #     x5 = self.downconv5(x4) #512 H/8,W/8
        
    #     return x2,x3,x4,x5,seg_ft1,seg_ft2
    
    # def decoder(self,x2,x3,x4,x5,seg_ft2):
        
    #     y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
    #     y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
    #     y1 = self.upconv1(torch.cat([y2, x2,seg_ft2], dim=1)) #64 H,W
    #     output = y1
    #     return output
    def forward(self, input,seg_ft1,seg_ft2):

        # x2,x3,x4,x5 = self.encoder(input)

        # y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        # y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        # y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        # output = self.decoder(x2,x3,x4,x5)
        x1 = self.downconv1(input) #64 H,W
        # input_seg = input[:,0,:,:]
        
        x2 = self.downconv2(x1) #128 H/2,W/2
        x3 = self.downconv3(torch.cat([x2, seg_ft1], dim=1)) #256 H/4,W/4
        x4 = self.downconv4(x3) #512 H/8,W/8
        x5 = self.downconv5(x4) #512 H/8,W/8
        y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        y1 = self.upconv1(torch.cat([y2, x2,seg_ft2], dim=1)) #64 H,W
        output = y1
        return output
        



class generator_seg_attention(nn.Module):
    def __init__(self,in_nc,out_nc,nf):
        super(generator_seg_attention,self).__init__()
        
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.relu = nn.ReLU(True)
        self.downconv1 = nn.Sequential(#input H,W 3 output H,W 64  1
            SpectralNorm(nn.Conv2d(in_nc, nf, 5, 1, 2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )

        self.downconv2 = nn.Sequential(#input H,W 64 output H/2,W/2 128  2
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.downconv3 = nn.Sequential(#input H/2,W/2 128 output H/4,W/4 256  3
            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.downconv4 = nn.Sequential(# input H/4,W/4 256 output H/8,W/8 512  4
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )

        self.downconv5 = nn.Sequential(# input H/8,W/8 512 output H/8,W/8 512  5
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )


        self.upconv3 = nn.Sequential(# input H/8,W/8 1024 output H/4,W/4 256  6
            SpectralNorm(nn.ConvTranspose2d(nf * 16, nf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.upconv2 = nn.Sequential(# input H/4,W/4 512 output H/2,W/2 128  7
            SpectralNorm(nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.upconv1 = nn.Sequential(#input H/2,W/2 256 output H,W 3  8
            SpectralNorm(nn.ConvTranspose2d(nf * 4, nf, 4, 2, 1)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 5, 1, 2),
            nn.Tanh(),
        )

        # forward method
    # def encoder(self,input):
    #     seg_ft1,seg_ft2 = self.seg_net(input)
    #     x1 = self.downconv1(input) #64 H,W
    #     x2 = self.downconv2(x1) #128 H/2,W/2
    #     # x2 = self.downconv2(torch.cat([x1,seg_ft1],dim=1))
    #     x3 = self.downconv3(torch.cat([x2,seg_ft1],dim=1)) #256 H/4,W/4
    #     x4 = self.downconv4(x3) #512 H/8,W/8
    #     x5 = self.downconv5(x4) #512 H/8,W/8
        
    #     return x2,x3,x4,x5,seg_ft1,seg_ft2
    
    # def decoder(self,x2,x3,x4,x5,seg_ft2):
        
    #     y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
    #     y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
    #     y1 = self.upconv1(torch.cat([y2, x2,seg_ft2], dim=1)) #64 H,W
    #     output = y1
    #     return output
    def forward(self, input,seg_ft1,seg_ft2):

        # x2,x3,x4,x5 = self.encoder(input)

        # y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        # y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        # y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        # output = self.decoder(x2,x3,x4,x5)
        x1 = self.downconv1(input) #64 H,W
        # input_seg = input[:,0,:,:]
        
        x2 = self.downconv2(x1) #128 H/2,W/2
        x3 = self.downconv3(x2*torch.sigmoid(seg_ft1)) #256 H/4,W/4
        x4 = self.downconv4(x3) #512 H/8,W/8
        x5 = self.downconv5(x4) #512 H/8,W/8
        y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        y2 = y2 * torch.sigmoid(seg_ft2)
        y1 = self.upconv1(torch.cat([y2, x2], dim=1))#+ torch.cat([y2, x2], dim=1) * torch.sigmoid(seg_ft2)) #64 H,W
        output = y1
        return output




class generator_seg_attention_upsample(nn.Module):
    def __init__(self,in_nc,out_nc,nf):
        super(generator_seg_attention_upsample,self).__init__()
        
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.relu = nn.ReLU(True)
        self.downconv1 = nn.Sequential(#input H,W 3 output H,W 64  1
            SpectralNorm(nn.Conv2d(in_nc, nf, 3, 1, 2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )

        self.downconv2 = nn.Sequential(#input H,W 64 output H/2,W/2 128  2
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.downconv3 = nn.Sequential(#input H/2,W/2 128 output H/4,W/4 256  3
            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.downconv4 = nn.Sequential(# input H/4,W/4 256 output H/8,W/8 512  4
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )

        self.downconv5 = nn.Sequential(# input H/8,W/8 512 output H/8,W/8 512  5
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )


        self.upconv3 = nn.Sequential(# input H/8,W/8 1024 output H/4,W/4 256  6
            nn.Upsample(256//4,mode = 'nearest'),
            SpectralNorm(nn.Conv2d(nf*16,nf*4,5,1,2)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.upconv2 = nn.Sequential(# input H/4,W/4 512 output H/2,W/2 128  7
        
            # SpectralNorm(nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)),
            nn.Upsample(256//2,mode = 'nearest'),
            SpectralNorm(nn.Conv2d(nf*8,nf*2,5,1,2)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.upconv1 = nn.Sequential(#input H/2,W/2 256 output H,W 3  8
            # SpectralNorm(nn.ConvTranspose2d(nf * 4, nf, 4, 2, 1)),
            nn.Upsample(256,mode = 'nearest'),
            SpectralNorm(nn.Conv2d(nf*4,nf,5,1,2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 5, 1, 2),
            nn.Tanh(),
        )

        # forward method
    # def encoder(self,input):
    #     seg_ft1,seg_ft2 = self.seg_net(input)
    #     x1 = self.downconv1(input) #64 H,W
    #     x2 = self.downconv2(x1) #128 H/2,W/2
    #     # x2 = self.downconv2(torch.cat([x1,seg_ft1],dim=1))
    #     x3 = self.downconv3(torch.cat([x2,seg_ft1],dim=1)) #256 H/4,W/4
    #     x4 = self.downconv4(x3) #512 H/8,W/8
    #     x5 = self.downconv5(x4) #512 H/8,W/8
        
    #     return x2,x3,x4,x5,seg_ft1,seg_ft2
    
    # def decoder(self,x2,x3,x4,x5,seg_ft2):
        
    #     y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
    #     y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
    #     y1 = self.upconv1(torch.cat([y2, x2,seg_ft2], dim=1)) #64 H,W
    #     output = y1
    #     return output
    def forward(self, input,seg_ft1,seg_ft2):

        # x2,x3,x4,x5 = self.encoder(input)

        # y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        # y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        # y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        # output = self.decoder(x2,x3,x4,x5)
        x1 = self.downconv1(input) #64 H,W
        # input_seg = input[:,0,:,:]
        
        x2 = self.downconv2(x1) #128 H/2,W/2
        x3 = self.downconv3(x2*torch.sigmoid(seg_ft1)) #256 H/4,W/4
        x4 = self.downconv4(x3) #512 H/8,W/8
        x5 = self.downconv5(x4) #512 H/8,W/8
        y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        y2 = y2 * torch.sigmoid(seg_ft2)
        y1 = self.upconv1(torch.cat([y2, x2], dim=1))#+ torch.cat([y2, x2], dim=1) * torch.sigmoid(seg_ft2)) #64 H,W
        output = y1
        return output

class discriminator(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.dis = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_nc, nf, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
        )

    # forward method
    def forward(self, input):

        output = self.dis(input)

        return F.sigmoid(output)

class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


