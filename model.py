from doctest import OutputChecker
import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from torchvision import models
from torch.autograd import Variable
import numpy as np
import math
from haar import * 

norm_layer = nn.InstanceNorm2d

class ChannelAtten(nn.Module):
    def __init__(self, in_features):
        super(ChannelAtten, self).__init__()

        channel_block = [ 
                          nn.Conv2d(in_features, in_features, 1),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(in_features, in_features, 1),
                          nn.Sigmoid()
                        ]
        
        self.channel_block = nn.Sequential(*channel_block)

    def forward(self, x):
        base = F.adaptive_avg_pool2d(x, (1, 1))
        base = base.view(base.size(0), base.size(1), 1, 1)
        base = self.channel_block(base).expand(base.size(0), base.size(1), x.size(2), x.size(3))
        out = torch.mul(x, base)

        return out
    
class ResidualAttenBlockn2(nn.Module):
    def __init__(self, in_features, size):
        super(ResidualAttenBlockn2, self).__init__()

        conv_block1 = [ norm_layer(in_features),
                        nn.ZeroPad2d((size - 1) // 2),
                        nn.Conv2d(in_features, in_features, size),
                        nn.ReLU(inplace=True),
                        ChannelAtten(in_features),
                        nn.Conv2d(in_features, in_features, 1)
        ]

        conv_block2 = [ norm_layer(in_features),
                        nn.ZeroPad2d((size - 1) // 2),
                        nn.Conv2d(in_features, in_features, size),
                        ]
        
        conv_block3 = [ norm_layer(in_features),
                        nn.Conv2d(in_features, in_features, 1)
                        ]

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.conv_block3 = nn.Sequential(*conv_block3)

    def forward(self, x):
        out1 = self.conv_block1(x) + x
        out = self.conv_block2(out1)
        out = mpatten(out)
        out = self.conv_block3(out) + out1

        return out

class ResidualAttenBlockn3(nn.Module):
    def __init__(self, in_features, size):
        super(ResidualAttenBlockn3, self).__init__()

        conv_block1 = [ norm_layer(in_features),
                        nn.ZeroPad2d((size - 1) // 2),
                        nn.Conv2d(in_features, in_features, size),
                        nn.ReLU(inplace=True),
                        ChannelAtten(in_features),
                        nn.Conv2d(in_features, in_features, 1)
        ]

        conv_block2 = [ norm_layer(in_features),
                        nn.ZeroPad2d((size - 1) // 2),
                        nn.Conv2d(in_features, in_features, size),
                        ]
        
        conv_block3 = [ norm_layer(in_features),
                        nn.Conv2d(in_features, in_features, 1)
                        ]

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.conv_block3 = nn.Sequential(*conv_block3)

    def forward(self, x):
        out1 = self.conv_block1(x) + x
        out = self.conv_block2(out1)
        out = mpatten_haar(out)
        out = self.conv_block3(out) + out1

        return out

class Upsampling(nn.Module):
    def __init__(self, in_features, out_features):
        super(Upsampling, self).__init__()
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)

class Downsampling(nn.Module):
    def __init__(self, in_features, out_features):
        super(Downsampling, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [  nn.ZeroPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        in_channel = 64
        self.model1 = Downsampling(in_channel, in_channel * 2)
        #Residual
        in_channel *= 2
        self.model2 = Downsampling(in_channel, in_channel * 2)
        in_channel *= 2

        res2 = []
        for _ in range(n_residual_blocks - 1):
            res2 += [ResidualAttenBlockn3(in_channel, 3)]
        res2 += [ResidualAttenBlockn2(in_channel, 3)]
        self.res2 = nn.Sequential(*res2)

        #Upsampling
        self.model5 = Upsampling(in_channel, in_channel // 2)
        in_channel = in_channel // 2
        self.model6 = Upsampling(in_channel, in_channel // 2)
        in_channel = in_channel // 2

        # Output layer
        model_end = [ nn.ZeroPad2d(3),
                      nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model_end += [nn.Sigmoid()]

        self.model_end = nn.Sequential(*model_end)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out) #1/2
        out = self.model2(out) #1/4
        out = self.res2(out)
        out = self.model5(out)
        out = self.model6(out)
        out =self.model_end(out)

        return out
    
    def middle_out(self, x):
        out = self.model0(x)
        out = self.model1(out) #1/2
        out = self.model2(out) #1/4
        out = self.res2[-1].conv_block2(out)

        return out
    
    def atten(x):
        out = mpatten(x)
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GlobalGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', use_sig=False, n_UPsampling=0):
        assert(n_blocks >= 0)
        super(GlobalGenerator2, self).__init__()        
        activation = nn.ReLU(True)        

        mult = 8
        model = [nn.ReflectionPad2d(4), nn.Conv2d(input_nc, ngf*mult, kernel_size=7, padding=0), norm_layer(ngf*mult), activation]

        ### downsample
        for i in range(n_downsampling):
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=4, stride=2, padding=1),
                      norm_layer(ngf * mult // 2), activation]
            mult = mult // 2

        if n_UPsampling <= 0:
            n_UPsampling = n_downsampling

        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_UPsampling):
            next_mult = mult // 2
            if next_mult == 0:
                next_mult = 1
                mult = 1

            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * next_mult), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * next_mult)), activation]
            mult = next_mult

        if use_sig:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        else:      
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input, cond=None):
        return self.model(input)


class InceptionV3(nn.Module): #avg pool
    def __init__(self, num_classes, isTrain, use_aux=True, pretrain=False, freeze=True, every_feat=False):
        super(InceptionV3, self).__init__()
        """ Inception v3 expects (299,299) sized images for training and has auxiliary output
        """

        self.every_feat = every_feat

        self.model_ft = models.inception_v3(pretrained=pretrain)
        stop = 0
        if freeze and pretrain:
            for child in self.model_ft.children():
                if stop < 17:
                    for param in child.parameters():
                        param.requires_grad = False
                stop += 1

        num_ftrs = self.model_ft.AuxLogits.fc.in_features #768
        self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net
        num_ftrs = self.model_ft.fc.in_features #2048
        self.model_ft.fc = nn.Linear(num_ftrs,num_classes)

        self.model_ft.input_size = 299

        self.isTrain = isTrain
        self.use_aux = use_aux

        if self.isTrain:
            self.model_ft.train()
        else:
            self.model_ft.eval()


    def forward(self, x, cond=None, catch_gates=False):
        # N x 3 x 299 x 299
        x = self.model_ft.Conv2d_1a_3x3(x)

        # N x 32 x 149 x 149
        x = self.model_ft.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model_ft.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.model_ft.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model_ft.Conv2d_4a_3x3(x)

        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.model_ft.Mixed_5b(x)
        feat1 = x
        # N x 256 x 35 x 35
        x = self.model_ft.Mixed_5c(x)
        feat11 = x
        # N x 288 x 35 x 35
        x = self.model_ft.Mixed_5d(x)
        feat12 = x
        # N x 288 x 35 x 35
        x = self.model_ft.Mixed_6a(x)
        feat2 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6b(x)
        feat21 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6c(x)
        feat22 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6d(x)
        feat23 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6e(x)

        feat3 = x

        # N x 768 x 17 x 17
        aux_defined = self.isTrain and self.use_aux
        if aux_defined:
            aux = self.model_ft.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model_ft.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model_ft.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        feats = F.dropout(x, training=self.isTrain)
        # N x 2048 x 1 x 1
        x = torch.flatten(feats, 1)
        # N x 2048
        x = self.model_ft.fc(x)
        # N x 1000 (num_classes)

        if self.every_feat:
            # return feat21, feats, x
            return x, feat21

        return x, aux


def weight(x):
    n, c, h, w = x.size()
    x_sum_w = x.sum(dim=-1).unsqueeze(-1).expand(n, c, h, w)
    x_sum_h = x.sum(dim=-2).unsqueeze(-2).expand(n, c, h, w)
    map = x_sum_h + x_sum_w
    return map

def weight_ww(x):
    n, c, h, w = x.size()
    x_sum = x.sum(dim=-1).unsqueeze(-1).expand(n, c, h, w)
    x_sum2 = torch.transpose(x,2,3).sum(dim=-1).unsqueeze(-1).expand(n, c, h, w)
    map = x_sum + x_sum2
    return map

def weight_hh(x):
    n, c, h, w = x.size()
    x_sum = x.sum(dim=-2).unsqueeze(-2).expand(n, c, h, w)
    x_sum2 = torch.transpose(x,2,3).sum(dim=-2).unsqueeze(-2).expand(n, c, h, w)
    map = x_sum + x_sum2
    return map

def mpatten(x):
    atten_wh = weight(torch.matmul(x, x))
    atten_ww = weight_ww(torch.matmul(x, torch.transpose(x, 2, 3)))
    atten_hh = weight_hh(torch.matmul(torch.transpose(x, 2, 3), x))
    atten_hw = weight(torch.matmul(torch.transpose(x, 2, 3), torch.transpose(x, 2, 3)))
    atten = atten_wh + atten_hh + atten_ww + atten_hw
    out = torch.mul(x,atten)
    return out

def mpatten_haar(x):
    x_LL = mpatten(LL(x))
    x_HL = mpatten(HL(x))
    x_LH = mpatten(LH(x))
    x_HH = mpatten(HH(x))
    out = inverse_haar(x_LL, x_HL, x_LH, x_HH)
    return out
