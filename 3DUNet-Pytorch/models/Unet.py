from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Attention_block(nn.Module):  #返回加权的feature map
    def __init__(self, F_g, F_l, F_int): #F_g深层的channel，F_l当前层的channel， F_int中间过程的channel
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()  #转成sttention系数
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 来自high level的gating signal 卷积(已经上采样)
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1,downsampling = False):
        super(ResBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1) #stride = 2表示用于下采样,这里要求图片维度能被2整除

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        )


        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1)  #体素大小不变
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):   #resblock的变体，被认为是效果最好
        res=self.shortcut(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = x + res
        return out

class Up(nn.Module):#包含上采样+特征融合+卷积
    def __init__(self, down_in_channels, out_channels, conv_block=None, interpolation=False, haveattention = False):
        super(Up, self).__init__()
        """
        down_in_channels: the channel of high level;
        out_channels: the channel of this level
        """
        in_channels = out_channels
        self.haveattention = haveattention
        if interpolation == True:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#插值 各个维度* scale_factor
            #align_corners=True 表示根据两边均匀调和插值，不然就只根据一个点去插值
        else:
            self.up = nn.ConvTranspose3d(down_in_channels, down_in_channels, 2, stride=2)  #反卷积

        self.attention = Attention_block(down_in_channels,in_channels,in_channels//2) #
        #!!!//保证是整除，结果为int，否则为float

        if conv_block is None:
            self.conv = nn.Conv3d(in_channels + down_in_channels, out_channels,3,stride=1,padding=1)
        else:
            self.conv = conv_block(in_channels + down_in_channels, out_channels, stride=1) #使用其他卷积
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, down_x, x):
        up_x = self.up(down_x)  #先进性一次上采样(注意channel不变，还是down_in_channel)
        if self.haveattention:
            x = self.attention(up_x, x) #获取attention加权的feature map, channel与x相同
        x = torch.cat((x, up_x), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Down(nn.Module): #包含一个一个改变channel的卷积+降采样操作和一个不改变channel的卷积
    def __init__(self, in_channels, out_channels, conv_block=None, ispooling = True):
        super(Down, self).__init__()
        if conv_block is None:
            self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride= 1,padding= 1)
            self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,stride= 1,padding= 1)
        else:
            self.conv1 = conv_block(in_channels, out_channels)
            self.conv2 = conv_block(out_channels, out_channels)

        if ispooling:
            self.down = nn.MaxPool3d(2, stride=2)  # 维度降为原来的一半
                        # 自适应池化Adaptive Pooling会根据输入的参数来控制输出output_size，而标准的Max/AvgPooling是通过kernel_size，stride与padding来计算output_size
        else:
            self.down =ResBlock(in_channels,out_channels, stride = 2)  #通过残差卷积进行下采样

    def forward(self, x):
        out = self.conv1(x)
        x = self.down(out)
        x = self.conv2(x)
        return out, x

class UNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, conv_block = None):
        super(UNet, self).__init__()
        """
        filter_num_list: channel_num in each layer
        conv_block:  Convolution block_type in Down and Up

        """
        self.inc = nn.Conv3d(in_channels, filter_num_list[0], 3,stride=1,padding=1)
        # down
        self.down1 = Down(filter_num_list[0], filter_num_list[1], conv_block=conv_block)
        self.down2 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block)
        self.down3 = Down(filter_num_list[2], filter_num_list[3], conv_block=conv_block)
        #self.down4 = Down(filter_num_list[3], filter_num_list[4], conv_block=conv_block)
        #self.down5 = Down(filter_num_list[4], filter_num_list[5], conv_block=conv_block)

        self.bridge = nn.Conv3d(filter_num_list[3], filter_num_list[4],3 , stride=1, padding=1)
        #self.res1 = ResBlock(filter_num_list[0], filter_num_list[0], 1, downsampler = False)
        #self.res2 = ResBlock(filter_num_list[1], filter_num_list[1], 1, downsampler = False)
        #self.res3 = ResBlock(filter_num_list[2], filter_num_list[2], 1, downsampler = False)
        #self.res4 = ResBlock(filter_num_list[3], filter_num_list[3], 1, downsampler = False)
        #self.res5 = ResBlock(filter_num_list[4], filter_num_list[4], 1, downsampler = False)

        # up
        #self.up1 = Up(filter_num_list[5], filter_num_list[4], filter_num_list[4],conv_block=conv_block)
        self.up2 = Up(filter_num_list[4], filter_num_list[3], conv_block=conv_block)
        self.up3 = Up(filter_num_list[3], filter_num_list[2], conv_block=conv_block)
        self.up4 = Up(filter_num_list[2], filter_num_list[1], conv_block=conv_block)
        self.up5 = Up(filter_num_list[1], filter_num_list[0], conv_block=conv_block)

        self.conv = nn.Conv3d(filter_num_list[1], filter_num_list[0], 3, padding=1)
        self.class_conv = nn.Conv3d(filter_num_list[0], class_num, kernel_size=1)  #1*1*1conv

    def forward(self, input):

        x = input
        x = self.inc(x)
        conv1, x = self.down1(x) #conv表示后面被cat的值，x是下采样的值
        conv2, x = self.down2(x)
        conv3, x = self.down3(x)
        #conv4, x = self.down4(x)

        #conv5, x = self.down5(x)
        #x = self.up1(x, conv5)
        x = self.bridge(x)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)
        #x = self.up5(x, conv1)

        x = self.conv(x)
        x = self.class_conv(x)

        #x = nn.Softmax(1)(x)
        return x












# 未测试
class SEBlock(nn.Module):
    def __init__(self,in_channels,out_channels,net_mode='2d'):
        super(SEBlock,self).__init__()

        if net_mode == '2d':
            self.gap=nn.AdaptiveAvgPool2d(1)
            conv=nn.Conv2d
        elif net_mode == '3d':
            self.gap=nn.AdaptiveAvgPool3d(1)
            conv=nn.Conv3d
        else:
            self.gap=None
            conv=None

        self.conv1=conv(in_channels,out_channels,1)
        self.conv2=conv(in_channels,out_channels,1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        inpu=x
        x=self.gap(x)
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.sigmoid(x)

        return inpu*x
# 未测试
class DenseBlock(nn.Module):
    def __init__(self,channels,conv_num,net_mode='2d'):
        super(DenseBlock,self).__init__()
        self.conv_num=conv_num
        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.relu=nn.ReLU()
        self.conv_list=[]
        self.bottle_conv_list=[]
        for i in conv_num:
            self.bottle_conv_list.append(conv(channels*(i+1),channels*4,1))
            self.conv_list.append(conv(channels*4,channels,3,padding=1))


    def forward(self,x):

        res_x=[]
        res_x.append(x)

        for i in self.conv_num:
            inputs=torch.cat(res_x,dim=1)
            x=self.bottle_conv_list[i](inputs)
            x=self.relu(x)
            x=self.conv_list[i](x)
            x=self.relu(x)
            res_x.append(x)

        return x
class SegSEBlock(nn.Module):  #为了解决在卷积池化过程中feature map的不同通道所占的重要性不同带来的损失问题
    def __init__(self, in_channels, rate=2, net_mode='2d'):
        super(SegSEBlock, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None
        self.in_channels = in_channels
        self.rate = rate
        self.gp = nn.AvgPool3d(1)
        self.dila_conv = conv(self.in_channels, self.in_channels // self.rate, 3, padding=2, dilation=self.rate) #dilation 膨胀
        #//整除  padding扩充边界
        self.conv1 = conv(self.in_channels // self.rate, self.in_channels, 1)

    def forward(self, input):
                                   #这里应该有一个全局平均池化？
        x = self.dila_conv(input) #降维
        x = self.conv1(x)  #升维
        x = nn.Sigmoid()(x)

        return x
#在传统的卷积池化过程中，默认feature map的每个通道是同等重要的，而在实际的问题中，不同通道的重要性是有差异的(注意力机制)
#SE模块是在channel维度上做attention或者gating操作，这种注意力机制让模型可以更加关注信息量最大的channel特征，而抑制那些不重要的channel特征
#只要卷积核是3，padding=1，输入跟输出的矩阵就是一样大，然后只需要设置channel就行了
class RecombinationBlock(nn.Module):  #让网络学习到如何混合信息来生成更加特别的特征
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='2d'):
        super(RecombinationBlock, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = conv(self.in_channels, self.expan_channels, 1)
        self.skip_conv = conv(self.in_channels, self.out_channels, 1)
        self.zoom_conv = conv(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = conv(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)

    def forward(self, input):
        x = self.expansion_conv(input)  #增加feature  ,扩充channel

        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6()(x)
            x = self.norm_conv(x)

        se_x = self.segse_block(x)

        x = x * se_x

        x =   self.zoom_conv(x)  #变成输出channel
        skip_x = self.skip_conv(input) #输出channel
        out = x + skip_x
        return out

def main():

    model = UNet(1, [32, 48, 64, 96, 128], 2, net_mode='3d',conv_block=ResBlock)
    print(model)

if __name__ == '__main__':
    main()

