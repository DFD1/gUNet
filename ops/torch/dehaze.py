import torch.nn as nn
import torch
from ops.torch.cbam import CBAM
from ops.torch.SpatialAttention import SpatialAttention
#from DCNv2_latest.dcn_v2 import DCN
from torchinfo import summary

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )

class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.td = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.ka(a)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# class DCNBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(DCNBlock, self).__init__()
#         self.dcn = DCN(
#             in_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1
#         )
#
#     def forward(self, x):
#         return self.dcn(x)


class Block(nn.Module):
    def __init__(
        self,
        conv,
        dim,
        kernel_size,
    ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
        #self.pdu = PDU(dim)
        self.cbam = CBAM(dim)
        #self.SpatialAttention = SpatialAttention(dim,dim)

    def forward(self, x): #Estimated Total Size (MB): 1359.65
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        temp = res
        res1 = self.calayer(res)
        res1 = self.palayer(res1)
        res2 = self.cbam(temp)
        #res3 = self.pdu(temp)
        res = res1 * res2
        res += x
        return res


    # def forward(self, x): #Estimated Total Size (MB): 6231.35
    #     res = self.act1(self.conv1(x))
    #     res = res + x
    #     res = self.conv2(res)
    #     temp = res
    #     res1 = self.calayer(res)
    #     res1 = self.palayer(res1)
    #     res2 = self.SpatialAttention(temp)
    #     res = res1 * res2
    #     res += x
    #     return res

    # def forward(self, x): #Estimated Total Size (MB): 1359.65
    #     res = self.act1(self.conv1(x))
    #     res = res + x
    #     res = self.conv2(res)
    #     res = self.calayer(res)
    #     res = self.palayer(res)
    #     res = self.cbam(res)
    #     res += x
    #     return res


class DehazeModule(nn.Module):
    def __init__(self, blocks=6, conv=default_conv, io_channel=192):
        super(DehazeModule, self).__init__()
        self.dim = io_channel
        kernel_size = 3
        modules = [Block(conv, self.dim, kernel_size) for _ in range(blocks)]
        self.gp = nn.Sequential(*modules)
        #self.dcn_block = DCNBlock(self.dim, self.dim)

    def forward(self, x):
        res = self.gp(x)
       #res_dcn1 = self.dcn_block(res)
        #res_dcn2 = self.dcn_block(res_dcn1)
        return res

if __name__ == "__main__":
    model = DehazeModule()
    summary(model, input_size=(1, 192, 256, 256))

