import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from models.norm_layer import *
from torchinfo import summary
from ops.torch.cga import *
from ops.torch.dehaze import DehazeModule
from ops.torch.cbam import CBAM
from ops.torch.Aspp import ASPP
# 原论文
class ConvLayer(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.depth = net_depth
		self.net_depth = self.depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
		)

		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.proj = nn.Conv2d(dim, dim, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X) # X[1,32,256,256] -> [1,32,256,256]
		out = self.proj(out)
		return out #[1,32,256,256]

# 改进
class ConvLayer_improve(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.depth = net_depth
		self.net_depth = self.depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect'),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.Wg = nn.Sequential(
			ASPP(dim)
		)

		self.proj = nn.Conv2d(dim, dim, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		return out




class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter//(2**i), num_filter//(2**(i+1)), kernel_size, stride, padding, bias, activation, norm=None)
            )
            self.down_convs.append(
                ConvBlock(num_filter//(2**(i+1)), num_filter//(2**i), kernel_size, stride, padding, bias, activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft- len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft-i-1](ft_fusion - ft_h_list[i]) + ft_l_list[len(ft_h_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion

class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter*(2**i), num_filter*(2**(i+1)), kernel_size, stride, padding, bias, activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter*(2**(i+1)), num_filter*(2**i), kernel_size, stride, padding, bias, activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft- len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft-i-1](ft_fusion - ft_l_list[i]) + ft_h_list[len(ft_l_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion

class BasicBlock(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer_improve, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
		super().__init__()
		self.norm = norm_layer(dim)
		self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)
	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		return x

class BasicBlock_GCA(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid,reduction=8):
		super().__init__()
		self.norm = norm_layer(dim)
		self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)
		self.sa = SpatialAttention()
		self.ca = ChannelAttention(dim, reduction)
		self.pa = PixelAttention(dim)
	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		res = x
		cattn = self.ca(res)
		sattn = self.sa(res)
		pattn1 = sattn + cattn
		pattn2 = self.pa(res, pattn1)
		res = res * pattn2
		x = res + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer_improve, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x


class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1) #torch.sum(in_feats, dim=1) 对 in_feats 张量沿着第二个维度进行求和，即将所有特征图相应位置的像素值相加，
		# 得到一个大小为 [batch_size, channel, height, width] 的张量。这个张量的形状已经把原来的五维张量中的第二个维度 height 给消除了，将其变成了四维张量，其中的通道维度 channel 保持不变，表示特征图的通道数；
		# height 和 width 表示特征图的高度和宽度，表示特征图中的像素点。这个张量是后续计算注意力系数的输入，表示融合所有特征图后的信息。
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out


class gUNet(nn.Module):
	def __init__(self, kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer_improve, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths) # 7
		half_num = stage_num // 2  # 3
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()
		self.DFF1 = nn.ModuleList() #MSBDN模型中的DFF模块
		self.DFF2 = nn.ModuleList()  # MSBDN模型中的DFF模块

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)

		#DFF1 编码器
		self.DFF1.append(Encoder_MDCBlock1(48,2,mode='iter2'))
		self.DFF1.append(Encoder_MDCBlock1(96,3,mode='iter2'))
		self.DFF1.append(Encoder_MDCBlock1(192,4,mode='iter2'))

		#DFF2 解码器
		self.DFF2.append(Decoder_MDCBlock1(24, 4, mode='iter2'))
		self.DFF2.append(Decoder_MDCBlock1(48,3,mode='iter2'))
		self.DFF2.append(Decoder_MDCBlock1(96,2,mode='iter2'))

		#DFE模块
		self.dehaze_block = DehazeModule()
	def forward(self, x):
		feat = self.inconv(x)

		skips = []
		feature_mem = []
		for i in range(self.half_num):
			feat = self.layers[i](feat)
			feature_mem.append(feat)
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)
			feat = self.DFF1[i](feat,feature_mem)

		feat = self.layers[self.half_num](feat) #[1,192,32,32]
		feat = self.dehaze_block(feat)
		feature_mem.clear()
		feature_mem.append(feat)

		for i in range(self.half_num-1, -1, -1):  # feat 192 -> 96 -> 48 -> 24  192,32,32  96,64,64
			feat = self.ups[i](feat)              # res8x_1 -> [1, 64, 32, 32]  feature_mem_up -> [1, 128, 16, 16]
			feat = self.DFF2[i](feat, feature_mem)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)
			feature_mem.append(feat)
		x = self.outconv(feat) + x

		return x

class gUNet_before(nn.Module):
	def __init__(self, kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet_before, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths) # 7
		half_num = stage_num // 2  # 3
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()
		#self.DFF1 = nn.ModuleList() #MSBDN模型中的DFF模块
		#self.DFF2 = nn.ModuleList()  # MSBDN模型中的DFF模块

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)

		#DFF1 编码器
		#self.DFF1.append(Encoder_MDCBlock1(48,2,mode='iter2'))
		#self.DFF1.append(Encoder_MDCBlock1(96,3,mode='iter2'))
		#self.DFF1.append(Encoder_MDCBlock1(192,4,mode='iter2'))

		#DFF2 解码器
		#self.DFF2.append(Decoder_MDCBlock1(24, 4, mode='iter2'))
		#self.DFF2.append(Decoder_MDCBlock1(48,3,mode='iter2'))
		#self.DFF2.append(Decoder_MDCBlock1(96,2,mode='iter2'))

		#DFE模块
		#self.dehaze_block = DehazeModule()
	def forward(self, x):
		feat = self.inconv(x)

		skips = []
		#feature_mem = []
		for i in range(self.half_num):
			feat = self.layers[i](feat)
			#feature_mem.append(feat)
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)
			#feat = self.DFF1[i](feat,feature_mem)

		feat = self.layers[self.half_num](feat) #[1,192,32,32]
		#feat = self.dehaze_block(feat)
		#feature_mem.clear()
		#feature_mem.append(feat)

		for i in range(self.half_num-1, -1, -1):  # feat 192 -> 96 -> 48 -> 24  192,32,32  96,64,64
			feat = self.ups[i](feat)              # res8x_1 -> [1, 64, 32, 32]  feature_mem_up -> [1, 128, 16, 16]
			#feat = self.DFF2[i](feat, feature_mem)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)
			#feature_mem.append(feat)
		x = self.outconv(feat) + x

		return x

__all__ = ['gUNet', 'gunet_t', 'gunet_s', 'gunet_b', 'gunet_d','gunet_before']

# Normalization batch size of 16~32 may be good
def gunet_before():
	return gUNet_before(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_t():	# 4 cards 2080Ti
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_s():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[4, 4, 4, 8, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_b():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[8, 8, 8, 16, 8, 8, 8], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_d():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[16, 16, 16, 32, 16, 16, 16], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

if __name__ == '__main__':
	network = gunet_t()
	x = torch.randn(16, 3, 256, 256)
	summary(network,input_size=(1,3,256,256))
