import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=512):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:] #torch.Size([256, 256])

        image_features = self.mean(x) #(1,512,1,1)
        image_features = self.conv(image_features) #(1,256,1,1)
        image_features = F.upsample(image_features, size=size, mode='bilinear') #(1,256,256,256)

        atrous_block1 = self.atrous_block1(x) #(1,256,256,256)
        atrous_block6 = self.atrous_block6(x) #(1,256,256,256)
        atrous_block12 = self.atrous_block12(x) #(1,256,256,256)
        atrous_block18 = self.atrous_block18(x) #(1,256,256,256)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net #(1,256,256,256)

if __name__ == "__main__":
    model = ASPP(512)
    summary(model, input_size=(1, 512, 256, 256))