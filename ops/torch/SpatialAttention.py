import torch
import torch.nn as nn
from torchinfo import summary


class SpatialAttention(nn.Module):
	def __init__(self, nc, number, norm_layer=nn.BatchNorm2d):
		super(SpatialAttention, self).__init__()
		self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = norm_layer(nc)
		self.prelu = nn.PReLU(nc)
		self.conv2 = nn.Conv2d(nc, number, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = norm_layer(number)

		self.conv3 = nn.Conv2d(number, number, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
		self.conv4 = nn.Conv2d(number, number, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
		self.conv5 = nn.Conv2d(number, number, kernel_size=3, stride=1, padding=7, dilation=7, bias=False)

		self.fc1 = nn.Conv2d(number * 4, 1, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU()
		self.fc2 = nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x0 = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.prelu(x)
		x = self.conv2(x)
		x = self.bn2(x)

		x1 = x
		x2 = self.conv3(x)
		x3 = self.conv4(x)
		x4 = self.conv5(x)

		se = torch.cat([x1, x2, x3, x4], dim=1)

		se = self.fc1(se)
		se = self.relu(se)
		se = self.fc2(se)
		se = self.sigmoid(se)

		return se

if __name__ == "__main__":
    model = SpatialAttention(512,512)
    summary(model, input_size=(1, 512, 256, 256))