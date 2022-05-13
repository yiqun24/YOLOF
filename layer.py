import torch.nn as nn
# noinspection PyTypeChecker


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, p=0, dilation=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
