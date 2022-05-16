import torch.nn as nn

import weight_init
from layer import Conv

from misc import nms # add


class Bottleneck(nn.Module):
    def __init__(self, in_channels=512, mid_channels=128, dilation=1):
        super(Bottleneck, self).__init__()
        self.main_path = nn.Sequential(
            Conv(in_channels, mid_channels, k=1),
            Conv(mid_channels, mid_channels, k=3, p=dilation, dilation=dilation),
            Conv(mid_channels, in_channels, k=1)
        )

    def forward(self, x):
        return x + self.main_path(x)


# noinspection PyTypeChecker
class DilatedEncoder(nn.Module):
    def __init__(self, in_channels=1024, encoder_channels=512, block_mid_channels=128, residual_block_num=4,
                 dilation_rates=None):  
        super(DilatedEncoder, self).__init__()
        if dilation_rates is None:
            dilation_rates = [2, 4, 6, 8]
        assert len(dilation_rates) == residual_block_num

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.block_mid_channels = block_mid_channels
        # ============================================= 算法改进加的
        self.L = [] # add
        self.W = [1]*len(dilation_rates) # add 
        self.conv1 = nn.Conv2d(in_channels, encoder_channels, kernel_size=1) # add
        self.conv2 = nn.Conv2d(in_channels, encoder_channels, kernel_size=1) #add
        self.conv3 = nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1) #add
        
        # ============================================= 算法改进加的
        # 为什么没有BatchNorm? 这样处理的目的是什么?
        self.lateral = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels, kernel_size=1),
            nn.BatchNorm2d(encoder_channels)
        )
        self.fpn = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels)
        )
        encoder_blocks = []
        for dilation in dilation_rates:
            encoder_blocks.append(
                Bottleneck(
                    encoder_channels,
                    block_mid_channels,
                    dilation
                )
            )
        self.dilation_encoders = nn.Sequential(*encoder_blocks)

        

        self._init_weights()

    def _init_weights(self):
        # module(): generator
        # BatchNorm vs bias?
        weight_init.c2_xaver_fill(self.lateral[0])
        weight_init.c2_xaver_fill(self.fpn[0])
        for m in [self.lateral[1], self.fpn[1]]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        for m in self.dilation_encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        output = self.fpn(self.lateral(feature))  # 先过一层projector
        self.L.append(output)  # 结果存一份，用于特征融合
        for i in range(0,len(self.dilation_encoders)):   # 每个 backbone 做一次
            output = self.dilation_encoders[i](output)
            self.L.append(output)  # 结果存一份，用于特征融合
            weight = nms(self.conv1(output),self.conv2(self.L[i-1]))   # 计算权重
            weight = nn.Softmax(self.conv3(weight))                    # 计算权重 
            self.W[i]=weight
            output = feature*self.W[i][0] + self.L[i-1] * self.W[i-1][0] # 
            self.L.append(output)                             # 计算结果先记录下来
        eps = 1
        for i in range(1,len(self.W)):
            sumW += self.W[i]
        sumW += eps # 防止除零
        p1_4 = self.L[0]* self.W[0][0]/sumW
        for i in range(1,len(self.W)-1):
            p1_4 += self.L[i]* self.W[i]/sumW
        p5 = (self.L[i]*self.W[i][0] + p1_4*self.W[1][0])/(self.W[i][0] +self.W[1][0]+eps)
        return p5







def build_encoder():
    # add config!
    return DilatedEncoder()


if __name__ == '__main__':
    encoder = DilatedEncoder()
