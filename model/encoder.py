import torch.nn as nn

from .weight_init import c2_xaver_fill
from .layer import Conv


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
        c2_xaver_fill(self.lateral[0])
        c2_xaver_fill(self.fpn[0])
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
        out = self.lateral(feature)
        out = self.fpn(out)
        return self.dilation_encoders(out)


if __name__ == '__main__':
    encoder = DilatedEncoder()
