import torch
import torch.nn as nn
from typing import Tuple
from layer import Conv


# noinspection PyTypeChecker
class Decoder(nn.Module):
    def __init__(self,
                 in_channels=256,
                 kernel_size=3,
                 padding=1,
                 num_classes=80,
                 num_anchors=1):
        super(Decoder, self).__init__()
        self.INF = 1e8

        self.cls_subnet = nn.Sequential(
            Conv(in_channels, in_channels, k=kernel_size, p=padding),
            Conv(in_channels, in_channels, k=kernel_size, p=padding),
        )
        self.reg_subnet = nn.Sequential(
            Conv(in_channels, in_channels, k=kernel_size, p=padding),
            Conv(in_channels, in_channels, k=kernel_size, p=padding),
            Conv(in_channels, in_channels, k=kernel_size, p=padding),
            Conv(in_channels, in_channels, k=kernel_size, p=padding),
        )
        self.cls_score = nn.Conv2d(in_channels,
                                   num_anchors * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.reg_pred = nn.Conv2d(in_channels,
                                  num_anchors * 4,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        # Is there a object?
        self.object_pred = nn.Conv2d(in_channels,
                                     num_anchors,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.SyncBatchNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        cls_score = self.cls_score(self.cls_subnet(feature))
        reg_feats = self.reg_subnet(feature)
        bbox_reg = self.reg_pred(reg_feats)
        obj_pred = self.object_pred(reg_feats)

        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        # implicit object prediction
        obj_pred = obj_pred.view(N, -1, 1, H, W)
        # plus? normalize?
        normalized_cls_score = cls_score + obj_pred - torch.log(
            1. + torch.clamp(cls_score.exp(), max=self.INF) +
            torch.clamp(obj_pred.exp(), max=self.INF)
        )

        # [B, A, K, H, W] -> [B, H*W*A, k]
        normalized_cls_score = normalized_cls_score.permute(0, 3, 4, 1, 2)
        normalized_cls_score = normalized_cls_score.reshape(N, -1, self.num_classes)

        bbox_reg = bbox_reg.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        bbox_reg = bbox_reg.view(N, -1, 4)

        return normalized_cls_score, bbox_reg


def build_decoder():
    return Decoder()
