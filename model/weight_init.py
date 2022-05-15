import math
import torch.nn as nn


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def c2_xaver_fill(module: nn.Module):
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_weight(m: nn.Module, zero_init_final_gamma=False):
    # Performs ResNet-style weight initialization. ?
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN. Why? bias?
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # out_channels ?
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
                hasattr(m, "final_bn") and m.final_bn and zero_init_final_gamma
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()
