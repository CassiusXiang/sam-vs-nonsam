from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        se = self.pool(x).view(batch, channels)
        se = self.fc1(se)
        se = self.act1(se)
        se = self.fc2(se)
        se = self.act2(se)
        se = se.view(batch, channels, 1, 1)
        return x * se

class MBConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        expanded_channels = input_channels * expand_ratio
        self.use_skip = stride == 1 and input_channels == output_channels
        self.block = nn.Sequential(OrderedDict([
            ("0_expansion_conv", nn.Conv2d(input_channels, expanded_channels, kernel_size=1, bias=False)),
            ("1_expansion_norm", nn.BatchNorm2d(expanded_channels)),
            ("2_expansion_act", nn.SiLU(inplace=True)),
            ("3_depthwise_conv", nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False)),
            ("4_depthwise_norm", nn.BatchNorm2d(expanded_channels)),
            ("5_depthwise_act", nn.SiLU(inplace=True)),
            ("6_se", SEBlock(expanded_channels, reduction=4)),
            ("7_projection_conv", nn.Conv2d(expanded_channels, output_channels, kernel_size=1, bias=False)),
            ("8_projection_norm", nn.BatchNorm2d(output_channels)),
        ]))

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out

class EfficientNet(nn.Module):
    def __init__(self, in_channels: int, labels: int):
        super(EfficientNet, self).__init__()
        config = [
            (1, 3, 16, 1, 1),
            (6, 3, 24, 2, 2),
            (6, 5, 40, 2, 2),
            (6, 3, 80, 2, 3),
            (6, 5, 112, 1, 3),
            (6, 5, 192, 2, 4),
            (6, 3, 320, 1, 1),
        ]
        self.f = nn.Sequential(OrderedDict([
            ("0_stem_conv", nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)),
            ("1_stem_norm", nn.BatchNorm2d(32)),
            ("2_stem_act", nn.SiLU(inplace=True)),
        ]))
        blocks = []
        current_channels = 32
        for stage_idx, (expand_ratio, kernel_size, output_channels, stride, repeats) in enumerate(config):
            for i in range(repeats):
                s = stride if i == 0 else 1
                block_input_channels = current_channels if i == 0 else output_channels
                blocks.append(
                    (f"{stage_idx}_{i}_mbconv", MBConv(block_input_channels, output_channels, kernel_size, s, expand_ratio))
                )
                if i == 0:
                    current_channels = output_channels
        self.f.add_module("3_blocks", nn.Sequential(OrderedDict(blocks)))
        self.f.add_module("4_head", nn.Sequential(OrderedDict([
            ("0_conv", nn.Conv2d(current_channels, 1280, kernel_size=1, bias=False)),
            ("1_norm", nn.BatchNorm2d(1280)),
            ("2_act", nn.SiLU(inplace=True)),
            ("3_pool", nn.AdaptiveAvgPool2d(1)),
            ("4_flatten", nn.Flatten()),
            ("5_linear", nn.Linear(1280, labels)),
        ])))
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)