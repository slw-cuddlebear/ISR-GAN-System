# src/models/generator.py
import torch
import torch.nn as nn


class SimplifiedChannelAttention(nn.Module):
    """简化通道注意力模块 (SCA)"""

    def __init__(self, channels, reduction=4):
        super(SimplifiedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y + x  # 残差连接


class LightweightResidualDenseBlock(nn.Module):
    """轻量化残差密集块 (LRDB)"""

    def __init__(self, n_feats=64, growth_channel=32):
        super(LightweightResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, growth_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(n_feats + growth_channel, growth_channel, 3, padding=1)
        self.conv3 = nn.Conv2d(n_feats + 2 * growth_channel, n_feats, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.attention = SimplifiedChannelAttention(n_feats)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        cat1 = torch.cat([x, x1], dim=1)
        x2 = self.act(self.conv2(cat1))
        cat2 = torch.cat([x, x1, x2], dim=1)
        out = self.conv3(cat2)
        out = self.attention(out)
        return out + x


class Generator(nn.Module):
    """改进生成器：轻量化残差密集块 + 简化通道注意力，4倍上采样"""

    def __init__(self, config):
        super(Generator, self).__init__()
        n_feats = config.n_feats
        n_blocks = config.n_res_blocks

        # 浅层特征提取
        self.conv_first = nn.Conv2d(3, n_feats, 3, padding=1)

        # 残差密集块堆叠
        self.blocks = nn.Sequential(
            *[LightweightResidualDenseBlock(n_feats, config.growth_channel) for _ in range(n_blocks)])

        # 特征融合
        self.conv_mid = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # 上采样部分（两次PixelShuffle，每次2倍，共4倍）
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 输出层
        self.conv_last = nn.Conv2d(n_feats, 3, 3, padding=1)
        self.tanh = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, 64, 64)
        feat = self.conv_first(x)
        residual = feat
        feat = self.blocks(feat)
        feat = self.conv_mid(feat)
        feat = feat + residual  # 全局残差连接
        out = self.upsample(feat)
        out = self.conv_last(out)
        out = self.tanh(out)    # 输出值范围[-1,1]
        return out