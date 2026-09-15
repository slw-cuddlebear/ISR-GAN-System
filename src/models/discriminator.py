# src/models/discriminator.py
import torch
import torch.nn as nn


class ConditionalDiscriminator(nn.Module):
    """条件判别器，输入4通道（RGB+语义掩码），输出真伪概率"""

    def __init__(self, in_channels=4, feat_channels=64):
        super(ConditionalDiscriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            # 第0层: Conv 4->64, 3x3, stride1, padding1
            nn.Conv2d(in_channels, feat_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 第2层: Conv 64->128, stride2
            nn.Conv2d(feat_channels, feat_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 第5层: Conv 128->256, stride2
            nn.Conv2d(feat_channels * 2, feat_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 第8层: Conv 256->512, stride2
            nn.Conv2d(feat_channels * 4, feat_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 第11层: Conv 512->1024, stride2
            nn.Conv2d(feat_channels * 8, feat_channels * 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 全局平均池化 + 全连接输出1维
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feat_channels * 16, 1)

    def forward(self, img, mask):
        """
        img: (B,3,H,W) 范围[-1,1]
        mask: (B,1,H,W) 语义掩码，值范围[0,1]（我们暂时生成全1掩码）
        """
        x = torch.cat([img, mask], dim=1)  # (B,4,H,W)
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)  # (B,1024,1,1)
        x = x.view(x.size(0), -1)  # (B,1024)
        out = self.fc(x)  # (B,1)
        return out