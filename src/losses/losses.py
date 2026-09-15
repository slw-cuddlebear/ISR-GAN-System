# src/losses/losses.py
import torch
import torch.nn as nn
import torchvision.models as models


class RegionWeightedMSELoss(nn.Module):
    """区域加权MSE损失，如果mask为None则退化为普通MSE"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None, class_weights={0: 1.0, 1: 5.0, 2: 5.0, 3: 5.0}):
        """
        pred, target: (B,3,H,W) 范围[-1,1]
        mask: (B,1,H,W) 语义类别索引 (0,1,2,3)，可选
        """
        diff = (pred - target) ** 2
        if mask is not None:
            weight_map = torch.zeros_like(mask, dtype=torch.float32)
            for cls, w in class_weights.items():
                weight_map[mask == cls] = w
            weight_map = weight_map.expand(-1, 3, -1, -1)
            loss = (diff * weight_map).mean()
        else:
            loss = diff.mean()
        return loss


class PerceptualLoss(nn.Module):
    """基于VGG19的感知损失，使用第9层ReLU激活前的特征"""

    def __init__(self, layer=9):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        # 输入范围[-1,1] -> 映射到[0,1]供VGG使用
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2
        pred_feat = self.feature_extractor(pred_norm)
        target_feat = self.feature_extractor(target_norm)
        return self.criterion(pred_feat, target_feat)


class AdversarialLoss(nn.Module):
    """生成器的对抗损失（非饱和损失）"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, fake_pred):
        # fake_pred: 判别器对生成图像的输出概率
        target = torch.ones_like(fake_pred)
        return self.bce(fake_pred, target)