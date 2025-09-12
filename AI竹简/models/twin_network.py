import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DistanceLoss(nn.Module):
    #距离损失函数
    def __init__(self, margin=2.0):
        super(DistanceLoss, self).__init__()
        self.margin = margin

    def forward(self, features1, features2, label):
        distance = F.pairwise_distance(features1, features2)
        # 正样本：最小化距离，负样本：最大化距离（至少到margin）
        loss = torch.mean(
            label * torch.pow(distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        return loss


class TwinNetwork(nn.Module):
    #孪生网络模型
    def __init__(self, feature_dim=128):
        super(TwinNetwork, self).__init__()
        # 使用ResNet18作为骨干网络
        backbone = models.resnet18(weights=None)

        # 修改输入层为单通道
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 复制原始权重的平均值到新的单通道
        with torch.no_grad():
            if hasattr(original_conv, 'weight'):
                backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # 去掉最后的分类层
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # 添加特征映射层
        self.feature_mapper = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim)
        )

    def extract_features(self, x):
        # 提取单个图像特征
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.feature_mapper(features)
        return features

    def forward(self, img1, img2):
        # 前向传播
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        return features1, features2


def build_twin_model(feature_dim=128):
    # 创建孪生网络模型
    return TwinNetwork(feature_dim=feature_dim)


def create_distance_loss(margin=2.0):
    # 创建距离损失函数
    return DistanceLoss(margin=margin)
