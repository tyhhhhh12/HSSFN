import torch
import torch.nn as nn

import torch.nn.functional as F
# from model.FreqFusion import FreqFusion
from FreqFusion import FreqFusion



#PSFEM
class SpectralFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 对每个像素应用 1D 卷积（沿通道轴）
        self.conv1d3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1d5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1d7 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

        # 自适应特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # 输入形状: (batch, channels, height, width)
        batch_size, channels, height, width = x.size()
        # 转置为 (batch, H, W, C)，以便对每个像素应用 1D 卷积
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, C)
        b3 = self.conv1d3(x)
        b5 = self.conv1d5(x)
        b7 = self.conv1d7(x)
        x = self.fusion(torch.cat([b3, b5, b7], dim=1))
        return x.permute(0, 3, 1, 2)


#改进卷积模块 添加残差连接
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)  # 添加跳跃连接

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# 修改通道
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


#输出层
class OutputBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 全局平均池化将特征图压缩到1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层输出num_classes个类别
        self.fc = nn.Linear(in_channels, num_classes)
        self.activation = nn.Softmax(dim=1)  # 多分类推荐使用Softmax

    def forward(self, x):
        # 输入特征图尺寸：(batch, channels, H, W)
        x = self.global_avg_pool(x)         # → (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)           # 展平成 (batch, channels)
        x = self.fc(x)                      # → (batch, num_classes)
        x = self.activation(x)
        return x


# 权重加权融合（可学习的特征融合）
class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels , in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # 动态权重融合
        weight = self.conv(torch.cat([x, y], dim=1))
        # print(weight.shape)
        return x * weight + y * (1 - weight)


# 定义U-Net 网络
class UNet(nn.Module):
    def __init__(self,num_class,n_bandas):
        super(UNet, self).__init__()

        # 编码器部分
        self.enc1 = ConvBlock(n_bandas, 256)
        self.enc2 = ConvBlock(256, 512)
        self.enc3 = ConvBlock(512, 1024)
        self.enc4 = ConvBlock(1024, 2048)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 中间部分 （改为光谱提取模块）
        self.spectral_features1 = SpectralFeatureExtractor(16,16)
        self.spectral_features2 = SpectralFeatureExtractor(8,8)
        self.spectral_features3 = SpectralFeatureExtractor(4,4)
        self.spectral_features4 = SpectralFeatureExtractor(2,2)


        # 局部融合
        self.feature_fusion1 = FeatureFusion(in_channels=256)
        self.feature_fusion2 = FeatureFusion(in_channels=512)
        self.feature_fusion3 = FeatureFusion(in_channels=1024)
        self.feature_fusion4 = FeatureFusion(in_channels=2048)

        # 特征融合
        self.ff1 = FreqFusion(hr_channels=1024, lr_channels=1024)
        self.ff2 = FreqFusion(hr_channels=512, lr_channels=512)
        self.ff3 = FreqFusion(hr_channels=256, lr_channels=256)

        # 融合
        self.conv1x1_4 = conv1x1(2048, 1024)
        self.conv1 = conv(1024, 512)
        self.conv2 = conv(512, 256)
        self.conv3 = conv(256, 128)


        # 输出部分
        self.head = OutputBlock(in_channels=128, num_classes=num_class)


    def forward(self, x):
        pad_width = (0, 1, 0, 1, 0, 0, 0, 0)
        x = F.pad(x, pad_width, value=0)

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        spectral_features1 = self.spectral_features1(enc1)
        spectral_features2 = self.spectral_features2(enc2)
        spectral_features3 = self.spectral_features3(enc3)
        spectral_features4 = self.spectral_features4(enc4)

        fused_features1 = self.feature_fusion1(enc1, spectral_features1)
        fused_features2 = self.feature_fusion2(enc2, spectral_features2)
        fused_features3 = self.feature_fusion3(enc3, spectral_features3)
        fused_features4 = self.feature_fusion4(enc4, spectral_features4)


        fused_features4= self.conv1x1_4(fused_features4)
        dec4 = fused_features4
        _, fused_features3, dec4_up = self.ff1(hr_feat=fused_features3, lr_feat=dec4)
        dec3 = self.conv1(torch.cat([fused_features3 + dec4_up]))
        _, fused_features2, dec3_up = self.ff2(hr_feat=fused_features2, lr_feat=dec3)
        dec2 = self.conv2(torch.cat([fused_features2 + dec3_up]))
        _, fused_features1, dec2_up = self.ff3(hr_feat=fused_features1, lr_feat=dec2)
        dec1 = self.conv3(torch.cat([fused_features1 + dec2_up]))


        out = self.head(dec1)
        return out




