import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # 初始化时直接处理 alpha
        if self.alpha is not None:
            # 将 alpha 转换为张量并分配到指定设备
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device='cuda')
        else:
            self.alpha = None  # 默认不使用类别权重

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C) - 未归一化的 logits，C 是类别数
            labels: (B,) - 整数标签（0-based）
        Returns:
            loss: 标量或 (B,) 张量
        """
        # 提取正确类别的概率 p_t
        # 沿着由dim指定的轴收集数值。
        logits = logits.to('cuda')
        labels = labels.to('cuda')
        pt = logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # probs, shape=(bs, 1)

        # 自定义交叉熵损失（避免 log(0)）
        pt = torch.clamp(pt, min=1e-10, max=1.0)
        BCE_loss = -torch.log(pt)

        # Focal Loss 公式
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        # 处理 alpha 权重
        if self.alpha is not None:
            alpha = self.alpha[labels]
            focal_loss = alpha * focal_loss

        # 返回损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.squeeze(1)  # 返回 (B,) 张量