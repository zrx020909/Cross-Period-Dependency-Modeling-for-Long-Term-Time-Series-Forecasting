import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from typing import List, Optional


class DynamicPeriodDetection(nn.Module):
    """
    自适应周期检测模块：
    结合自相关和小波变换动态选择Top-K个显著周期。
    """

    def __init__(self, max_period: int = 100, num_periods: int = 3):
        super().__init__()
        self.max_period = max_period
        self.num_periods = num_periods
        # 用于动态权重学习的轻量网络
        self.period_weights = nn.Sequential(
            nn.Linear(max_period, 64),
            nn.ReLU(),
            nn.Linear(64, num_periods),
            nn.Softmax(dim=-1)
        )


    def autocorrelation(self, x: torch.Tensor) -> torch.Tensor:
        """ 自相关分析 (Batch-wise) """
        batch_size, length = x.shape[0], x.shape[1]
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        x_pad = F.pad(x, (0, self.max_period), "constant", 0)
        corr = F.conv1d(
            x_pad.unsqueeze(1),
            x.unsqueeze(1).flip(dims=[-1]),
            padding=self.max_period
        )[:, 0, :self.max_period]
        return corr / torch.arange(length, length - self.max_period, -1, device=x.device)

    def wavelet_transform(self, x: torch.Tensor) -> torch.Tensor:
        """ 连续小波变换 (PyTorch实现需自定义或调用外部库) """
        # 注：实际部署需替换为PyTorch兼容的小波变换（如使用pytorch_wavelets）
        scales = np.arange(1, self.max_period + 1)
        batch_cwt = []
        for i in range(x.shape[0]):
            coeffs, _ = pywt.cwt(x[i].cpu().numpy(), scales, 'morl')
            batch_cwt.append(torch.from_numpy(np.abs(coeffs).sum(axis=1)))
        return torch.stack(batch_cwt, dim=0).to(x.device)

    def forward(self, x: torch.Tensor) -> (List[int], torch.Tensor):
        """
        返回:
            - periods: 选择的周期长度列表
            - scores: 各周期的权重得分 [Batch, num_periods]
        """
        # 并行计算自相关和小波能量
        autocorr = self.autocorrelation(x)  # [Batch, max_period]
        wavelet_energy = self.wavelet_transform(x)  # [Batch, max_period]

        # 融合两种检测方法（可学习加权）
        combined = autocorr + wavelet_energy  # 简单相加，也可用门控机制
        scores = self.period_weights(combined)  # [Batch, num_periods]

        # 选择Top-K个周期（基于得分）
        _, topk_indices = torch.topk(scores.mean(dim=0), self.num_periods)
        periods = (topk_indices + 1).tolist()  # 周期长度 >=1

        return periods, scores


class AdaptiveTimesBlock(nn.Module):
    """ 改进的TimesNet时域分支块（仅含自适应周期检测） """

    def __init__(self, in_dim: int, out_dim: int, max_period: int = 100):
        super().__init__()
        self.period_detector = DynamicPeriodDetection(max_period)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        self.out_conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)

    def reshape_to_2d(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """ 将时序数据按周期reshape为2D张量 """
        batch, length, dim = x.shape
        if length % period != 0:
            pad_len = period - (length % period)
            x = F.pad(x, (0, 0, 0, pad_len), "constant", 0)
        return x.reshape(batch, period, -1, dim).permute(0, 3, 1, 2)  # [B, D, P, L/P]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 动态检测周期
        periods, scores = self.period_detector(x)  # periods: List[int], scores: [B, K]

        # 2. 多周期处理
        period_features = []
        for i, p in enumerate(periods):
            # 2D卷积处理
            x_2d = self.reshape_to_2d(x, p)  # [B, D, P, L/P]
            conv_out = self.conv2d(x_2d)  # [B, D, P, L/P]

            # 还原为1D并加权
            conv_1d = conv_out.permute(0, 2, 3, 1).reshape(x.shape[0], -1, conv_out.shape[1])
            weighted = scores[:, i].unsqueeze(-1) * conv_1d  # [B, L, D]
            period_features.append(weighted)

        # 3. 融合多周期特征
        output = sum(period_features)  # [B, L, D]
        output = self.out_conv(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output

