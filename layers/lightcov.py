import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthSpectra(nn.Module):
    """
    def __init__(self, in_channels, out_channels, num_kernels=3, reduction_ratio=4):
        super().__init__()
        # 1. Bottleneck 通道压缩
        self.bottleneck = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        # 2. 分组卷积的多分支设计
        self.kernels = nn.ModuleList([
            nn.Conv2d(
                in_channels // reduction_ratio,
                out_channels,
                kernel_size=2 * i + 1,  # 1x1, 3x3, 5x5...
                padding=i,  # 保持输出尺寸不变
                groups=in_channels // reduction_ratio  # 分组卷积（深度可分离卷积的组数）
            )
            for i in range(num_kernels)
        ])
        # 3. 可选的后续融合层（提升特征整合能力）
        self.fusion = nn.Conv2d(out_channels * num_kernels, out_channels, kernel_size=1)
    def forward(self, x):
        # Step 1: 通道压缩
        x = self.bottleneck(x)  # [B, C//4, H, W]
        # Step 2: 多分支分组卷积
        branch_outputs = []
        for kernel in self.kernels:
            branch_outputs.append(kernel(x))  # 每个分支输出 [B, C_out, H, W]
        # Step 3: 融合分支（替代原版的均值融合）
        fused = self.fusion(torch.cat(branch_outputs, dim=1))  # [B, C_out, H, W]
        return fused
    """

    def __init__(self, in_channels, out_channels, num_kernels=3, reduction_ratio=4):
        super().__init__()
        self.reduction_ratio = reduction_ratio

        # 1️⃣ 通道压缩
        mid_channels = max(1, in_channels // reduction_ratio)
        self.bottleneck = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        # 2️⃣ 多尺度深度卷积分支
        self.kernels = nn.ModuleList([
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=2 * i + 1,
                padding=i,
                groups=mid_channels,  # 深度可分离卷积
                bias=False
            )
            for i in range(num_kernels)
        ])

        # 3️⃣ 动态尺度权重（自适应选择不同时间尺度）
        self.scale_weights = nn.Parameter(torch.ones(num_kernels))

        # 4️⃣ 融合层
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 5️⃣ 通道注意力 (SE block)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 6️⃣ 残差映射
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        res = self.res_conv(x)

        # Step 1: Bottleneck 压缩
        x = self.bottleneck(x)

        # Step 2: 多尺度卷积分支
        branch_outputs = [conv(x) for conv in self.kernels]

        # Step 3: 动态融合不同时间尺度
        weights = F.softmax(self.scale_weights, dim=0)
        fused = sum(w * b for w, b in zip(weights, branch_outputs))

        # Step 4: 1×1 融合 + 通道注意力
        fused = self.fusion(fused)
        fused = self.norm(fused) #+
        fused = fused * self.channel_att(fused)
        fused = self.dropout(fused)
        # Step 5: 残差连接
        out = fused + res
        return out
