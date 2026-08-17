import torch
import torch.nn as nn


class LightInception(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super().__init__()

        # 1. Bottleneck: reduce channel dimension
        hidden_channels = max(1, in_channels // reduction_ratio)

        self.bottleneck = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=False
        )

        # ==========================================================
        # 2. Three parallel multi-scale dilated convolution branches
        # ==========================================================

        # Branch 1: 1x1 convolution
        # Local feature projection without expanding receptive field
        self.branch1 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            dilation=1,
            bias=False
        )

        # Branch 2: 3x3 convolution, dilation = 1
        # Capture local intra-period patterns
        self.branch2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False
        )

        # Branch 3: 3x3 convolution, dilation = 2
        # Capture broader inter-period dependencies
        self.branch3 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False
        )

        # ==========================================================
        # 3. Feature fusion
        # ==========================================================
        self.fusion = nn.Conv2d(
            out_channels * 3,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.activation = nn.GELU()

    def forward(self, x):

        # Step 1: Bottleneck
        x = self.bottleneck(x)

        # Step 2: Three parallel branches
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # Step 3: Concatenate multi-scale features
        out = torch.cat([x1, x2, x3], dim=1)

        # Step 4: Fuse the multi-scale representations
        out = self.fusion(out)

        # Step 5: Nonlinear activation
        out = self.activation(out)

        return out