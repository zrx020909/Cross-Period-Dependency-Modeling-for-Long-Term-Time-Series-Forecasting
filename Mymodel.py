import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.Conv_Blocks import EfficientInception
from layers.Conv_Blocks import DSConv
#from layers.SelfAttention_Family import TemporalFrequencyAttention
import math
from layers.lightcov import LightInception
from models.Freq_Conv import SpectralBranch
from models.CPIB import CrossPeriodAttention
from models.AdaFFT import LinearAdaptive


# ===== FFT 提取候选周期 =====
def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)  # [B, F, C]
    freq_energy = xf.abs().mean(0).mean(-1)  # [F]
    freq_energy[0] = 0
    _, top_list = torch.topk(freq_energy, k)
    top_list = top_list.detach().cpu().numpy()

    period_list = []
    for idx in top_list:
        if idx == 0:
            p = x.shape[1]
        else:
            p = max(2, x.shape[1] // idx)
        period_list.append(p)

    raw_weight = xf.abs().mean(-1)[:, top_list]  # [B, k]
    return period_list, raw_weight



class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        self.N = configs.enc_in

        self.conv = nn.Sequential(
            LightInception(
                configs.d_model,
                configs.d_ff
            ),
            nn.GELU(),
            LightInception(
                configs.d_ff,
                configs.d_model
            )
        )

        self.adaptive_weight = LinearAdaptive(self.k)
        self.cross_attn = CrossPeriodAttention(configs.d_model, configs.top_k)

        # 新增频域分支
        self.freq_branch = SpectralBranch(d_model=configs.d_model)

        self.gate = nn.Sequential(
            nn.Linear(2 * configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Linear(configs.d_model // 2, 1)
        )



    def forward(self, x):
        B, T, N = x.size()

        # 1. 周期识别
        period_list, raw_weight = FFT_for_Period(x, self.k)

        # 2. 学习到的周期权重
        period_weight = self.adaptive_weight(raw_weight)  # [B, k]

        res = []

        for i in range(self.k):
            period = period_list[i]
            total_len = self.seq_len + self.pred_len

            # 判断是否整除
            if total_len % period != 0:
                # 向上取整，使长度可整除 period
                length = ((total_len // period) + 1) * period
                pad_len = length - total_len

                # ------ 镜像周期填充（Mirror Periodic Padding）------
                # x: [B, L, C]
                mirror_source = x[:, :-1, :]  # 去掉最后一个点避免重复 [B, L-1, C]

                # 反转（从后往前镜像）
                mirror_source = torch.flip(mirror_source, dims=[1])

                # 取前 pad_len 个时间步
                mirror_padding = mirror_source[:, :pad_len, :]

                # 拼接
                out = torch.cat([x, mirror_padding], dim=1)

            else:
                length = total_len
                out = x

            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2)  # .contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)

        # ⚡ 跨周期注意力 + FFT 聚合
        time_out = self.cross_attn(res, period_weight)

        # 频域分支
        freq_out = self.freq_branch(x)  # [B, L, N]

        # 动态时间-频率门控融合
        gate_input = torch.cat([time_out, freq_out], dim=-1)
        G = torch.sigmoid(self.gate(gate_input))

        out = G * time_out + (1 - G) * freq_out

        total_out = out + x

        return total_out


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ 模型初始化完成，总参数量: {total_params / 1e6:.2f}M")
        print(f"可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
