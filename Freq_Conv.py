import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # 按幅值找候选频率
    freq_energy = abs(xf).mean(0).mean(-1)
    freq_energy[0] = 0
    _, top_list = torch.topk(freq_energy, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: [*, d_model]
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.weight * x_norm

class SpectralBranch(nn.Module):

    def __init__(self, d_model, hidden=64, kernel_size=9, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale

        # 频域滤波器：Conv1d + ReLU + Conv1d
        self.mask_net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden, d_model, kernel_size=1)
        )

        # 输出归一化
        #self.norm = nn.LayerNorm(d_model)
        self.norm = RMSNorm(d_model)

        # 动态门控：根据输入序列的统计信息（均值+方差）预测 α
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, T, C = x.shape
        # FFT
        Xf = torch.fft.rfft(x, dim=1)  # [B, F, C]
        real, imag = Xf.real, Xf.imag
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-6)  # [B, F, C]

        # 频域滤波 mask
        mag_ch_first = mag.permute(0, 2, 1)  # [B, C, F]
        mask = self.mask_net(mag_ch_first)  # [B, C, F]
        mask = torch.sigmoid(mask).permute(0, 2, 1).contiguous()  # [B, F, C]

        real_new, imag_new = real * mask, imag * mask
        Yf = torch.complex(real_new, imag_new)

        # ====== 频域门控 α (基于频谱统计) ======
        mean_feat = mag.mean(1)  # [B, C]
        var_feat = mag.var(1)  # [B, C]
        stat_feat = torch.cat([mean_feat, var_feat], dim=-1)  # [B, 2C]
        alpha = self.gate_fc(stat_feat).unsqueeze(1).expand(-1, Xf.size(1), -1)  # [B, F, C]

        # 频域融合
        #Zf = alpha * Yf + (1 - alpha) * Xf
        Zf = Xf + 0.5 * Yf  # 原始频谱Xf直接与滤波频谱Yf相加（0.5为固定缩放）
        z = torch.fft.irfft(Zf, n=T, dim=1)  # [B, T, C]

        return self.residual_scale * self.norm(z)


class TimesBlock(nn.Module):
    """
    改进版 TimesBlock：
    - 时域 Inception 2D 卷积分支
    - 频域卷积分支
    - 融合（可学习门控 α）
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model

        # 原始 2D Inception 卷积
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

        # 新增频域分支
        self.freq_branch = SpectralBranch(d_model=configs.d_model)

        # 分支融合门控（初始化偏向时域）
        self.gate = nn.Parameter(torch.tensor(0.9))
        # 时-频融合门控（基于时域统计）
        """
        self.gate_fc = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Sigmoid()
        )
        """

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([B, (length - (self.seq_len + self.pred_len)), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape -> 2D conv
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)  # [B, L, N, K]

        # 自适应聚合 (FFT 权重)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        time_out = torch.sum(res * period_weight, -1)  # [B, L, N]

        # 频域分支
        freq_out = self.freq_branch(x)  # [B, L, N]

        # 融合 (门控 α)
        alpha = torch.sigmoid(self.gate)

        out = alpha * time_out + (1 - alpha) * freq_out

        # 残差
        out = out + x
        return out


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
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
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
        # TimesNet
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
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
