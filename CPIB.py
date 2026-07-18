import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from models.AdaFFT import LinearAdaptive


# ==================== FFT 选周期 ====================
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = xf.abs().mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    periods = (x.shape[1] // top_list.cpu()).tolist()
    return periods, xf.abs().mean(-1)[:, top_list]  # [k], [B,k]

# ==================== 跨周期注意力模块 ====================
class CrossPeriodAttention(nn.Module):
    def __init__(self, d_model, k, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.k = k
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, res, period_weight):
        # res: [B, L, N, K], period_weight: [B, K]
        B, L, N, K = res.shape
        device = res.device

        # normalize and reshape
        x = res.permute(0, 1, 3, 2).reshape(B * L, K, N)     # [B*L, K, N]
        x = self.norm(x)

        # linear proj
        Q = self.q_proj(x)   # [B*L, K, d_model]
        K_ = self.k_proj(x)
        V = self.v_proj(x)

        # split heads
        def split_heads(t):
            # t: [B*L, K, d_model] -> [B*L, num_heads, K, head_dim]
            return t.view(B * L, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Qh = split_heads(Q)
        Kh = split_heads(K_)
        Vh = split_heads(V)

        # scaled dot-product
        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / self.scale   # [B*L, heads, K, K]
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, Vh)   # [B*L, heads, K, head_dim]

        # combine heads
        context = context.permute(0, 2, 1, 3).reshape(B * L, K, self.d_model)  # [B*L, K, d_model]
        out = self.o_proj(context)
        out = self.out_dropout(out)

        # residual
        out = out + x   # [B*L, K, d_model]
        out = out.view(B, L, K, N).permute(0, 1, 3, 2)   # [B, L, N, K]

        # period weighting (use softmax here)
        w = F.softmax(period_weight, dim=-1).to(device)    # [B, K]
        w = w.view(B, 1, 1, K)
        fused = (out * w).sum(-1)   # [B, L, N]
        return fused

# ==================== TimesBlock ====================
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        self.cross_attn = CrossPeriodAttention(configs.d_model, configs.top_k)
        self.adaptive_weight = LinearAdaptive(self.k)

    def forward(self, x):
        B, T, N = x.size()
        #period_list, period_weight = FFT_for_Period(x, self.k)
        #res = []
        # 1. 周期识别
        #period_list, raw_weight = FFT_for_Period(x, self.k)
        period_list, period_weight = FFT_for_Period(x, self.k)

        # 2. 学习到的周期权重
        #period_weight = self.adaptive_weight(raw_weight)  # [B, k]
        res = []
        for i in range(self.k):
            period = period_list[i]
            """
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            """
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                pad_len = length - (self.seq_len + self.pred_len)
                padding = x[:, :pad_len, :]  # 用前面的序列循环补齐
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)  # [B,L,N,K]

        # ⚡ 跨周期注意力 + FFT 聚合
        res = self.cross_attn(res, period_weight)

        return res + x  # 残差连接

# ==================== Model ====================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None