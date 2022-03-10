# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module): 
    def __init__(self, d_model, len_max=4096): 
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(len_max, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, len_max).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module): 
    def __init__(self, c_in, d_model): 
        super().__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular'
        )
        for m in self.modules(): 
            if isinstance(m, nn.Conv1d): 
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )

    def forward(self, x): 
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x


class TemporalEmbedding(nn.Module): 
    def __init__(self, d_model, freq='h'): 
        super().__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x): 
        return self.embed(x)


class DataEmbedding(nn.Module): 
    def __init__(self, c_in, d_model, freq='h', dropout=0.1): 
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark): 
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class FFN(nn.Module): 
    def __init__(self, d_model, d_ff, dropout=0.1): 
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x): 
        """
        Input
        ----------
        x
            Shape (B, len_seq, d_model)
        
        Output
        ----------
        out
            Shape (B, len_seq, d_model)
        """
        out = self.conv1(x.transpose(-1, 1))
        out = self.dropout(self.activation(out))
        out = self.dropout(self.conv2(out)).transpose(-1, 1)
        out = self.norm(x + out)
        return out


class EncoderLayer(nn.Module): 
    def __init__(self, self_attn, d_model, d_ff, dropout=0.1, output_attn=False): 
        super().__init__()
        self.self_attn = self_attn
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_attn = output_attn

    def forward(self, x, self_attn_mask=None): 
        """
        Input
        ----------
        x
            Shape (B, len_enc, d_model)
        self_attn_mask
            None or Shape (len_enc, len_enc)

        Output
        ----------
        out
            Shape (B, len_enc, d_model)
        (self_attn_weight, None)
            Shape (B, ln_heads, en_enc, len_enc)
        """
        # Self-attention
        x = x.transpose(0, 1)
        out, self_attn_weight = self.self_attn(
            x, x, x, 
            need_weights=self.output_attn, 
            attn_mask=self_attn_mask
        )
        x = self.norm1(x + self.dropout(out))
        # FFN
        x = x.transpose(0, 1)
        out = self.ffn(x)
        return out, (self_attn_weight, None)


class DecoderLayer(nn.Module): 
    def __init__(self, self_attn, cross_attn, d_model, d_ff, dropout=0.1, output_attn=False): 
        super().__init__()
        self.self_attn = self_attn
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = cross_attn
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_attn = output_attn

    def forward(self, x, enc_out, self_attn_mask=None, cross_attn_mask=None): 
        """
        Input
        ----------
        x
            Shape (B, len_label+len_pred, d_model)
        enc_out
            Shape (B, len_enc, d_model)
        self_attn_mask
            None or Shape (len_label+len_pred, len_label+len_pred)
        cross_attn_mask
            None or Shape (len_label+len_pred, len_enc)

        Output
        ----------
        out
            Shape (B, len_label+len_pred, d_model)
        (self_attn_weight, cross_attn_weight)
            self_attn_weight is of Shape (B, n_heads, len_label+len_pred, len_label+len_pred)
            cross_attn_weight is of Shape (B, n_heads, len_label+len_pred, len_enc)
        """
        # Self-attention
        x = x.transpose(0, 1)
        out, self_attn_weight = self.self_attn(
            x, x, x, 
            need_weights=self.output_attn, 
            attn_mask=self_attn_mask
        )
        x = self.norm1(x + self.dropout(out))
        # Cross-attention
        enc_out = enc_out.transpose(0, 1)
        out, cross_attn_weight = self.cross_attn(
            x, enc_out, enc_out, 
            need_weights=self.output_attn, 
            attn_mask=cross_attn_mask
        )
        x = self.norm2(x + self.dropout(out))
        # FFN
        x = x.transpose(0, 1)
        out = self.ffn(x)
        return out, (self_attn_weight, cross_attn_weight)


def get_triangular_causal_mask(x): 
    # x: Shape (B, len_seq, d_model)
    device = x.device
    len_seq = x.shape[1]
    return torch.triu(
        torch.ones((len_seq, len_seq)), diagonal=1
    ).to(device, dtype=torch.bool)
