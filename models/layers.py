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
        out = self.conv2(out).transpose(-1, 1)
        out = self.norm(x + self.dropout(out))
        return out


class MultiheadAttention(nn.Module): 
    def __init__(self, d_model, n_heads=8, dropout=0.1): 
        super().__init__()
        assert d_model % n_heads == 0
        d_qkv = d_model // n_heads
        d_inner = d_qkv * n_heads
        self.scale = d_model ** (-0.5)
        self.Q_proj = nn.Linear(d_model, d_inner)
        self.K_proj = nn.Linear(d_model, d_inner)
        self.V_proj = nn.Linear(d_model, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        # (len, B, d_inner) -> (len, B, n_heads, d_qkv)
        self.reshape_for_attn = lambda x: x.reshape(*x.shape[:-1], n_heads, d_qkv)
        # (len, B, n_heads, d_qkv) -> (len, B, d_inner)
        self.recover_from_attn = lambda x: x.reshape(*x.shape[:-2], d_inner)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.disable_inc() # Incremental decoding is disabled by default

    def multihead_attention(self, Q, K, V, attn_mask=None): 
        attn_scores = torch.einsum('ibnd, jbnd -> ijbn', (Q, K)) * self.scale
        if attn_mask is not None: 
            assert attn_mask.dtype == torch.bool, 'Only bool type is supported for masks.'
            assert attn_mask.ndim == 2, 'Only 2D attention mask is supported'
            assert attn_mask.shape == attn_scores.shape[:2], 'Incorrect mask shape: {}. Expect: {}'.format(attn_mask.shape, attn_scores.shape[:2])
            attn_mask = attn_mask.view(*attn_mask.shape, 1, 1)
            attn_scores.masked_fill_(attn_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        out = torch.einsum('ijbn, jbnd -> ibnd', (attn_weights, V))
        out = self.recover_from_attn(out)
        attn_weights = attn_weights.permute(2, 3, 0, 1)
        return out, attn_weights

    def forward(self, query, key, value, need_weights=False, attn_mask=None): 
        """
        Input
        ----------
        query
            Shape (L, B, d_model)
        key
            Shape (S, B, d_model)
        value
            Shape (S, B, d_model)
        attn_mask
            None or Shape (L, S) with type torch.bool

        Output
        ----------
        out
            Shape (S, B, d_model)
        attn
            None or Shape (B, n_heads, L, S)
        """
        if self.incremental_decoding: 
            # sanity check
            assert hasattr(self, 'K') and hasattr(self, 'V')
            assert attn_mask is None # automatically causal
            return self.inc_decode(query, key, value, need_weights=need_weights)
        # in-sample projection
        Q, K, V = (
            self.reshape_for_attn(self.Q_proj(query)), 
            self.reshape_for_attn(self.K_proj(key)), 
            self.reshape_for_attn(self.V_proj(value))
        )
        # attention mechanism
        out, attn = self.multihead_attention(Q, K, V, attn_mask=attn_mask)
        # output layer
        out = self.out_proj(out)
        return out, attn if need_weights else None

    @property
    def incremental_decoding(self): 
        return self._inc

    def enable_inc(self): 
        self._inc = True

    def disable_inc(self): 
        self._inc = False

    def reset_inc(self): 
        """Clear cached computed keys and values"""
        self.K = None # Shape (S, B, n_heads, d_qkv)
        self.V = None # Shape (S, B, n_heads, d_qkv)

    def inc_decode(self, new_query, new_key, new_value, need_weights=False, attn_mask=None): 
        """
        Input
        ----------
        new_query
            Shape (l, B, d_model)
        new_key
            Shape (l, B, d_model)
        new_value
            Shape (l, B, d_model)
        attn_mask
            None by assumption

        Output
        ----------
        out
            Shape (l, B, d_model)
        attn
            None or Shape (l, S, B, n_heads)
        """
        # projection
        q, k, v = (
            self.reshape_for_attn(self.Q_proj(new_query)), 
            self.reshape_for_attn(self.K_proj(new_key)), 
            self.reshape_for_attn(self.V_proj(new_value))
        )
        # concatenation
        self.K = k if self.K is None else torch.cat((self.K, k), dim=0).contiguous()
        self.V = v if self.V is None else torch.cat((self.V, v), dim=0).contiguous()
        # one-step decoding
        out, attn = self.multihead_attention(q, self.K, self.V, attn_mask=attn_mask)
        # output
        out = self.out_proj(out)
        return out, attn if need_weights else None


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
            attn_mask=None if self_attn_mask is None else self_attn_mask.squeeze()
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
            attn_mask=None if self_attn_mask is None else self_attn_mask.squeeze()
        )
        x = self.norm1(x + self.dropout(out))
        # Cross-attention
        enc_out = enc_out.transpose(0, 1)
        out, cross_attn_weight = self.cross_attn(
            x, enc_out, enc_out, 
            need_weights=self.output_attn, 
            attn_mask=None if cross_attn_mask is None else cross_attn_mask.squeeze()
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
        torch.ones((1, 1, len_seq, len_seq)), diagonal=1
    ).to(device, dtype=torch.bool)
