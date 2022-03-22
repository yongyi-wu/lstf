# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_triangular_causal_mask(x): 
    """
    Input
    ----------
    x
        Shape (B, len, d)

    Output
    ----------
    mask
        Shape (1, 1, len, len) with upper triangle filled with True
    """
    device = x.device
    len_seq = x.shape[1]
    mask = torch.triu(
        torch.ones((1, 1, len_seq, len_seq)), diagonal=1
    ).to(device, dtype=torch.bool)
    return mask


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
    def __init__(self, c_in, d_model, bias=True): 
        super().__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular', 
            bias=bias
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
    def __init__(self, d_model, bias=True, freq='h'): 
        super().__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=bias)
    
    def forward(self, x): 
        return self.embed(x)


class DataEmbedding(nn.Module): 
    def __init__(self, c_in, d_model, pos=True, bias=True, freq='h', dropout=0.1): 
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model, bias=bias)
        self.pos = pos
        if self.pos: 
            self.position_embedding = PositionalEmbedding(d_model)
        else: 
            self.position_embedding = None
        self.temporal_embedding = TemporalEmbedding(d_model, bias=bias, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark): 
        x = (
            self.value_embedding(x) + \
            (self.position_embedding(x) if self.pos else 0) + \
            self.temporal_embedding(x_mark)
        )
        return self.dropout(x)


class FFN(nn.Module): 
    def __init__(self, d_model, d_ff, bias=True, dropout=0.1): 
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=bias)
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
        x
            Shape (B, len_seq, d_model)
        """
        x = self.conv1(x.transpose(-1, 1))
        x = self.dropout(self.activation(x))
        x = self.conv2(x).transpose(-1, 1)
        return x


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
        self.reshape_for_attn = lambda x: x.reshape(*x.shape[:-1], n_heads, d_qkv).contiguous()
        # (len, B, n_heads, d_qkv) -> (len, B, d_inner)
        self.recover_from_attn = lambda x: x.reshape(*x.shape[:-2], d_inner).contiguous()
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
            Shape (L, B, d_model)
        attn
            None or Shape (B, n_heads, L, S)
        """
        if self.incremental_decoding: 
            # Sanity check
            assert hasattr(self, 'K') and hasattr(self, 'V')
            assert attn_mask is None # automatically causal
            return self.inc_decode(query, key, value, need_weights=need_weights)
        # In-sample projection
        Q, K, V = (
            self.reshape_for_attn(self.Q_proj(query)), 
            self.reshape_for_attn(self.K_proj(key)), 
            self.reshape_for_attn(self.V_proj(value))
        )
        # Attention mechanism
        out, attn = self.multihead_attention(Q, K, V, attn_mask=attn_mask)
        # Output layer
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
        self.norm2 = nn.LayerNorm(d_model)
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
        x
            Shape (B, len_enc, d_model)
        (self_attn_weight, None)
            Shape (B, n_heads, len_enc, len_enc)
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
        x = self.norm2(x + self.dropout(out))
        return x, (self_attn_weight, None)


class DecoderLayer(nn.Module): 
    def __init__(self, self_attn, cross_attn, d_model, d_ff, dropout=0.1, output_attn=False): 
        super().__init__()
        self.self_attn = self_attn
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = cross_attn
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
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
        x
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
        x = self.norm3(x + self.dropout(out))
        return x, (self_attn_weight, cross_attn_weight)


class SeriesDecomposition(nn.Module): 
    def __init__(self, len_window): 
        super().__init__()
        self.avgpool = nn.AvgPool1d(
            kernel_size=len_window, 
            stride=1, 
            padding=(len_window - 1) // 2, 
            count_include_pad=False
        )

    def forward(self, x): 
        """
        Input
        ----------
        x
            Shape (B, len, d)
        
        Output
        ----------
        x_s
            Seasonality, Shape (B, len, d)
        x_t
            Trend, Shape (B, len, d)
        """
        x_t = self.avgpool(x.transpose(-1, -2)).transpose(-1, -2)
        x_s = x - x_t
        return x_s, x_t


class MultiheadAutoCorrelation(nn.Module): 
    def __init__(self, d_model, n_heads=8, c_sampling=1): 
        super().__init__()
        assert d_model % n_heads == 0
        d_qkv = d_model // n_heads
        d_inner = d_qkv * n_heads
        self.Q_proj = nn.Linear(d_model, d_inner)
        self.K_proj = nn.Linear(d_model, d_inner)
        self.V_proj = nn.Linear(d_model, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        # (len, B, d_inner) -> (B, n_heads, d_qkv, len)
        self.reshape_for_attn = lambda x: x.reshape(*x.shape[:-1], n_heads, d_qkv).permute(1, 2, 3, 0).contiguous()
        # (B, n_heads, d_qkv, len) -> (len, B, d_inner)
        self.recover_from_attn = lambda x: x.permute(3, 0, 1, 2).reshape(x.shape[3], x.shape[0], d_inner).contiguous()
        self.n_heads = n_heads
        self.c_sampling = c_sampling

    def time_delay_agg(self, V, corr_scores): 
        """
        Input
        ----------
        V
            Shape (B, n_heads, d_qkv, S)
        corr_scores
            Shape (B, n_heads, d_qkv, S)

        Output
        ----------
        agg_out
            Shape (B, n_heads, d_qkv, S)
        """
        B, H, E, S = V.shape
        index = torch.arange(S).reshape(1, 1, 1, S, 1).to(V.device) # (B, n_heads, d_qkv, S, topk)
        # Find topk
        topk = int(self.c_sampling * math.log(S))
        corr_scores, taus = torch.topk(torch.mean(corr_scores, dim=-2), topk, dim=-1) # Shape (B, n_heads, topk)
        corr_weights = torch.softmax(corr_scores, dim=-1).reshape(B, H, 1, 1, topk)
        # Aggregation
        delayed_index = (index + taus.reshape(B, H, 1, 1, topk)) % S
        delayed_index = delayed_index.expand(B, H, E, S, topk)
        V = V.unsqueeze(-1).expand(B, H, E, S, topk)
        delayed_out = torch.gather(V, dim=-2, index=delayed_index)
        agg_out = torch.mean(delayed_out * corr_weights, dim=-1)
        return agg_out

    def multihead_autocorrelation(self, Q, K, V): 
        # Padding or truncation of length
        L, S = Q.shape[-1], V.shape[-1]
        if L > S: 
            zeros = torch.zeros_like(Q[..., :(L - S)])
            V = torch.cat([V, zeros], dim=-1)
            K = torch.cat([K, zeros], dim=-1)
        else: 
            V = V[..., :L]
            K = K[..., :L]
        # Period-based dependencies
        Q_fft = torch.fft.rfft(Q, dim=-1)
        K_fft = torch.fft.rfft(K, dim=-1)
        res = Q_fft * torch.conj(K_fft)
        corr_scores = torch.fft.irfft(res, dim=-1)
        # Time delay aggregation
        out = self.time_delay_agg(V, corr_scores) 
        out = self.recover_from_attn(out)
        return out, corr_scores

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
            None by assumption

        Output
        ----------
        out
            Shape (L, B, d_model)
        corr
            None or Shape (B, n_heads, d_qkv, L)
        """
        assert attn_mask is None
        # In-sample projection
        Q, K, V = (
            self.reshape_for_attn(self.Q_proj(query)), 
            self.reshape_for_attn(self.K_proj(key)), 
            self.reshape_for_attn(self.V_proj(value))
        )
        # Autocorrelation mechanism
        out, corr = self.multihead_autocorrelation(Q, K, V)
        # Output layer
        out = self.out_proj(out)
        return out, corr if need_weights else None


class AutoformerEncoderLayer(nn.Module): 
    def __init__(self, self_attn, d_model, d_ff, len_window=25, dropout=0.1, output_attn=False): 
        super().__init__()
        self.self_attn = self_attn
        self.decomp1 = SeriesDecomposition(len_window)
        self.ffn = FFN(d_model, d_ff, bias=False, dropout=dropout)
        self.decomp2 = SeriesDecomposition(len_window)
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
        x
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
        # Decomposition
        x, _ = self.decomp1((x + self.dropout(out)).transpose(0, 1))
        # FFN
        out = self.ffn(x)
        # Decomposition
        x, _ = self.decomp2(x + self.dropout(out))
        return x, (self_attn_weight, None)


class AutoformerDecoderLayer(nn.Module): 
    def __init__(self, self_attn, cross_attn, d_model, d_ff, d_dec_out, len_window=25, dropout=0.1, output_attn=False): 
        super().__init__()
        self.self_attn = self_attn
        self.decomp1 = SeriesDecomposition(len_window)
        self.cross_attn = cross_attn
        self.decomp2 = SeriesDecomposition(len_window)
        self.ffn = FFN(d_model, d_ff, bias=False, dropout=dropout)
        self.decomp3 = SeriesDecomposition(len_window)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_dec_out, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            padding_mode='circular', 
            bias=False
        )
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
        x
            Shape (B, len_label+len_pred, d_model)
        trend
            Shape (B, len_label+len_pred, d_dec_out)
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
        # Decomposition
        x, trend1 = self.decomp1((x + self.dropout(out)).transpose(0, 1))
        # Cross-attention
        x = x.transpose(0, 1)
        enc_out = enc_out.transpose(0, 1)
        out, cross_attn_weight = self.cross_attn(
            x, enc_out, enc_out, 
            need_weights=self.output_attn, 
            attn_mask=None if cross_attn_mask is None else cross_attn_mask.squeeze()
        )
        # Decomposition
        x, trend2 = self.decomp2((x + self.dropout(out)).transpose(0, 1))
        # FFN
        out = self.ffn(x)
        # Decomposition
        x, trend3 = self.decomp3(x + self.dropout(out))
        # Output
        trend = self.out_proj(
            (trend1 + trend2 + trend3).transpose(-1, -2)
        ).transpose(-1, -2)
        return (x, trend), (self_attn_weight, cross_attn_weight)
