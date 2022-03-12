# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import MultiheadAttention

from .layers import DataEmbedding, EncoderLayer, get_triangular_causal_mask
from exp import BaseEstimator


class DecoderOnly(nn.Module): 
    def __init__(
        self, 
        d_dec_in, d_dec_out, 
        d_model=512, 
        n_heads=8, 
        n_dec_layers=1, 
        d_ff=2048, 
        dropout=0.0, 
        freq='h', 
        output_attn=False, 
    ): 
        super().__init__()
        self.dec_embedding = DataEmbedding(d_dec_in, d_model, freq, dropout)
        self.decoder = nn.ModuleList([
            # EncoderLayer is used here because no cross attention is performed
            EncoderLayer(
                MultiheadAttention(d_model, n_heads, dropout=dropout), 
                d_model, 
                d_ff, 
                dropout=dropout, 
                output_attn=output_attn
            ) for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_dec_out)

    def forward(
        self, 
        x_dec, x_time_dec, 
        dec_self_mask=None, 
    ): 
        """
        Input
        ----------
        x_dec
            Shape (B, len_label+len_pred, d_dec_in)
        x_time_dec
            Shape (B, len_label+len_pred, d_temporal)
        dec_self_mask
            None or Shape (len_label+len_pred, len_label+len_pred)

        Output
        ----------
        out
            Shape (B, len_label+len_pred, d_dec_out)
        (None, (None, None), (dec_self_weights, None))
            Attention weights for each layer
        """
        # Decoder
        dec_out = self.dec_embedding(x_dec, x_time_dec)
        dec_self_weights = []
        for layer in self.decoder: 
            dec_out, (self_attn_weight, _) = layer(
                dec_out, 
                self_attn_mask=dec_self_mask
            )
            dec_self_weights.append(self_attn_weight)
        dec_out = self.dec_norm(dec_out)
        # Output projection
        out = self.out_proj(dec_out)
        return out, ((None, None), (dec_self_weights, None))


class DecoderOnlyEstimator(BaseEstimator): 
    def __init__(self, cfg): 
        assert cfg.len_enc == 0 and cfg.len_label > 0
        super().__init__(cfg)

    def get_model(self): 
        return DecoderOnly(
            self.cfg.d_dec_in, 
            self.cfg.d_dec_out, 
            d_model=self.cfg.d_model, 
            n_heads=self.cfg.n_heads, 
            n_dec_layers=self.cfg.n_dec_layers, 
            d_ff=self.cfg.d_ff, 
            dropout=self.cfg.dropout, 
            freq=self.cfg.freq, 
            output_attn=self.cfg.output_attn
        )

    def _step(self, data): 
        assert data['x'].shape[1] == data['x_time'].shape[1] == 0
        y = data['y'].to(self.device, dtype=torch.float)
        dec_y_time = data['y_time'].to(self.device, dtype=torch.float)

        B, _, d_dec_in = y.shape
        dec_y = torch.zeros((B, self.cfg.len_pred, d_dec_in)).to(self.device, dtype=torch.float)
        dec_y = torch.cat((y[:, :self.cfg.len_label, :], dec_y), dim=1)

        if self.mode == 'train': 
            self.optimizer.zero_grad()

        yhat, _ = self.model(
            dec_y, dec_y_time, 
            dec_self_mask=get_triangular_causal_mask(dec_y)
        )
        yhat = yhat[:, -self.cfg.len_pred:, :]
        y = y[:, -self.cfg.len_pred:, :]

        loss = self.criterion(yhat, y)
        if self.mode == 'train': 
            loss.backward()
            self.optimizer.step()

        return loss.item(), yhat.detach().cpu().numpy(), y.detach().cpu().numpy()
