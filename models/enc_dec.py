# -*- coding: utf-8 -*-

import torch
from torch import nn

from .layers import (
    DataEmbedding, MultiheadAttention, EncoderLayer, DecoderLayer, get_triangular_causal_mask
)
from exp import BaseEstimator


class EncoderDecoder(nn.Module): 
    def __init__(
        self, 
        d_enc_in, d_dec_in, d_dec_out, 
        d_model=512, 
        n_heads=8, 
        n_enc_layers=2, 
        n_dec_layers=1, 
        d_ff=2048, 
        dropout=0.0, 
        temp=True, 
        freq='h', 
        output_attn=False, 
    ): 
        super().__init__()
        self.enc_embedding = DataEmbedding(d_enc_in, d_model, temp=temp, freq=freq, dropout=dropout)
        self.dec_embedding = DataEmbedding(d_dec_in, d_model, temp=temp, freq=freq, dropout=dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(
                MultiheadAttention(d_model, n_heads=n_heads, dropout=dropout), 
                d_model, 
                d_ff, 
                dropout=dropout, 
                output_attn=output_attn
            ) for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([
            DecoderLayer(
                MultiheadAttention(d_model, n_heads=n_heads, dropout=dropout), 
                MultiheadAttention(d_model, n_heads=n_heads, dropout=dropout), 
                d_model, 
                d_ff, 
                dropout=dropout, 
                output_attn=output_attn
            ) for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_dec_out)

    def encode(self, enc_in, self_attn_mask=None): 
        enc_out = enc_in
        enc_self_weights = []
        for layer in self.encoder: 
            enc_out, (self_attn_weight, _) = layer(
                enc_out, 
                self_attn_mask=self_attn_mask
            )
            enc_self_weights.append(self_attn_weight)
        enc_out = self.enc_norm(enc_out)
        return enc_out, (enc_self_weights, None)

    def decode(self, enc_out, dec_in, self_attn_mask=None, cross_attn_mask=None): 
        dec_out = dec_in
        dec_self_weights = []
        dec_enc_weights = []
        for layer in self.decoder: 
            dec_out, (self_attn_weight, cross_attn_weight) = layer(
                dec_out, enc_out, 
                self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask
            )
            dec_self_weights.append(self_attn_weight)
            dec_enc_weights.append(cross_attn_weight)
        dec_out = self.dec_norm(dec_out)
        return dec_out, (dec_self_weights, dec_enc_weights)

    def forward(
        self, 
        x_enc, x_time_enc, 
        x_dec, x_time_dec, 
        enc_self_mask=None, 
        dec_self_mask=None, 
        dec_enc_mask=None
    ): 
        """
        Input
        ----------
        x_enc
            Shape (B, len_enc, d_enc_in)
        x_time_enc
            Shape (B, len_enc, d_temporal)
        x_dec
            Shape (B, len_label+len_pred, d_dec_in)
        x_time_dec
            Shape (B, len_label+len_pred, d_temporal)
        enc_self_mask
            None or Shape (len_enc, len_enc)
        dec_self_mask
            None or Shape (len_label+len_pred, len_label+len_pred)
        dec_enc_mask
            None or Shape (len_label+len_pred, len_enc)

        Output
        ----------
        out
            Shape (B, len_label+len_pred, d_dec_out)
        ((enc_self_weights, None), (dec_self_weights, dec_enc_weights))
            Attention weights for each layer
        """
        # Encoder
        enc_in = self.enc_embedding(x_enc, x_time_enc)
        enc_out, (enc_self_weights, _) = self.encode(
            enc_in, 
            self_attn_mask=enc_self_mask
        )
        # Decoder
        dec_in = self.dec_embedding(x_dec, x_time_dec)
        dec_out, (dec_self_weights, dec_enc_weights) = self.decode(
            enc_out, dec_in, 
            self_attn_mask=dec_self_mask, cross_attn_mask=dec_enc_mask
        )
        # Output projection
        out = self.out_proj(dec_out)
        return out, ((enc_self_weights, None), (dec_self_weights, dec_enc_weights))


class EncoderDecoderEstimator(BaseEstimator): 
    def get_model(self): 
        return EncoderDecoder(
            self.cfg.d_enc_in, 
            self.cfg.d_dec_in, 
            self.cfg.d_dec_out, 
            d_model=self.cfg.d_model, 
            n_heads=self.cfg.n_heads, 
            n_enc_layers=self.cfg.n_enc_layers, 
            n_dec_layers=self.cfg.n_dec_layers, 
            d_ff=self.cfg.d_ff, 
            dropout=self.cfg.dropout, 
            temp=not self.cfg.no_temporal, 
            freq=self.cfg.freq, 
            output_attn=self.cfg.output_attn
        )

    def _step(self, data): 
        enc_x = data['x'].to(self.device, dtype=torch.float)
        enc_x_time = data['x_time'].to(self.device, dtype=torch.float)
        y = data['y'].to(self.device, dtype=torch.float)
        dec_y_time = data['y_time'].to(self.device, dtype=torch.float)

        B, _, d_dec_in = y.shape
        dec_y = torch.zeros((B, self.cfg.len_pred, d_dec_in)).to(self.device, dtype=torch.float)
        dec_y = torch.cat((y[:, :self.cfg.len_label, :], dec_y), dim=1)

        if self.mode == 'train': 
            self.optimizer.zero_grad()

        yhat, _ = self.model(
            enc_x, enc_x_time, dec_y, dec_y_time, 
            dec_self_mask=get_triangular_causal_mask(dec_y)
        )
        yhat = yhat[:, -self.cfg.len_pred:, :]
        y = y[:, -self.cfg.len_pred:, :]

        loss = self.criterion(yhat, y)
        if self.mode == 'train': 
            loss.backward()
            self.optimizer.step()

        return loss.item(), yhat.detach().cpu().numpy(), y.detach().cpu().numpy()
