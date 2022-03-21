# -*- coding: utf-8 -*-

import torch
from torch import nn

from .layers import (
    DataEmbedding, MultiheadAutoCorrelation, AutoformerEncoderLayer, AutoformerDecoderLayer, SeriesDecomposition
)
from exp import BaseEstimator


class Autoformer(nn.Module): 
    def __init__(
        self, 
        d_enc_in, d_dec_in, d_dec_out, 
        d_model=512, 
        n_heads=8, 
        n_enc_layers=2, 
        n_dec_layers=1, 
        d_ff=2048, 
        dropout=0.0, 
        freq='h', 
        len_window=25, 
        c_sampling=1, 
        output_attn=False, 
    ): 
        super().__init__()
        self.enc_embedding = DataEmbedding(d_enc_in, d_model, pos=False, freq=freq, dropout=dropout)
        self.dec_embedding = DataEmbedding(d_dec_in, d_model, pos=False, freq=freq, dropout=dropout)
        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(
                MultiheadAutoCorrelation(d_model, n_heads=n_heads, dropout=dropout, c_sampling=c_sampling), 
                d_model, 
                d_ff, 
                len_window=len_window, 
                dropout=dropout, 
                output_attn=output_attn
            ) for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.decomp = SeriesDecomposition(len_window)
        self.decoder = nn.ModuleList([
            AutoformerDecoderLayer(
                MultiheadAutoCorrelation(d_model, n_heads=n_heads, dropout=dropout, c_sampling=c_sampling), 
                MultiheadAutoCorrelation(d_model, n_heads=n_heads, dropout=dropout, c_sampling=c_sampling), 
                d_model, 
                d_ff, 
                d_dec_out, 
                len_window=len_window, 
                dropout=dropout, 
                output_attn=output_attn
            ) for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model, elementwise_affine=False)
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

    def decode(self, enc_out, dec_in_s, dec_in_t, self_attn_mask=None, cross_attn_mask=None): 
        dec_out_s = dec_in_s
        dec_out_t = dec_in_t
        dec_self_weights = []
        dec_enc_weights = []
        for layer in self.decoder: 
            (dec_out_s, res_t), (self_attn_weight, cross_attn_weight) = layer(
                dec_out_s, enc_out, 
                self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask
            )
            dec_out_t = dec_out_t + res_t
            dec_self_weights.append(self_attn_weight)
            dec_enc_weights.append(cross_attn_weight)
        dec_out_s = self.dec_norm(dec_out_s)
        return (dec_out_s, dec_out_t), (dec_self_weights, dec_enc_weights)

    def forward(
        self, 
        x_enc, x_time_enc, 
        x_dec_s, x_dec_t, x_time_dec, 
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
            None
        dec_self_mask
            None
        dec_enc_mask
            None

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
        dec_in_s = self.dec_embedding(x_dec_s, x_time_dec)
        (dec_out_s, dec_out_t), (dec_self_weights, dec_enc_weights) = self.decode(
            enc_out, dec_in_s, x_dec_t, 
            self_attn_mask=dec_self_mask, cross_attn_mask=dec_enc_mask
        )
        # Output projection
        out = self.out_proj(dec_out_s) + dec_out_t
        return out, ((enc_self_weights, None), (dec_self_weights, dec_enc_weights))


class AutoformerEstimator(BaseEstimator): 
    def get_model(self): 
        return Autoformer(
            self.cfg.d_enc_in, 
            self.cfg.d_dec_in, 
            self.cfg.d_dec_out, 
            d_model=self.cfg.d_model, 
            n_heads=self.cfg.n_heads, 
            n_enc_layers=self.cfg.n_enc_layers, 
            n_dec_layers=self.cfg.n_dec_layers, 
            d_ff=self.cfg.d_ff, 
            dropout=self.cfg.dropout, 
            freq=self.cfg.freq, 
            len_window=self.cfg.len_window, 
            c_sampling=self.cfg.c_sampling, 
            output_attn=self.cfg.output_attn
        )

    def get_dec_input(self, x_enc, x_dec): 
        if isinstance(self.model, nn.DataParallel): 
            decomp = self.model.module.decomp
        else: 
            decomp = self.model.decomp
        x_dec_s, x_dec_t = decomp(x_enc)
        zeros = torch.zeros((x_dec.shape[0], self.cfg.len_pred, x_dec.shape[2]), device=x_dec.device)
        x_mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.cfg.len_pred, 1)
        x_dec_s = torch.cat((x_dec_s[:, -self.cfg.len_label:, :], zeros), dim=1)
        x_dec_t = torch.cat((x_dec_t[:, -self.cfg.len_label:, :], x_mean), dim=1)
        return x_dec_s, x_dec_t

    def _step(self, data): 
        enc_x = data['x'].to(self.device, dtype=torch.float)
        enc_x_time = data['x_time'].to(self.device, dtype=torch.float)
        y = data['y'].to(self.device, dtype=torch.float)
        dec_y_time = data['y_time'].to(self.device, dtype=torch.float)
        dec_y_s, dec_y_t = self.get_dec_input(enc_x, y)

        if self.mode == 'train': 
            self.optimizer.zero_grad()

        yhat, _ = self.model(
            enc_x, enc_x_time, dec_y_s, dec_y_t, dec_y_time
        )
        yhat = yhat[:, -self.cfg.len_pred:, :]
        y = y[:, -self.cfg.len_pred:, :]

        loss = self.criterion(yhat, y)
        if self.mode == 'train': 
            loss.backward()
            self.optimizer.step()

        return loss.item(), yhat.detach().cpu().numpy(), y.detach().cpu().numpy()
