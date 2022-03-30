# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from .enc_dec import EncoderDecoder
from .layers import get_triangular_causal_mask
from exp import BaseEstimator


class Autoregression(EncoderDecoder): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.disable_incremental_decoding() # Incremental decoding is disabled by default

    @property
    def incremental_decoding(self): 
        return self._inc

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
        if self.incremental_decoding: 
            out, (dec_self_weights, dec_enc_weights) = self.inc_decode(
                enc_out, x_dec, x_time_dec, 
                cross_attn_mask=dec_enc_mask
            )
        else: 
            dec_in = self.dec_embedding(x_dec, x_time_dec)
            dec_out, (dec_self_weights, dec_enc_weights) = self.decode(
                enc_out, dec_in, 
                self_attn_mask=dec_self_mask, cross_attn_mask=dec_enc_mask
            )
            # Output projection
            out = self.out_proj(dec_out)
        return out, ((enc_self_weights, None), (dec_self_weights, dec_enc_weights))

    def enable_incremental_decoding(self, len_label, len_pred): 
        self.len_label = len_label
        self.len_pred = len_pred
        self._inc = True
        for layer in self.decoder: 
            layer.self_attn.enable_inc()

    def disable_incremental_decoding(self): 
        self._inc = False
        for layer in self.decoder: 
            layer.self_attn.disable_inc()

    def reset_incremental_decoding_states(self): 
        for layer in self.decoder: 
            layer.self_attn.reset_inc()

    def inc_decode(
        self, 
        enc_out, 
        x_dec, x_time_dec, 
        cross_attn_mask=None
    ): 
        assert self.incremental_decoding
        self.reset_incremental_decoding_states()
        # Decoder
        out = None
        for i in range(self.len_pred): 
            if i == 0: 
                # First step with right-shifted labels
                new_dec_in = self.dec_embedding(x_dec[:, :self.len_label + 1, ...], x_time_dec[:, :self.len_label + 1, ...])
            else: 
                # Input previous prediction
                new_dec_in = (
                    self.dec_embedding.value_embedding(new_out) + 
                    self.dec_embedding.position_embedding.pe[:, self.len_label + i].unsqueeze(1) + 
                    self.dec_embedding.temporal_embedding(x_time_dec[:, self.len_label + i, :].unsqueeze(1))
                )
                new_dec_in = self.dec_embedding.dropout(new_dec_in)
            new_dec_out, _ = self.decode(enc_out, new_dec_in, cross_attn_mask=cross_attn_mask)
            # Output projection
            new_out = self.out_proj(new_dec_out)[:, -1:, :]
            out = new_out if out is None else torch.cat((out, new_out), dim=1)
        return out.contiguous(), (None, None)


class AutoregressionEstimator(BaseEstimator): 
    def get_model(self): 
        return Autoregression(
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
        if isinstance(self.model, nn.DataParallel): 
            if self.mode == 'train': 
                self.model.module.disable_incremental_decoding()
            else: 
                self.model.module.enable_incremental_decoding(self.cfg.len_label, self.cfg.len_pred)
        else: 
            if self.mode == 'train': 
                self.model.module.disable_incremental_decoding()
            else: 
                self.model.enable_incremental_decoding(self.cfg.len_label, self.cfg.len_pred)

        enc_x = data['x'].to(self.device, dtype=torch.float)
        enc_x_time = data['x_time'].to(self.device, dtype=torch.float)
        y = data['y'].to(self.device, dtype=torch.float)
        dec_y_time = data['y_time'].to(self.device, dtype=torch.float)
        dec_y_time = F.pad(dec_y_time[:, :-1, :], (0, 0, 1, 0))

        B, _, d_dec_in = y.shape
        dec_y = torch.zeros((B, self.cfg.len_pred, d_dec_in)).to(self.device, dtype=torch.float)
        dec_y = torch.cat((y[:, :self.cfg.len_label, :], dec_y), dim=1)
        dec_y = F.pad(dec_y[:, :-1, :], (0, 0, 1, 0))

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
