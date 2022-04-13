#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lstf


data=Synthetic
lenc=96
llabel=48
lpred=192
dm=128
dff=512
ne=2
nd=1


# Vanilla Autoformer
for dataset in sinx x sinx_x sinx_sqrtx sinx_x2_asym sinx_x2_sym xsinx sinx_sin2x_sin4x sinx_c
do
    for l_win in 5 13 25 51
    do
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/synth/${dataset}.npy \
            --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$dataset/ \
            --len_enc $lenc \
            --len_label $llabel \
            --len_pred $lpred \
            --model autoformer \
            --attn autocorrelation \
            --d_model $dm \
            --d_ff $dff \
            --n_enc_layers $ne \
            --n_dec_layers $nd \
            --len_window $l_win \
            --no_temporal \
            --output_attn \
            --lr_schedule \
            --devices 0 1 2 3
    done
done

# No decomposition block
for dataset in sinx x sinx_x sinx_sqrtx sinx_x2_asym sinx_x2_sym xsinx sinx_sin2x_sin4x sinx_c
do
    python main.py \
        --data $data \
        --data_path /usr2/home/yongyiw/data/synth/${dataset}.npy \
        --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$dataset/ \
        --len_enc $lenc \
        --len_label $llabel \
        --len_pred $lpred \
        --model autoformer \
        --attn autocorrelation \
        --d_model $dm \
        --d_ff $dff \
        --n_enc_layers $ne \
        --n_dec_layers $nd \
        --len_window 0 \
        --no_temporal \
        --output_attn \
        --lr_schedule \
        --devices 0 1 2 3
done

# Vanilla attention block
for dataset in sinx x sinx_x sinx_sqrtx sinx_x2_asym sinx_x2_sym xsinx sinx_sin2x_sin4x sinx_c
do
    python main.py \
        --data $data \
        --data_path /usr2/home/yongyiw/data/synth/${dataset}.npy \
        --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$dataset/ \
        --len_enc $lenc \
        --len_label $llabel \
        --len_pred $lpred \
        --model autoformer \
        --attn dot \
        --d_model $dm \
        --d_ff $dff \
        --n_enc_layers $ne \
        --n_dec_layers $nd \
        --len_window 25 \
        --no_temporal \
        --output_attn \
        --lr_schedule \
        --devices 0 1 2 3
done
