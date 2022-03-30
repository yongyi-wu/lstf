#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lstf


data=Synthetic
l_enc=96
l_label=48
l_pred=192
d_model=128
d_ff=512
ne=2
nd=1

# Vanilla Autoformer
for dataset in sinx sinx_x x xsinx sinx_sin2x_sin4x sinx_c
do
    for l_win in 5 13 25 51
    do
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/synth/${dataset}.npy \
            --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$dataset/ \
            --len_enc $l_enc \
            --len_label $l_label \
            --len_pred $l_pred \
            --model autoformer \
            --attn autocorrelation \
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
for dataset in sinx sinx_x x xsinx
do
    python main.py \
        --data $data \
        --data_path /usr2/home/yongyiw/data/synth/${dataset}.npy \
        --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$dataset/ \
        --len_enc $l_enc \
        --len_label $l_label \
        --len_pred $l_pred \
        --model autoformer \
        --attn autocorrelation \
        --n_enc_layers $ne \
        --n_dec_layers $nd \
        --len_window 0 \
        --no_temporal \
        --output_attn \
        --lr_schedule \
        --devices 0 1 2 3
done

# Vanilla attention block
for dataset in sinx_sin2x_sin4x sinx_c
do
    python main.py \
        --data $data \
        --data_path /usr2/home/yongyiw/data/synth/${dataset}.npy \
        --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$dataset/ \
        --len_enc $l_enc \
        --len_label $l_label \
        --len_pred $l_pred \
        --model autoformer \
        --attn dot \
        --n_enc_layers $ne \
        --n_dec_layers $nd \
        --len_window 25 \
        --no_temporal \
        --output_attn \
        --lr_schedule \
        --devices 0 1 2 3
done
