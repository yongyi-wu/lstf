#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate informer


data=ETTm2
for model in enc-dec dec
do
    for l_pred in 96 192 336 720
    do
        if [ $model = enc-dec ]
        then
            ne=2
            nd=1
            l_enc=96
            l_label=48
        else
            ne=0
            nd=1
            l_enc=0
            l_label=96
        fi
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/ETT-small/ETTm2.csv \
            --ckpt /usr2/home/yongyiw/ckpt/ltsf/$data/$model/ \
            --model $model \
            --len_enc $l_enc \
            --len_label $l_label \
            --len_pred $l_pred \
            --d_enc_in 7 \
            --d_dec_in 7 \
            --d_dec_out 7 \
            --n_enc_layers $ne \
            --n_dec_layers $nd \
            --output_attn \
            --lr_schedule \
            --devices 0 1 2 3
    done
done


data=Electricity
for model in enc-dec dec
do
    for l_pred in 96 192 336 720
    do
        if [ $model = enc-dec ]
        then
            ne=2
            nd=1
            l_enc=96
            l_label=48
        else
            ne=0
            nd=1
            l_enc=0
            l_label=96
        fi
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/electricity/electricity.csv \
            --ckpt /usr2/home/yongyiw/ckpt/ltsf/$data/$model/ \
            --model $model \
            --len_enc $l_enc \
            --len_label $l_label \
            --len_pred $l_pred \
            --d_enc_in 321 \
            --d_dec_in 321 \
            --d_dec_out 321 \
            --n_enc_layers $ne \
            --n_dec_layers $nd \
            --output_attn \
            --lr_schedule \
            --devices 0 1 2 3
    done
done


data=Exchange
for model in enc-dec dec
do
    for l_pred in 96 192 336 720
    do
        if [ $model = enc-dec ]
        then
            ne=2
            nd=1
            l_enc=96
            l_label=48
        else
            ne=0
            nd=1
            l_enc=0
            l_label=96
        fi
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/exchange_rate/exchange_rate.csv \
            --ckpt /usr2/home/yongyiw/ckpt/ltsf/$data/$model/ \
            --model $model \
            --len_enc $l_enc \
            --len_label $l_label \
            --len_pred $l_pred \
            --d_enc_in 8 \
            --d_dec_in 8 \
            --d_dec_out 8 \
            --n_enc_layers $ne \
            --n_dec_layers $nd \
            --output_attn \
            --lr_schedule \
            --devices 0 1 2 3
    done
done


data=Traffic
for model in enc-dec dec
do
    for l_pred in 96 192 336 720
    do
        if [ $model = enc-dec ]
        then
            ne=2
            nd=1
            l_enc=96
            l_label=48
        else
            ne=0
            nd=1
            l_enc=0
            l_label=96
        fi
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/traffic/traffic.csv \
            --ckpt /usr2/home/yongyiw/ckpt/ltsf/$data/$model/ \
            --model $model \
            --len_enc $l_enc \
            --len_label $l_label \
            --len_pred $l_pred \
            --d_enc_in 862 \
            --d_dec_in 862 \
            --d_dec_out 862 \
            --n_enc_layers $ne \
            --n_dec_layers $nd \
            --output_attn \
            --lr_schedule \
            --devices 0 1 2 3
    done
done


data=Weather
for model in enc-dec dec
do
    for l_pred in 96 192 336 720
    do
        if [ $model = enc-dec ]
        then
            ne=2
            nd=1
            l_enc=96
            l_label=48
        else
            ne=0
            nd=1
            l_enc=0
            l_label=96
        fi
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/weather/weather.csv \
            --ckpt /usr2/home/yongyiw/ckpt/ltsf/$data/$model/ \
            --model $model \
            --len_enc $l_enc \
            --len_label $l_label \
            --len_pred $l_pred \
            --d_enc_in 21 \
            --d_dec_in 21 \
            --d_dec_out 21 \
            --n_enc_layers $ne \
            --n_dec_layers $nd \
            --output_attn \
            --lr_schedule \
            --devices 0 1 2 3
    done
done


data=ILI
for model in enc-dec dec
do
    for l_pred in 24 36 48 60
    do
        if [ $model = enc-dec ]
        then
            ne=2
            nd=1
            l_enc=36
            l_label=18
        else
            ne=0
            nd=1
            l_enc=0
            l_label=36
        fi
        python main.py \
            --data $data \
            --data_path /usr2/home/yongyiw/data/illness/national_illness.csv \
            --ckpt /usr2/home/yongyiw/ckpt/ltsf/$data/$model/ \
            --model $model \
            --len_enc $l_enc \
            --len_label $l_label \
            --len_pred $l_pred \
            --d_enc_in 7 \
            --d_dec_in 7 \
            --d_dec_out 7 \
            --n_enc_layers $ne \
            --n_dec_layers $nd \
            --output_attn \
            --lr_schedule \
            --devices 0 1 2 3
    done
done
