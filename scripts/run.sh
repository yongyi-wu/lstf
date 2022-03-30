#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lstf


for model in enc-dec dec auto autoformer
do
    for data in ETTm2 Electricity Exchange Traffic Weather ILI
    do
        if [ $model != auto ]
        then
            ne=2
            nd=1
            if [ $data != ILI ]
            then
                l_enc=96
                l_label=48
            else
                l_enc=36
                l_label=18
            fi
        else
            ne=0
            nd=1
            if [ $data != ILI ]
            then
                l_enc=0
                l_label=96
            else
                l_enc=0
                l_label=36
            fi
        fi

        if [ $model = autoformer ]
        then
            attn=autocorrelation
        else
            attn=dot
        fi

        case $data in
            ETTm2)
                dpath=/usr2/home/yongyiw/data/ETT-small/ETTm2.csv
                ein=7
                din=7
                dout=7
                ;;
            Electricity)
                dpath=/usr2/home/yongyiw/data/electricity/electricity.csv
                ein=321
                din=321
                dout=321
                ;;
            Exchange)
                dpath=/usr2/home/yongyiw/data/exchange_rate/exchange_rate.csv
                ein=8
                din=8
                dout=8
                ;;
            Traffic)
                dpath=/usr2/home/yongyiw/data/traffic/traffic.csv
                ein=862
                din=862
                dout=862
                ;;
            Weather)
                dpath=/usr2/home/yongyiw/data/weather/weather.csv
                ein=21
                din=21
                dout=21
                ;;
            ILI)
                dpath=/usr2/home/yongyiw/data/illness/national_illness.csv
                ein=7
                din=7
                dout=7
                ;;
            *)
                dpath=/dev/null
                ein=7
                din=7
                dout=7
                ;;
        esac

        if [ $data != ILI ]
        then
            l_preds=( 96 192 336 720 )
        else
            l_preds=( 24 36 48 60 )
        fi

        for l_pred in ${l_preds[@]}
        do
            python main.py \
                --data $data \
                --data_path $dpath \
                --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$model/ \
                --len_enc $l_enc \
                --len_label $l_label \
                --len_pred $l_pred \
                --model $model \
                --attn $attn \
                --d_enc_in $ein \
                --d_dec_in $din \
                --d_dec_out $dout \
                --n_enc_layers $ne \
                --n_dec_layers $nd \
                --output_attn \
                --lr_schedule \
                --devices 0 1 2 3
        done
    done
done
