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
                lenc=96
                llabel=48
            else
                lenc=36
                llabel=18
            fi
        else
            ne=0
            nd=1
            if [ $data != ILI ]
            then
                lenc=0
                llabel=96
            else
                lenc=0
                llabel=36
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
            lpreds=( 96 192 336 720 )
        else
            lpreds=( 24 36 48 60 )
        fi

        for lpred in ${lpreds[@]}
        do
            python main.py \
                --data $data \
                --data_path $dpath \
                --ckpt /usr2/home/yongyiw/ckpt/lstf/$data/$model/ \
                --len_enc $lenc \
                --len_label $llabel \
                --len_pred $lpred \
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
