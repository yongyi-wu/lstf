#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lstf


model=autoformer
ne=2
nd=1

for data in ETTm2 Electricity Exchange Traffic Weather ILI
do
    case $data in
        ETTm2)
            dpath=/usr2/home/yongyiw/data/ETT-small/ETTm2.csv
            dim=7
            ;;
        Electricity)
            dpath=/usr2/home/yongyiw/data/electricity/electricity.csv
            dim=321
            ;;
        Exchange)
            dpath=/usr2/home/yongyiw/data/exchange_rate/exchange_rate.csv
            dim=8
            ;;
        Traffic)
            dpath=/usr2/home/yongyiw/data/traffic/traffic.csv
            dim=862
            ;;
        Weather)
            dpath=/usr2/home/yongyiw/data/weather/weather.csv
            dim=21
            ;;
        ILI)
            dpath=/usr2/home/yongyiw/data/illness/national_illness.csv
            dim=7
            ;;
        *)
            dpath=/dev/null
            dim=-1
            ;;
    esac

    if [ $data != ILI ]
    then
        lenc=96
        llabel=48
        lpreds=( 96 192 336 720 )
    else
        lenc=36
        llabel=18
        lpreds=( 24 36 48 60 )
    fi

    for lpred in ${lpreds[@]}
    do
        for attn in autocorrelation dot
        do
            for lwin in 25 0
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
                    --d_enc_in $dim \
                    --d_dec_in $dim \
                    --d_dec_out $dim \
                    --n_enc_layers $ne \
                    --n_dec_layers $nd \
                    --len_window $lwin \
                    --output_attn \
                    --lr_schedule \
                    --devices 0 1 2 3
            done
        done
    done
done
