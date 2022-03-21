# -*- coding: utf-8 -*-

import argparse
import os

import torch

from models import EncoderDecoderEstimator, DecoderOnlyEstimator, AutoregressionEstimator, AutoformerEstimator


def parse_args(): 
    parser = argparse.ArgumentParser('[Experiment] Time Series Forecasting via Transformers')

    task = parser.add_argument_group('Task')
    task.add_argument('--desc', type=str, default='exp', help='Experiment description')
    task.add_argument('--data', type=str, choices=['ETTm2', 'Electricity', 'Exchange', 'Traffic', 'Weather', 'ILI'], help='Data of interest')
    task.add_argument('--data_path', type=str, help='Path to the dataset')
    task.add_argument('--task', type=str, default='M', choices=['U', 'M'], help='Univariate or multivariate forecasting')
    task.add_argument('--ckpt', type=str, default='/usr2/home/yongyiw/ckpt/ltsf/', help='Location to store model checkpoints')
    task.add_argument('--freq', type=str, default='h', help='Sampling rate duing the time feature extraction')
    task.add_argument('--len_enc', type=int, help='Input sequence length to the encoder')
    task.add_argument('--len_label', type=int, help='Prepended length of the decoder')
    task.add_argument('--len_pred', type=int, help='One-path prediction length of the decoder')

    model = parser.add_argument_group('Model')
    model.add_argument('--model', type=str, choices=['enc-dec', 'dec', 'auto', 'autoformer'], help='Architecture name')
    model.add_argument('--d_enc_in', type=int, help='Dimension of encoder input')
    model.add_argument('--d_dec_in', type=int, help='Dimension of decoder input')
    model.add_argument('--d_dec_out', type=int, help='Dimension of decoder output')
    model.add_argument('--d_model', type=int, default=512, help='Hidden dimension of model')
    model.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    model.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    model.add_argument('--n_enc_layers', type=int, default=2, help='Number of encoder layers')
    model.add_argument('--n_dec_layers', type=int, default=1, help='Number of decoder layers')
    model.add_argument('--len_window', type=int, default=25, help='Sliding window size for series decomposition')
    model.add_argument('--c_sampling', type=int, default=3, help='Constant sampling factor for attention blocks')

    training = parser.add_argument_group('Training')
    training.add_argument('--dropout', type=float, default=0.05, help='dropout')
    training.add_argument('--output_attn', action='store_true', help='Whether to output attention weight matrices')
    training.add_argument('--n_epochs', type=int, default=6, help='Maximum training epochs')
    training.add_argument('--batch_size', type=int, default=32, help='Batch size of training input data')
    training.add_argument('--n_reps', type=int, default=3, help='Number of repetitive runs')
    training.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    training.add_argument('--lr', type=float, default=1E-4, help='Optimizer learning rate')
    training.add_argument('--lr_schedule', action='store_true',help='Whether to use learning rate schedule')
    training.add_argument('--devices', type=int, default=[0], nargs='*',help='GPU device id(s); if not provided, use CPU instead')

    args = parser.parse_args()
    args.config = '{}_{}_tf{}_f{}_le{}_ll{}_lp{}_m{}_ein{}_din{}_dout{}_dm{}_dff{}_nh{}_ne{}_nd{}_lw{}_c{}_dr{}_E{}_B{}_p{}_lr{}_sch{}'.format(
        args.desc.upper(), os.path.basename(args.data_path), args.task, args.freq, args.len_enc, args.len_label, args.len_pred, 
        args.model, args.d_enc_in, args.d_dec_in, args.d_dec_out, args.d_model, args.d_ff, args.n_heads, args.n_enc_layers, args.n_dec_layers, args.len_window, args.c_sampling, 
        args.dropout, args.n_epochs, args.batch_size, args.patience, args.lr, int(args.lr_schedule)
    )
    return args


def main(): 
    cfg = parse_args()
    print(cfg)

    if cfg.model == 'enc-dec': 
        Estimator = EncoderDecoderEstimator
    elif cfg.model == 'dec': 
        Estimator = DecoderOnlyEstimator
    elif cfg.model == 'auto': 
        Estimator = AutoregressionEstimator
    elif cfg.model == 'autoformer': 
        Estimator = AutoformerEstimator
    else: 
        raise KeyError(cfg.model)

    for i in range(cfg.n_reps): 
        print('########## EXPERIMENT {} ##########'.format(i))
        estimator = Estimator(cfg)
        trainloader, devloader, testloader = estimator.get_data()

        # Train and Development
        for _ in range(cfg.n_epochs): 
            early_stop = estimator.train(trainloader, devloader)
            if estimator.scheduler is not None: 
                estimator.scheduler.step()
            if early_stop: 
                break

        # Test
        estimator.test(testloader, estimator.ckpt_path)

        torch.cuda.empty_cache()

    return 0


if __name__ == '__main__': 
    main()
