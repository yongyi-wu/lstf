# -*- coding: utf-8 -*-

import argparse

import torch

from models import EncoderDecoderEstimator, DecoderOnlyEstimator

def parse_args(): 
    parser = argparse.ArgumentParser('[Experiment] Time Series Forecasting via Transformers')

    task = parser.add_argument_group('Task')
    task.add_argument('--data', type=str, choices=['ETT'], help='Data of interest')
    task.add_argument('--data_path', type=str, default='ETTh1.csv', help='Path to the dataset')    
    task.add_argument('--task', type=str, choices=['U', 'M'], help='Univariate or multivariate forecasting')
    task.add_argument('--ckpt', type=str, help='Location to store model checkpoints')
    task.add_argument('--freq', type=str, default='h', help='Sampling rate duing the time feature extraction')
    task.add_argument('--t_feature', type=str, default='timeF', help='Temporal feature encoding')

    model = parser.add_argument_group('Model')
    model.add_argument('--model', type=str, choices=['enc-dec', 'dec'])
    model.add_argument('--d_enc_in', type=int, default=7, help='Dimension of encoder input')
    model.add_argument('--d_dec_in', type=int, default=7, help='Dimension of decoder input')
    model.add_argument('--d_dec_out', type=int, default=7, help='Dimension of decoder output')
    model.add_argument('--d_model', type=int, default=512, help='Hidden dimension of model')
    model.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    model.add_argument('--n_enc_layers', type=int, default=2, help='Number of encoder layers')
    model.add_argument('--n_dec_layers', type=int, default=1, help='Number of decoder layers')
    model.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    
    training = parser.add_argument_group('Training')
    training.add_argument('--len_enc', type=int, help='Input sequence length to the encoder')
    training.add_argument('--len_label', type=int, help='Prepended length of the decoder')
    training.add_argument('--len_pred', type=int, help='One-path prediction length of the decoder')
    training.add_argument('--padding', type=int, default=0, help='Padding type')
    training.add_argument('--dropout', type=float, default=0.05, help='dropout')
    training.add_argument('--output_attn', action='store_true', help='Whether to output attention weight matrices')
    training.add_argument('--n_epochs', type=int, default=6, help='Maximum training epochs')
    training.add_argument('--batch_size', type=int, default=32, help='Batch size of training input data')
    training.add_argument('--n_reps', type=int, default=2, help='Number of repetitive runs')
    training.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    training.add_argument('--lr', type=float, default=1E-4, help='Optimizer learning rate')
    training.add_argument('--lr_schedule', action='store_true',help='Whether to use learning rate schedule')
    training.add_argument('--devices', type=int, default=[0], nargs='*',help='GPU device id(s); if not provided, use CPU instead')

    return parser.parse_args()


def main(): 
    cfg = parse_args()
    print(cfg)

    if cfg.model == 'enc-dec': 
        Estimator = EncoderDecoderEstimator
    elif cfg.model == 'dec': 
        Estimator = DecoderOnlyEstimator
    else: 
        raise KeyError(cfg.model)

    for i in range(cfg.n_reps): 
        print('########## EXPERIMENT {} ##########'.format(i))
        # config = 

        estimator = Estimator(cfg)
        trainloader, devloader, testloader = estimator.get_data()

        # Train and Development
        for _ in range(cfg.epochs): 
            early_stop = estimator.train(trainloader, devloader)
            if estimator.scheduler is not None: 
                estimator.scheduler.step()
            if early_stop: 
                break

        # Test
        estimator.test(testloader)

        torch.cuda.empty_cache()

    return 0


if __name__ == '__main__': 
    main()
