# -*- coding: utf-8 -*-

import logging
import os
import random
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ETTDataset


def seed_everything(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_if_not_exists(new_dir): 
    if not os.path.exists(new_dir): 
        os.system('mkdir -p {}'.format(new_dir))


def config_logging(log_dir): 
    make_if_not_exists(log_dir)
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, 'exp.log')
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s\t%(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        handlers=[file_handler, stdout_handler]
    )
    logger = logging.getLogger('train')
    return logger


class BaseEstimator(object): 
    """A wrapper class to perform training, evluation or testing while accumulating and logging results"""
    def __init__(self, cfg): 
        self.cfg = cfg
        self.model = self._get_model()
        self.criterion, self.optimizer, self.scheduler = self._get_auxiliaries()
        self.device = self._get_devices()
        self.logger = self.cfg.logger
        self.writer = self.cfg.writer
        self.mode = None # {'train', 'dev', 'test'}

        self.epochs = 0
        self.train_steps = 0
        self.dev_steps = 0
        self.best_dev_loss = float('inf')

        make_if_not_exists(self.cfg.ckpt_dir)
        self.ckpt_path = os.path.join(self.cfg.ckpt_dir, '{}.pt'.format(self.cfg.config))

    def get_data(self): 
        if self.cfg.data == 'ETT': 
            Data = ETTDataset
        else: 
            raise KeyError(self.cfg.data)
        data_path = self.cfg.data_path
        len_enc, len_label, len_pred = self.cfg.len_enc, self.cfg.len_label, self.cfg.len_pred
        freq = self.cfg.freq

        trainset = Data('train', data_path, len_enc, len_label, len_pred, freq)
        trainloader = DataLoader(trainset, self.cfg.batch_size, shuffle=True)
        
        devset = Data('dev', data_path, len_enc, len_label, len_pred, freq)
        devloader = DataLoader(devset, self.cfg.batch_size * 4, shuffle=False)
        
        testset = Data('test', data_path, len_enc, len_label, len_pred, freq)
        testloader = DataLoader(testset, self.cfg.batch_size * 4, shuffle=False)
        
        return trainloader, devloader, testloader

    def get_model(self): 
        """
        Output
        ----------
        model
            nn.Module
        """
        raise NotImplementedError('Implement it in the subclass')

    def _get_auxiliaries(self): 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        if self.cfg.lr_schedule: 
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, 
                lambda epoch: 0.5 ** ((epoch - 1) // 1), 
                verbose=True
            )
        else: 
            scheduler = None
        return criterion, optimizer, scheduler

    def _get_devices(self): 
        if self.cfg.use_gpu: 
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu) if not self.cfg.use_multi_gpu else self.cfg.devices
            device = torch.device('cuda:{}'.format(self.cfg.gpu))
            print('Use CUDA: {}'.format(self.cfg.gpu))
        else: 
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _step(self, data): 
        """
        This function is responsible for feeding the data into the model and obtain predictions. 
        If `self.mode == 'train'`, perform backpropagation and update optimizer and, if given, scheduler; 
        if `self.mode == 'dev'`, we still compute loss and return ground true label, but no backpropagation nor optimizer update; 
        if `self.mode == 'test'`, no ground true label is presented so loss will not be calculated. 

        Input
        ----------
        data
            A dictionary of mini-batch input obtained from Dataset.__getitem__, each with shape (B, len_seq), type torch.tensor
            Before fed into the model, inputs should be convert to appropriate type(s) and device(s)

        Output
        ----------
        loss
            A scalar for the entire batch, type float; None if no label provided
        prob
            Model predictions as the probability for each label, shape (B, n_labels), type np.ndarray
        y
            Ground true labels for the batch, shape (B, n_labels), type np.ndarray; None if no label provided
        """
        raise NotImplementedError('Implement it in the subclass')

    def _write_stats(self, stats): 
        if self.writer is not None: 
            self.writer.add_scalar(
                '{}/loss'.format(self.mode), 
                stats['loss'], 
                self.train_steps if self.mode == 'train' else self.dev_steps
            )

    def train(self, trainloader, devloader=None): 
        if trainloader is None: 
            return None
        self.mode = 'train'
        self.model.train()
        assert self.optimizer is not None, 'Optimizer is required'
        self.epochs += 1
        # Train for one epoch
        tbar = tqdm(trainloader, dynamic_ncols=True)
        for data in tbar: 
            loss, *_ = self._step(data)
            self._write_stats({'loss': loss})
            tbar.set_description('train/loss - {:.4f}'.format(loss))
            self.train_steps += 1
        # Evaluate the training process on the development dataset
        if devloader is not None: 
            dev_loss, *_ = self.dev(devloader)
        return self.early_stopping(dev_loss)

    def _eval(self, evalloader): 
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        probs = []
        for data in tbar: 
            loss, prob, y = self._step(data)
            if self.mode == 'dev': 
                tbar.set_description('dev/loss - {:.4f}'.format(loss))
                eval_loss.append(loss)
                ys.append(y)
            probs.append(prob)
        loss = np.mean(eval_loss).item() if self.mode == 'dev' else None
        ys = np.concatenate(ys, axis=0) if self.mode == 'dev' else None
        probs = np.concatenate(probs, axis=0)
        self._write_stats({'loss': loss})
        return loss, probs, ys

    def dev(self, devloader): 
        if devloader is None: 
            return None
        self.mode = 'dev'
        results = self._eval(devloader)
        self.dev_steps += 1
        return results

    def test(self, testloader): 
        if testloader is None: 
            return None
        self.mode = 'test'
        results = self._eval(testloader)
        return results

    def early_stopping(self, dev_loss): 
        """Return True if early stopping is needed"""
        if self.best_dev_loss is None or dev_loss < self.best_dev_loss: 
            self.best_dev_loss = dev_loss
            self.save(self.ckpt_path)
            if self.logger is not None: 
                self.logger.info('Saving checkpoint: {}'.format(self.ckpt_path))
            self.patience = self.cfg.patience
            return False
        else: 
            self.patience -= 1
            if self.logger is not None: 
                self.logger.info('Patience: {} / {}'.format(self.patience, self.cfg.patience))
            if self.patience <= 0: 
                return True
            
    def save(self, ckpt_path): 
        checkpoint = {
            'epochs': self.epochs, 
            'train_steps': self.train_steps, 
            'dev_steps': self.dev_steps, 
            'best_dev_loss': self.best_dev_loss, 
            'model': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None, 
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(checkpoint, ckpt_path)

    def load(self, ckpt_path): 
        print('Loading checkpoint {}'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        self.epochs = checkpoint['epochs']
        self.train_steps = checkpoint['train_steps']
        self.dev_steps = checkpoint['dev_steps']
        self.best_dev_loss = checkpoint['best_dev_loss']
        self.model.load_state_dict(checkpoint['model'])
        if self.optimizer is not None: 
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else: 
            print('Optimizer is not loaded')
        if self.scheduler is not None: 
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else: 
            print('Scheduler is not loaded')
