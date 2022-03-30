# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch
from torch.utils.data import Dataset


class BaseTemporalFeature(object): 
    def __init__(self): 
        pass

    def __call__(self, index): 
        pass

    def __repr__(self): 
        return self.__class__.__name__ + "()"


class SecondOfMinute(BaseTemporalFeature): 
    def __call__(self, index): 
        return index.second / 59.0 - 0.5


class MinuteOfHour(BaseTemporalFeature): 
    def __call__(self, index): 
        return index.minute / 59.0 - 0.5


class HourOfDay(BaseTemporalFeature): 
    def __call__(self, index): 
        return index.hour / 23.0 - 0.5


class DayOfWeek(BaseTemporalFeature): 
    def __call__(self, index): 
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(BaseTemporalFeature): 
    def __call__(self, index): 
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(BaseTemporalFeature): 
    def __call__(self, index): 
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(BaseTemporalFeature): 
    def __call__(self, index): 
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(BaseTemporalFeature): 
    def __call__(self, index): 
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def get_temporal_features(dates, timeenc=1, freq='h'): 
    """
    Input
    ----------
    dates
        pd.DataFrame with a 'dates' column
    timeenc
        Different encoding schemes (e.g. whether or not normalize)
    freq
        If `timeenc == 0`
            * m - [month]
            * w - [month]
            * d - [month, day, weekday]
            * b - [month, day, weekday]
            * h - [month, day, weekday, hour]
            * t - [month, day, weekday, hour, *minute]
        If `timeenc == 1`
            * Q - [month]
            * M - [month]
            * W - [Day of month, week of year]
            * D - [Day of week, day of month, day of year]
            * B - [Day of week, day of month, day of year]
            * H - [Hour of day, day of week, day of month, day of year]
            * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
            * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]
    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into
    """
    if timeenc == 0: 
        freq_map = {
            'y':[], 
            'm':['month'], 
            'w':['month'], 
            'd':['month', 'day', 'weekday'], 
            'b':['month', 'day', 'weekday'], 
            'h':['month', 'day', 'weekday', 'hour'], 
            't':['month', 'day', 'weekday', 'hour', 'minute']
        }
        dates['month'] = dates['date'].apply(lambda row: row.month, True)
        dates['day'] = dates['date'].apply(lambda row: row.day, True)
        dates['weekday'] = dates['date'].apply(lambda row: row.weekday(), True)
        dates['hour'] = dates['date'].apply(lambda row: row.hour, True)
        dates['minute'] = dates['date'].apply(lambda row: row.minute, True)
        dates['minute'] = dates['minute'].map(lambda x: x // 15)
        freq = freq.lower()
        if freq in freq_map: 
            return dates[freq_map[freq]].values
        else: 
            RuntimeError('Unsupported frequency flag: {}'.format(freq))

    if timeenc == 1: 
        features_by_offsets = {
            offsets.YearEnd: [],
            offsets.QuarterEnd: [MonthOfYear],
            offsets.MonthEnd: [MonthOfYear],
            offsets.Week: [DayOfMonth, WeekOfYear],
            offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
            offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
            offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
            offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
            offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        }
        offset = to_offset(freq)
        for offset_type, feature_classes in features_by_offsets.items(): 
            if isinstance(offset, offset_type): 
                features = [cls() for cls in feature_classes]
                dates = pd.to_datetime(dates['date'].values)
                return np.array([feat(dates) for feat in features]).T
        raise RuntimeError('Unsupported frequency flag: {}'.format(freq))


class StandardScaler(): 
    def __init__(self): 
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data): 
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data): 
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data): 
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class BaseDataset(Dataset): 
    def __init__(self, mode, data_path, len_enc, len_label, len_pred, freq): 
        super().__init__()
        assert mode in {'train', 'dev', 'test'}
        self.mode = mode
        self.len_enc = len_enc
        self.len_label = len_label
        self.len_pred = len_pred
        self.freq = freq
        self._read_data(data_path)

    def __len__(self): 
        assert len(self.data) == len(self.time)
        return len(self.data) - self.len_enc - self.len_pred + 1

    def __getitem__(self, index): 
        enc_begin = index
        enc_end = enc_begin + self.len_enc
        dec_begin = max(enc_end - self.len_label, 0)
        dec_end = dec_begin + self.len_label + self.len_pred

        x = self.data[enc_begin:enc_end]
        x_time = self.time[enc_begin:enc_end]
        y = self.data[dec_begin:dec_end]
        y_time = self.time[dec_begin:dec_end]
        return {
            'x': x, 
            'x_time': x_time, 
            'y': y, 
            'y_time': y_time
        }


class ETTDataset(BaseDataset): 
    # ETTh, 1 hour per sample
    borders_h = [0, 12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
    # ETTm, 15 minutes per sample
    borders_m = [0, 12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]

    def _read_data(self, data_path): 
        # Read dataset
        m = re.fullmatch(r'.*ETT(.)\d?\.csv$', data_path)
        if m is None or m[1] not in {'h', 'm'}: 
            raise RuntimeError('Uninformative Filename')
        df = pd.read_csv(data_path)
        assert df.columns[0] == 'date'
        df_data = df.iloc[:, 1:]
        # df_time = df.iloc[:, :1]

        # Retrieve appropriate index
        borders = ETTDataset.borders_h if m[1] == 'h' else ETTDataset.borders_m
        train_start, train_end = borders[0], borders[1]
        dev_start, dev_end = borders[1] - self.len_enc, borders[2]
        test_start, test_end = borders[2] - self.len_enc, borders[3]
        if self.mode == 'train': 
            start, end = train_start, train_end
        elif self.mode == 'dev': 
            start, end = dev_start, dev_end
        elif self.mode == 'test': 
            start, end = test_start, test_end

        # Scale the data based on the training dataset
        self._scaler = StandardScaler()
        self._scaler.fit(df_data[train_start:train_end].values)
        self.data = self._scaler.transform(df_data[start:end].values)

        # Extract temporal features
        time = df.iloc[start:end, :1]
        time['date'] = pd.to_datetime(time['date'])
        # NOTE: For both ETTh and ETTm, always use timeenc = 1 and freq = 'h'
        self.time = get_temporal_features(time, timeenc=1, freq='h')


class OtherDataset(BaseDataset): 
    def _read_data(self, data_path): 
        # Read dataset
        df = pd.read_csv(data_path)
        assert 'date' in df.columns
        cols = list(df.columns)
        cols.remove('date')
        df = df[['date'] + cols]
        df_data = df.iloc[:, 1:]
        # df_time = df.iloc[:, :1]

        # Retrieve appropriate index
        n_train = int(len(df) * 0.7)
        n_test = int(len(df) * 0.2)
        n_dev = len(df) - n_train - n_test
        if n_dev < self.len_enc + self.len_pred: 
            # n_dev (int(len(df) * 0.2)) is too small
            n_test = n_dev = (n_test + n_dev) // 2
        borders = [0, n_train, n_train + n_dev, len(df)]
        train_start, train_end = borders[0], borders[1]
        dev_start, dev_end = borders[1] - self.len_enc, borders[2]
        test_start, test_end = borders[2] - self.len_enc, borders[3]
        if self.mode == 'train': 
            start, end = train_start, train_end
        elif self.mode == 'dev': 
            start, end = dev_start, dev_end
        elif self.mode == 'test': 
            start, end = test_start, test_end

        # Scale the data based on the training dataset
        self._scaler = StandardScaler()
        self._scaler.fit(df_data[train_start:train_end].values)
        self.data = self._scaler.transform(df_data[start:end].values)

        # Extract temporal features
        time = df.iloc[start:end, :1]
        time['date'] = pd.to_datetime(time['date'])
        # NOTE: Always use timeenc = 1 and freq = 'h'
        self.time = get_temporal_features(time, timeenc=1, freq='h')


class SyntheticDataset(BaseDataset): 
    def _read_data(self, data_path): 
        # Read dataset
        assert os.path.basename(data_path).endswith('.npy')
        data = np.load(data_path, allow_pickle=True)

        # Retrieve appropriate index
        n_train = int(len(data) * 0.7)
        n_test = int(len(data) * 0.2)
        n_dev = len(data) - n_train - n_test
        borders = [0, n_train, n_train + n_dev, len(data)]
        train_start, train_end = borders[0], borders[1]
        dev_start, dev_end = borders[1] - self.len_enc, borders[2]
        test_start, test_end = borders[2] - self.len_enc, borders[3]
        if self.mode == 'train': 
            start, end = train_start, train_end
        elif self.mode == 'dev': 
            start, end = dev_start, dev_end
        elif self.mode == 'test': 
            start, end = test_start, test_end

        # Scale the data based on the training dataset
        self._scaler = StandardScaler()
        self._scaler.fit(data[train_start:train_end])
        self.data = self._scaler.transform(data[start:end])

        # No temporal features
        self.time = np.empty((len(self.data), 0))
