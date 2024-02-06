import json
from typing import Dict, List, Tuple, Union

import os
import deeplake
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.datasets.tscontext_3modal_dataset import get_data_spower


class TsDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 24,
        label_len: int = 12,
        pred_len: int = 24,
        num_sites: int = 10,
        num_ignored_sites: int = 0
    ) -> None:

        """
        Parameters
        ----------
        data_dir : str
            Absolute directory path where the deeplake dataset is located.
        seq_len : int
            Length of the input sequence.
        label_len : int
            Length of the label sequence.
        pred_len : int
            Length of the prediction sequence.
        """

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.n_samples = []
        self.year_mapping = {}

        # mzq
        self.num_sites = num_sites
        self.num_ignored_sites = num_ignored_sites

        self.data_sp, self.data_sp_time, self.data_sp_time_dt, self.data_sp_length = (
            get_data_spower(data_dir=data_dir, solar_power_file='solar_power/shandong_solar_power_processed3.csv'))

        print('TS only dataset prepared, num_sites={}, sites_ignored={}'.format(num_sites, num_ignored_sites))

    def __len__(self) -> int:
        return (self.data_sp_length - self.seq_len - self.pred_len + 1) * (self.num_sites - self.num_ignored_sites)

    def __getitem__(self, idx: int):
        site_id = idx // (self.data_sp_length - self.seq_len - self.pred_len + 1)
        x_begin_index = idx % (self.data_sp_length - self.seq_len - self.pred_len + 1)
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index
        y_end_index = y_begin_index + self.pred_len

        ts_input = self.data_sp[site_id]['values'][x_begin_index: x_end_index].unsqueeze(-1)
        ts_target = self.data_sp[site_id]['values'][y_begin_index: y_end_index]  # .unsqueeze(-1)
        ts_time_x = self.data_sp_time[:, x_begin_index: x_end_index].transpose(-1, -2)
        ts_time_y = self.data_sp_time[:, y_begin_index: y_end_index].transpose(-1, -2)
        ts_time_dt = self.data_sp_time_dt.iloc[x_begin_index: x_end_index]
        ts_coords = self.data_sp[site_id]['lats_lons']
        # print('dataset getitem', ts_input.shape, ts_target.shape)
        """
        Args:
            ctx (torch.Tensor): Context frames of shape [B, T, C, H, W]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
            mask (bool): Whether to mask or not. Useful for inference
        """
        return_tensors = {
            'ts_input': ts_input,
            'ts_target': ts_target,
            'ts_time_x': ts_time_x,
            'ts_time_y': ts_time_y,
            'ts_coords': ts_coords,
        }
        # print('dataset', '='*10)
        # print('idx, x_begin_index', idx, x_begin_index)
        # print('ts_input', ts_input.shape)
        # print('ts_target', ts_target.shape)
        # print('ts_time', ts_time.shape)
        # print('stl_input', stl_input.shape)
        # print('stl_coords', stl_coords.shape)

        return return_tensors