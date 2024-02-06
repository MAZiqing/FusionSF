import json
from typing import Dict, List, Tuple, Union
import pdb

import os
import deeplake
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange, repeat
from src.datasets.tscontext_3modal_dataset import get_data_spower, get_data_satellite


class TsContext2MDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 24,
        label_len: int = 12,
        pred_len: int = 24,
        num_sites: int = 10,
        num_ignored_sites: int = 0,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        data_dir : str
            Absolute directory path where the deeplake dataset is located.
        seq_len: int
            Number of frames in the input sequence.
        label_len: int
            Number of frames in the label sequence.
        pred_len: int
            Number of frames in the prediction sequence.
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
        self.data_dir = './data'

        self.data_sp, self.data_sp_time, self.data_sp_time_dt, self.data_sp_length = (
            get_data_spower(data_dir=data_dir,
                            solar_power_file='solar_power/shandong_solar_power_processed3.csv',
                            num_sites=num_sites,
                            num_ignored_sites=num_ignored_sites))

        self.data_stl, self.data_stl_times, self.data_stl_coords = (
            get_data_satellite(data_dir=data_dir, satellite_dir='satellite'))

        print('ts context 2 modal dataset prepared, num_sites={}, sites_ignored={}'.format(
            num_sites, num_ignored_sites))

    def __len__(self) -> int:
        return (self.data_sp_length - self.seq_len - self.pred_len + 1) * (self.num_sites - self.num_ignored_sites)

    def __getitem__(self, idx: int):
        site_id = idx // (self.data_sp_length-self.seq_len-self.pred_len+1)

        x_begin_index = idx % (self.data_sp_length-self.seq_len-self.pred_len+1)
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index
        y_end_index = y_begin_index + self.pred_len

        stl_begin_index = (self.data_sp_time_dt.iloc[x_begin_index] - pd.to_datetime(self.data_stl_times[0]))
        stl_begin_index = int(stl_begin_index.total_seconds() // 3600)
        stl_end_index = stl_begin_index + self.seq_len

        stl_input = rearrange(self.data_stl[stl_begin_index: stl_end_index], 't h w c -> t c h w')
        stl_time = self.data_stl_times[stl_begin_index: stl_end_index]
        stl_coords = rearrange(self.data_stl_coords, 'h w c -> c h w')
        T, C, H, W = stl_input.shape
        
        # print('data_sp_len', self.data_sp_length)
        # print('index', idx, x_begin_index, x_end_index)
        # print('site_id', site_id)
        # print('data_sp len', len(self.data_sp[site_id]['values']))
        ts_input = self.data_sp[site_id]['values'][x_begin_index: x_end_index].unsqueeze(-1)
        ts_target = self.data_sp[site_id]['values'][y_begin_index: y_end_index] #.unsqueeze(-1)
        ts_time = repeat(self.data_sp_time[:, x_begin_index: x_end_index], 'c t -> t c h w', h=H, w=W)
        ts_time_dt = self.data_sp_time_dt.iloc[x_begin_index: x_end_index]
        ts_coords = self.data_sp[site_id]['lats_lons'].unsqueeze(-1).unsqueeze(-1)
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
            'ts_time': ts_time,
            'ts_coords': ts_coords,
            'stl_input': stl_input,
            'stl_coords': stl_coords
        }

        return return_tensors
