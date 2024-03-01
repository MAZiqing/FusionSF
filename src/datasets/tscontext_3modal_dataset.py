import json
from typing import Dict, List, Tuple, Union

import os
# import deeplake
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange, repeat


def get_data_spower(data_dir, solar_power_file, num_sites=10, num_ignored_sites=0):
    """
    Parameters
    -----------
    data_dir : str
            Absolute directory path where dataset is located.
    solar_power_file : str
            Relative directory path where solar power data is located in data_dir.
    num_sites : int
            number of power sites to be employed in the dataset, max=88
    num_ignored_sites : int
            number of power sites ignored, so the sites employed will be [num_ignored_sites: num_sites]

    Returns
    -----------
    data_sp : tensor
            solar power data tensor with shape [N, d]
    time : tensor
            timestamp tensor (month, day, hour) with shape [N, 3]
    time_dt : pd.Series
            timestamp series
    length : int
            dataset length
    """
    data_df = pd.read_csv(os.path.join(data_dir, solar_power_file),
                          parse_dates=['datetime'])
    data_df = data_df.fillna(0)
    data_sp = []
    for i, (k, v) in enumerate(data_df.groupby('site')):
        if i >= num_sites:
            break
        if i < num_ignored_sites:
            continue
        print('dataset k,v len', k, len(v))
        lats_lons = torch.tensor([v['lat'].values[0], v['lon'].values[0]])
        # v[v['power'] < 0]['power'] = 0
        values = torch.from_numpy(v['power'].values)
        if values.isnan().any():
            print('dataset contains nan len', k, len(v))
        data_sp += [{'site': k, 'values': values, 'lats_lons': lats_lons}]
    length = len(v)
    month = v['datetime'].apply(lambda x: x.month)
    day = v['datetime'].apply(lambda x: x.day)
    hour = v['datetime'].apply(lambda x: x.hour)
    time = torch.from_numpy(np.stack([month, day, hour], axis=0))
    time_dt = v['datetime']
    return data_sp, time, time_dt, length


def get_data_satellite(data_dir, satellite_dir, norm_stl=True):
    """
    Parameters
    -----------
    data_dir : str
            Absolute directory path where dataset is located.
    satellite_dir : str
            Relative directory path where satellite images is located in data_dir.
    """
    print('load satellite from: [{}]'.format(os.path.join(data_dir, satellite_dir)))
    array_satellite = np.load(os.path.join(data_dir, satellite_dir, 'satellite.npy'))
    T, H, W, C = array_satellite.shape
    if norm_stl:
        scaler = StandardScaler()
        array_satellite = scaler.fit_transform(array_satellite.reshape(T, -1)).reshape(T, H, W, C)
    data_satellite = torch.from_numpy(array_satellite)
    data_satellite_coords = torch.from_numpy(
        np.load(os.path.join(data_dir, satellite_dir, 'satellite_coords.npy')))
    data_satellite_times = np.load(os.path.join(data_dir, satellite_dir, 'satellite_times.npy'))
    return data_satellite, data_satellite_times, data_satellite_coords


def get_data_nwp(data_dir, nwp_file, norm_nwp=True):
    """
    Parameters
    -----------
    data_dir : str
            Absolute directory path where dataset is located.
    nwp_file : str
            Relative directory path where the numerical weather prediction (nwp) file is located in data_dir.
    """
    df = pd.read_csv(os.path.join(data_dir, nwp_file),
                     parse_dates=['fcst_date']).interpolate()
    # get nwp start time
    nwp_start_time = df['fcst_date'].iloc[0]

    # process nwp dataframe
    df['lat'] = np.round(df['lat'], 1)
    df['lon'] = np.round(df['lon'], 1)
    df = df.drop(columns=['fcst_date'])
    # normalize nwp dataframe
    columns = df.columns.drop(['lat', 'lon'])
    if norm_nwp:
        scaler = StandardScaler()
        df[columns] = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
    data_nwp_grouped = df.groupby(['lat', 'lon'])
    return data_nwp_grouped, nwp_start_time


class Ts3MDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        satellite_dir: str = 'satellite',
        seq_len: int = 24 * 2,
        label_len: int = 12,
        pred_len: int = 24 * 2,
        num_sites: int = 10,
        num_ignored_sites: int = 0,
        norm_nwp: bool = True,
        norm_stl: bool = True,
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
                            solar_power_file='solar_power/solar_power.csv',
                            num_sites=num_sites,
                            num_ignored_sites=num_ignored_sites))

        self.data_stl, self.data_stl_times, self.data_stl_coords = (
            get_data_satellite(data_dir=data_dir, satellite_dir=satellite_dir, norm_stl=norm_stl))

        self.data_ec_grouped, self.ec_start_time = (
            get_data_nwp(data_dir=data_dir, nwp_file='nwp/nwp.csv', norm_nwp=norm_nwp))

        print('ts context nwp 3 modal dataset prepared, num_sites={}, sites_ignored={}'.format(num_sites,
                                                                                               num_ignored_sites))

    def __len__(self) -> int:
        return (self.data_sp_length - self.seq_len - self.pred_len + 1) * (self.num_sites - self.num_ignored_sites)

    def __getitem__(self, idx: int):
        site_id = idx // (self.data_sp_length-self.seq_len-self.pred_len+1)

        # get index
        x_begin_index = idx % (self.data_sp_length-self.seq_len-self.pred_len+1)
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index
        y_end_index = y_begin_index + self.pred_len

        stl_begin_index = (self.data_sp_time_dt.iloc[x_begin_index] - pd.to_datetime(self.data_stl_times[0]))
        stl_begin_index = int(stl_begin_index.total_seconds() // 3600)
        stl_end_index = stl_begin_index + self.seq_len

        ec_begin_index = (self.data_sp_time_dt.iloc[x_begin_index] - self.ec_start_time)
        ec_begin_index = int(ec_begin_index.total_seconds() // 3600) + self.seq_len
        ec_end_index = ec_begin_index + self.pred_len

        # get data
        stl_input = rearrange(self.data_stl[stl_begin_index: stl_end_index], 't h w c -> t c h w')
        stl_time = self.data_stl_times[stl_begin_index: stl_end_index]
        stl_coords = rearrange(self.data_stl_coords, 'h w c -> c h w')
        T, C, H, W = stl_input.shape
        
        # print('data_sp_len', self.data_sp_length)
        ts_input = self.data_sp[site_id]['values'][x_begin_index: x_end_index].unsqueeze(-1)
        ts_target = self.data_sp[site_id]['values'][y_begin_index: y_end_index]  #.unsqueeze(-1)
        ts_time = repeat(self.data_sp_time[:, x_begin_index: x_end_index], 'c t -> t c h w', h=H, w=W)
        ts_time_dt = self.data_sp_time_dt.iloc[x_begin_index: x_end_index]
        ts_coords = self.data_sp[site_id]['lats_lons'].unsqueeze(-1).unsqueeze(-1)

        lat = np.round(float(ts_coords[0, 0, 0]), 1)
        lon = np.round(float(ts_coords[1, 0, 0]), 1)
        ec_input = self.data_ec_grouped.get_group((lat, lon)).values
        ec_input = torch.from_numpy(ec_input[ec_begin_index: ec_end_index])

        return_tensors = {
            'ts_input': ts_input,  # (torch.Tensor): Station timeseries of shape [T, C2]
            'ts_target': ts_target,  # (torch.Tensor): Target station timeseries of shape [T, C2]
            'ts_time': ts_time,  # (torch.Tensor): Time coordinates of shape [T, C3, H, W]
            'ts_coords': ts_coords,  # (torch.Tensor): Station coordinates of shape [2, 1, 1]
            'stl_input': stl_input,  # (torch.Tensor): Satellite image (stl) Context frames of shape [T, C1, H, W]
            'stl_coords': stl_coords,  # (torch.Tensor): Coordinates of context frames of shape [2, H, W]
            'ec_input': ec_input  # # (torch.Tensor): nwp Context frames of shape [T, C4]
        }
        return return_tensors

