from abc import ABC, abstractmethod
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


class ContextMixerModule(ABC, pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion"])

        self.model = model
        self.criterion = criterion

        self._set_metrics()

    def _set_metrics(self):
        if self.hparams.metrics.train is not None:
            for k in self.hparams.metrics.train:
                setattr(self, f"train_{k}", self.hparams.metrics.train[k])
        if self.hparams.metrics.val is not None:
            for k in self.hparams.metrics.val:
                setattr(self, f"val_{k}", self.hparams.metrics.val[k])

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def calc_dot_product(
        self,
        optflow_data: torch.Tensor,
        coords: torch.Tensor,
        station_coords: torch.Tensor,
    ):
        """
        Args:
            optflow_data: Optical flow data. Tensor of shape [B, T, 2*C, H, W]
            coords: Coordinates of each pixel. Tensor of shape [B, 2, H, W]
            station_coords: Coordinates of the station. Tensor of shape [B, 2, 1, 1]
        Returns:
            dp: Dot product between the optical flow and station vectors of shape [B, T, C, H, W]
        """
        optflow_data = rearrange(optflow_data, "b t (c n) h w -> b t c n h w", n=2)
        dist = station_coords - coords
        dist = repeat(
            dist,
            "b n h w -> b t c n h w",
            t=optflow_data.shape[1],
            c=optflow_data.shape[2],
        )
        optflow_data = F.normalize(
            rearrange(optflow_data, "b t c n h w -> b t c h w n"), p=2, dim=-1
        )
        dist = F.normalize(rearrange(dist, "b t c n h w -> b t c h w n"), p=2, dim=-1)
        dp = optflow_data[..., 0] * dist[..., 0] + optflow_data[..., 1] * dist[..., 1]
        return dp

    def prepare_batch(self, batch, use_target=True):
        # return_tensors = {
        #     'ts_input': ts_input,
        #     'ts_target': ts_target,
        #     'ts_time': ts_time,
        #     'ts_coords': ts_coords,
        #     'stl_input': stl_input,
        #     # 'stl_time': stl_time,
        #     'stl_coords': stl_coords
        # }
        x_ts = batch['ts_input'].float()
        x_ctx = batch['stl_input'].float()
        y_ts = batch['ts_target'].unsqueeze(-1).float()
        y_previous_ts = torch.randn_like(x_ts).float()
        ctx_coords = batch['stl_coords'].float()
        ts_coords = batch['ts_coords'].float()
        time_coords = batch['ts_time'].float()

        if use_target:
            return x_ts, x_ctx, y_ts, y_previous_ts, ctx_coords, ts_coords, time_coords
        return x_ts, x_ctx, ctx_coords, ts_coords, time_coords

    @abstractmethod
    def training_step(self, train_batch, batch_idx) -> Any:
        pass

    @abstractmethod
    def validation_step(self, val_batch, batch_idx) -> Any:
        pass
