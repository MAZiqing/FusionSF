import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt


class TSForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict,
        criterion: torch.nn.Module,
        seq_len: int,
        label_len: int,
        pred_len: int,
        padding: int = 0,
        inverse_scaling: bool = False,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion"])
        self.model = model
        self.criterion = criterion
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self._set_metrics()
        # self.test_tables = wandb.Table(columns=['id', 'input', 'predict', 'target'])
        self.figs = []
        self.out_dict = {'inputs': [], 'outputs': [], 'targets': []}

    def _set_metrics(self):
        for k in self.hparams.metrics.train:
            setattr(self, f"train_{k}", self.hparams.metrics.train[k])
        for k in self.hparams.metrics.val:
            setattr(self, f"val_{k}", self.hparams.metrics.val[k])

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # if self.hparams.padding == 0:
        #     decoder_input = torch.zeros((batch_y.size(0), self.hparams.pred_len, batch_y.size(-1))).type_as(batch_y)
        # else:  # self.hparams.padding == 1
        #     decoder_input = torch.ones((batch_y.size(0), self.hparams.pred_len, batch_y.size(-1))).type_as(batch_y)
        # decoder_input = torch.cat([batch_y[:, : self.hparams.label_len, :], decoder_input], dim=1)
        decoder_input = batch_y.clone()
        decoder_input[:, :, 0] = 0
        outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        if self.hparams.output_attention:
            outputs = outputs[0]
        return outputs

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

    def prepare_batch(self, batch):
        ts_x = batch['ts_input'].float()
        ts_y = batch['ts_target'].unsqueeze(-1).float()
        ts_coords = batch['ts_coords'].float()
        ts_time_x = batch['ts_time_x'].float()
        ts_time_y = batch['ts_time_y'].float()

        ts_coords_x = repeat(ts_coords, 'b d -> b t d', t=self.model.seq_len)
        ts_coords_y = repeat(ts_coords, 'b d -> b t d', t=self.model.pred_len)

        ts_x = torch.cat([ts_x, ts_coords_x], dim=-1)
        ts_y = torch.cat([ts_y, ts_coords_y], dim=-1)

        return ts_x, ts_y, ts_time_x, ts_time_y

    def training_step(self, train_batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = self.prepare_batch(train_batch)
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        outputs = outputs[:, -self.model.pred_len:, 0]
        batch_y = batch_y[:, -self.model.pred_len:, 0]

        loss = self.criterion(outputs, batch_y)
        self.train_loss(loss)

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            metric(outputs, batch_y)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def add_a_plot(self, xs, ys, labels):
        fig, ax = plt.subplots()
        for x, y in zip(xs, ys):
            ax.plot(x, y)
        ax.legend(labels)
        return fig

    def validation_step(self, val_batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = self.prepare_batch(val_batch)
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        # print(outputs.shape)
        # print(batch_y.shape)
        outputs = outputs[:, -self.model.pred_len:, 0]
        batch_y = batch_y[:, -self.model.pred_len:, 0]

        loss = self.criterion(outputs, batch_y)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            metric(outputs, batch_y)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {'grountruth': batch_y, 'prediction': outputs}

    def test_step(self, test_batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = self.prepare_batch(test_batch)
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        # print(outputs.shape)
        # print(batch_y.shape)
        outputs = outputs[:, -self.model.pred_len:, 0]
        batch_y = batch_y[:, -self.model.pred_len:, 0]

        # x_in = range(0, self.model.seq_len)
        # x_out = range(self.model.seq_len, self.model.seq_len+self.model.pred_len)
        # xs = [x_in, x_out, x_out]
        # ys = [i.cpu().detach().numpy() for i in [batch_x[0, ..., 0], outputs[0], batch_y[0]]]
        # labels = ['input', 'predict', 'target']
        # fig = self.add_a_plot(xs, ys, labels)
        # self.figs += [fig]
        self.out_dict['inputs'] += [batch_x[..., 0].cpu().detach().numpy()]
        self.out_dict['outputs'] += [outputs.cpu().detach().numpy()]
        self.out_dict['targets'] += [batch_y.cpu().detach().numpy()]

        loss = self.criterion(outputs, batch_y)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            metric(outputs, batch_y)
            self.log(
                f"test/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {'grountruth': batch_y, 'prediction': outputs}