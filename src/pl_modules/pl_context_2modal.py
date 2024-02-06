import torch
import wandb
from torchmetrics import MeanMetric

from . import ContextMixerModule


class Pl2Modal(ContextMixerModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(model, optimizer, scheduler, metrics, criterion, kwargs=kwargs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.out_dict = {'inputs': [], 'outputs': [], 'targets': []}

    def forward(self, x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask):
        # out, _, self_attention_scores, cross_attention_scores = self.model(
        out = self.model(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask
        )
        return out

    def prepare_batch(self, batch, use_target=True):
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

    def training_step(self, train_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(train_batch)

        y_hat = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True)
        y_hat = y_hat.mean(dim=2)
        # print('y_hat', y_hat.shape, y_ts.shape)
        loss = self.criterion(y_hat, y_ts)
        # print('train loss', loss)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(val_batch)

        y_hat = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False)
        y_hat = y_hat.mean(dim=2)

        # print('y_hat valid', y_hat.shape, y_ts.shape)
        loss = self.criterion(y_hat, y_ts)
        # print('mzq, loss got')

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat.clone(), y_ts.clone(), y_prev_ts.clone())
            else:
                metric(y_hat.clone(), y_ts.clone())
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat.clone().requires_grad_(), "ground_truth": y_ts.clone().requires_grad_()}
    
    def test_step(self, val_batch, batch_idx):
        val_batch = {key: value.clone() for key, value in val_batch.items()}
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(val_batch)

        y_hat = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False)
        y_hat = y_hat.mean(dim=2)

        loss = self.criterion(y_hat, y_ts)

        self.out_dict['inputs'] += [x_ts.cpu().detach().numpy()]
        self.out_dict['outputs'] += [y_hat.cpu().detach().numpy()]
        self.out_dict['targets'] += [y_ts.cpu().detach().numpy()]

        self.val_loss(loss)
        self.log("test/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"test/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat, "ground_truth": y_ts}
