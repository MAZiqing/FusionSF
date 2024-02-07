import pyrootutils
import os
os.environ['SLURM_JOB_ID'] = '1'
import torch
import numpy as np

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Optional, Tuple
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
# from src.datamodules.tscontext_3modal_datamodule import Ts3MDataModule

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.pl_module._target_}>")
    pl_module: LightningModule = hydra.utils.instantiate(cfg.pl_module)

    # test code
    datamodule.setup()
    tensor = datamodule.data_train.__getitem__(0)
    for k, v in tensor.items():
        tensor[k] = torch.cat([v.unsqueeze(0)]*7, dim=0)
    pl_module.training_step(tensor, 0)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    # for fig in pl_module.figs:
        # logger[0].experiment.log({"chart": fig})

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, inference_mode=False
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": pl_module,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None and cfg.resume == True:
            ckpt_path_ = Path(cfg.paths.output_dir) / "checkpoints" / "last.ckpt"
            if ckpt_path_.exists():
                ckpt_path = ckpt_path_
                print("resuming from", ckpt_path)
        trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    train_metrics = trainer.callback_metrics
    
    with torch.inference_mode(mode=False):
        if cfg.get("test"):
            log.info("Starting testing!")
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            print('before valid =======================')
            trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path)
            print('after valid =======================')
            log.info(f"Best ckpt path: {ckpt_path}")
    
    for k, v in pl_module.out_dict.items():
        save_dir = os.path.join(logger[0].save_dir, cfg.pl_module.model._target_)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        v = np.concatenate(v, axis=0)
        print('Save output, shape is: ', v.shape)
        np.save(os.path.join(save_dir, k + '.npy'), v)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    print('metric_dict is:', metric_dict)

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
