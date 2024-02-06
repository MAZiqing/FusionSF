#!/bin/bash
export SLURM_JOB_ID=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python main.py \
experiment=Informer \
datamodule.num_workers=4 \
datamodule.batch_size=64 \
seed=30 \
resume=False \
trainer.strategy=ddp \
pl_module.optimizer.lr=0.0002 \
trainer.max_epochs=50 \
logger.wandb.group=baseline \
callbacks.early_stopping.patience=10

python main.py \
experiment=Autoformer \
datamodule.num_workers=4 \
datamodule.batch_size=64 \
seed=42 \
resume=False \
trainer.strategy=ddp \
trainer.max_epochs=50 \
logger.wandb.group=baseline \
callbacks.early_stopping.patience=10

python main.py \
experiment=Crossformer \
datamodule.num_workers=4 \
datamodule.batch_size=64 \
seed=42 \
resume=False \
trainer.strategy=ddp \
trainer.max_epochs=50 \
logger.wandb.group=baseline \
callbacks.early_stopping.patience=10

python main.py \
experiment=Dlinear \
datamodule.num_workers=4 \
datamodule.batch_size=64 \
seed=42 \
resume=False \
trainer.strategy=ddp \
trainer.max_epochs=50 \
logger.wandb.group=baseline \
callbacks.early_stopping.patience=10

python main.py \
experiment=PatchTST \
datamodule.num_workers=4 \
datamodule.batch_size=16 \
pl_module.optimizer.lr=0.0002 \
seed=42 \
resume=False \
trainer.strategy=ddp \
trainer.max_epochs=50 \
logger.wandb.group=baseline \
callbacks.early_stopping.patience=20

python main.py \
experiment=LightTS \
datamodule.num_workers=4 \
datamodule.batch_size=64 \
seed=42 \
resume=False \
trainer.strategy=ddp \
trainer.max_epochs=50 \
logger.wandb.group=baseline \
callbacks.early_stopping.patience=10

python main.py \
experiment=FiLM \
datamodule.num_workers=4 \
datamodule.batch_size=64 \
seed=42 \
resume=False \
trainer.strategy=ddp \
trainer.max_epochs=50 \
logger.wandb.group=baseline \
callbacks.early_stopping.patience=10