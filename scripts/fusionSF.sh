#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


python main.py \
experiment=fusionsf_3modal \
datamodule.dataset.num_sites=10 \
datamodule.dataset.num_ignored_sites=0 \
datamodule.batch_size=16 \
seed=42 \
resume=False \
task_name=fusionSF_train10_test10 \
pl_module.model.ts_masking_ratio=0 \
pl_module.model.ctx_masking_ratio=0.99 \
pl_module.model.vq_in_ts=True \
pl_module.model.vq_in_ctx=True \
pl_module.model.vq_in_guide=False \
trainer.strategy=ddp \
trainer.max_epochs=100 \
callbacks.early_stopping.patience=20 \
logger.wandb.group=fusion_3m

