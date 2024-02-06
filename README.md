<div align="center">
  
# FusionSF: Fuse Heterogeneous Modalities in a Vector Quantized Framework for Robust Solar Power Forecasting

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/) 
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/gitbooo/TSF_context_Eumetsat/blob/neurips_2023/README.md#license)
 
</div>

## Description

This is the official repository to the paper ["FusionSF: Fuse Heterogeneous Modalities in a Vector Quantized Framework for Robust Solar Power Forecasting"](https://arxiv.org/) by **Ziqing Ma**\*, **Wenwei Wang**\*, **Tian Zhou**\*, Chao Chen, Bingqing Peng, Liang Sun and Rong Jin.
(* equal contribution)




[//]: # (## Citation)

[//]: # (If you use this codebase, or otherwise found our work valuable, please cite CrossViVit)

[//]: # ()
[//]: # (```)

[//]: # (@article{boussif2023enrich,)

[//]: # (  title   = {What if We Enrich day-ahead Solar Irradiance Time Series Forecasting with Spatio-Temporal Context?},)

[//]: # (  author  = {Oussama Boussif and Ghait Boukachab and Dan Assouline and Stefano Massaroli and Tianle Yuan and Loubna Benabbou and Yoshua Bengio},)

[//]: # (  year    = {2023},)

[//]: # (  journal = {arXiv preprint arXiv: 2306.01112})

[//]: # (})

[//]: # (```)

## Dataset
You can access the dataset as follows:

Folder: https://drive.google.com/drive/folders/1qGVOw-hAVQlO3n-1d4ZNHvL42L9PkdBK?usp=drive_link

Zip File: https://drive.google.com/file/d/18Y-kwNUkT9t5EBugWxtDoE5HnMfaFG2D/view?usp=drive_link

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/MAZiqing/FuisionSF.git
cd FuisionSF

# [OPTIONAL] create conda environment
conda create -n MyEnvName python=3.10
conda activate MyEnvName

# install requirements
pip install -r requirements.txt
```
## Experiments
To help the users reproduce our results, we released the sbatch scripts that we used.
 - FusionSF (145M): ``scripts/fusionSF.sh``

[//]: # (## Hyperparameter tuning:)

[//]: # (We use [orion]&#40;https://github.com/Epistimio/orion&#41; to optimize hyperparameters and it's well suited for launching distributed hyperparameter optimization on clusters. It also integrates nicely with pytorch-lightning as well as hydra through their hydra plugin, so make sure to check their repo if you want more information !)

[//]: # ()
[//]: # (You can launch the hyperparameter optimization using the following command:)

[//]: # (```)

[//]: # (CUDA_VISIBLE_DEVICES=0 python main.py -m hparams_search=[replace_with_model_to_be_tuned] experiment=[replace_with_model_to_be_tuned] seed=42 resume=True)

[//]: # (```)

[//]: # ()
[//]: # (We attached a sbatch script for optimizing CrossViViT's hyperparameters that you can find here: ``sbatch_scripts/crossvivit_tuning.sh``)

[//]: # (## Baselines)

[//]: # ()
[//]: # (In addition to the main contributions presented in the paper, this repository also includes the implementation of several baseline models. These baselines serve as reference models or starting points for comparison and evaluation.)

[//]: # ()
[//]: # (The following baseline models are included:)

[//]: # ()
[//]: # (  -  **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]]&#40;https://arxiv.org/pdf/2205.13504.pdf&#41;)

[//]: # (  -  **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]]&#40;https://arxiv.org/abs/2207.01186&#41;)

[//]: # (  -  **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]]&#40;https://arxiv.org/abs/2012.07436&#41; )

[//]: # (  -  **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]]&#40;https://arxiv.org/abs/2001.04451&#41;)

[//]: # (  -  **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]]&#40;https://arxiv.org/abs/2106.13008&#41;)

[//]: # (  -  **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]]&#40;https://arxiv.org/abs/2201.12740&#41; )

[//]: # (  -  **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]]&#40;https://openreview.net/forum?id=vSVLM2j9eie&#41;)

[//]: # (  -  **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [[ICLR 2023]]&#40;https://arxiv.org/abs/2211.14730&#41;)

[//]: # (  -  **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]]&#40;https://arxiv.org/abs/2205.08897&#41;)
  
## License

CrossViVit is licensed under the MIT License.

```
MIT License

Copyright (c) (2023) Ghait Boukachab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
