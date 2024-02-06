from typing import Any, Dict, Optional
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from src.datasets.tscontext_3modal_dataset import Ts3MDataset


class Ts2MDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.2
    ):
        super().__init__()

        self.save_hyperparameters()
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_all = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        train_ratio = self.train_ratio
        valid_ratio = self.valid_ratio
        test_ratio = self.test_ratio
        if not self.data_train and not self.data_val and not self.data_test:
            data_all = Ts3MDataset(**self.hparams.dataset)
            data_len = len(data_all)
            all_indices = np.arange(0, int(data_len))
            all_indices = all_indices.reshape([data_all.num_sites - data_all.num_ignored_sites, -1])
            N, L = all_indices.shape
            train_indices = all_indices[:, :int(L * train_ratio)].reshape(-1)
            valid_indices = all_indices[:, int(L * train_ratio): int(L * (train_ratio + valid_ratio))].reshape(-1)
            test_indices = all_indices[:, -int(L * test_ratio):].reshape(-1)
            self.data_train = Subset(data_all, train_indices)
            self.data_val = Subset(data_all, valid_indices)
            self.data_test = Subset(data_all, test_indices)
            if self.hparams.dataset.get('dataset_test'):
                print('use a different test dataset')
                data_all_test = Ts3MDataset(**self.hparams.dataset.dataset_test)
                all_indices_test = np.arange(0, int(len(data_all_test)))
                all_indices_test = all_indices_test.reshape([data_all_test.num_sites - data_all_test.num_ignored_sites, -1])
                N, L = all_indices_test.shape
                test_indices = all_indices_test[:, -int(L * test_ratio):].reshape(-1)
                self.data_test = Subset(data_all_test, test_indices)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            # num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            # num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            # num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
