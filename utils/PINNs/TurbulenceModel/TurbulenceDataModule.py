import lightning as L
from torch.utils.data import DataLoader

from utils.PINNs.TurbulenceModel.TurbulenceDataset import TurbulenceDataset


class TurbulenceDataModule(L.LightningDataModule):
    """
    Data Module for the Turbulence Model

    Args:
        train_dataset_path: Path to the training dataset
        val_dataset_path: Path to the validation dataset
        test_dataset_path: Path to the test dataset
        batch_size: Batch size for the data loader
        num_workers: Number of workers for the data loader
    """

    def __init__(
        self,
        train_dataset_path=None,
        val_dataset_path=None,
        test_dataset_path=None,
        batch_size=8,
        num_workers=8,
    ):
        super().__init__()

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, noise=0, stage=None, predict_dataset_path=None):
        """
        Setup the data for the given stage

        Args:
            stage: Stage to setup the data for (fit, test, val)
        """
        if stage == "fit":
            self.train_dataset = TurbulenceDataset(
                self.train_dataset_path, noise, phase="train"
            )
            self.val_dataset = TurbulenceDataset(
                self.val_dataset_path, noise, phase="val")
        elif stage == "test":
            self.test_dataset = TurbulenceDataset(
                self.test_dataset_path, noise, phase="test")
        elif stage == "predict":
            self.predict_dataset = TurbulenceDataset(
                predict_dataset_path, noise, phase="predict"
            )

    def train_dataloader(self):
        """
        Returns the training data loader

        Returns:
            DataLoader: Training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns the validation data loader

        Returns:
            DataLoader: Validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Returns the test data loader

        Returns:
            DataLoader: Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
            # persistent_workers=True,
        )

    def predict_dataloader(self):
        """
        Returns the predict data loader

        Returns:
            DataLoader: Predict data loader
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
            # persistent_workers=True,
        )
