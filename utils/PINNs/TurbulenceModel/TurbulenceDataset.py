from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import numpy as np


class TurbulenceDataset(Dataset):
    """
    Dataset class for the Turbulence model.

    Args:
        csv_file (string): Path to the csv file with the dataset.
        phase (string): The phase of the dataset (train, val, or test).
    """

    def __init__(self, csv_file, noise=0, phase="train"):
        # Load the dataset and add noise
        df = pd.read_csv(csv_file)
        if noise > 0:
            df_noise = df.drop(columns='Re_tau').applymap(
                lambda x: x + np.random.normal(0, noise/100))
            df_noise['Re_tau'] = df['Re_tau']
            df = df_noise.copy()

        # Model phase
        self.phase = phase

        # Select features and target
        self.features = df[
            [
                "y/delta",
                "y^+",
                "u_tau",
                "nu",
                "Re_tau"
            ]
        ].values

        # Standardize the features
        # self.scaler = StandardScaler()
        # self.features = self.scaler.fit_transform(self.features)

        # Convert the features to PyTorch tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)

        # Select targets if not in the predict phase
        if phase != "predict":
            # Select the target variables
            self.targets = df[
                [
                    "u'u'",
                    "v'v'",
                    "w'w'",
                    "u'v'",
                    "U",
                    "dU/dy",
                    "P",
                    "k",
                ]
            ].values
            # Convert the targets to PyTorch tensor
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: The sample from the dataset.
        """
        if self.phase != "predict":
            return {
                "features": self.features[idx],
                "targets": self.targets[idx],
            }
        else:
            return {"features": self.features[idx]}
