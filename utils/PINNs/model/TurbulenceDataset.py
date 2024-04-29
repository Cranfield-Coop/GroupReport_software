from torch.utils.data import Dataset
import pandas as pd
import torch


class TurbulenceDataset(Dataset):
    """
    Dataset class for the Turbulence model. This class loads the dataset from a CSV file and preprocesses it.

    Args:
        csv_file (string): Path to the csv file with the dataset.
        phase (string): The phase of the dataset (train, val, test, or predict).
    """

    def __init__(self, csv_file, phase="train"):
        # Load the dataset
        df = pd.read_csv(csv_file)

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
