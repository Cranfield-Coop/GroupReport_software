from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Import modules for data handling, model, and utilities
from TurbulenceModel.TurbulenceDataModule import TurbulenceDataModule
from TurbulenceModel.TurbulenceModelPINN import TurbulenceModelPINN

# Main execution block
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    torch.manual_seed(123)

    # Model initialization with specified architecture parameters
    batch_size = 32 # Number of samples per batch
    test_dataset_path = "data/channel/test_dataset.csv"  # Test dataset
    input_dim = 5  # Dimensionalityx of input features
    output_dim = 8  # Dimensionality of the model's output
    hidden_dim = 128  # Size of the model's hidden layers
    hidden_depth = 4  # Number of hidden layers
    learning_rate = 1e-4  # Initial learning rate
    max_epochs = 100000  # Maximum number of training epochs
    activation = "elu"  # Activation function for hidden layers
    loss_phys_momentum_weight = 100  # Weight for the momentum loss term (physical loss)
    loss_phys_k_weight = 100  # Weight for the k loss term (physical loss)
    loss_bound_U_weight = 1  # Weight for the U loss term (boundary loss)
    loss_bound_dUdy_weight = 1  # Weight for the dUdy loss term (boundary loss)
    loss_bound_P_weight = 1  # Weight for the P loss term (boundary loss)
    loss_bound_stress_weight = 1  # Weight for the stress loss term (boundary loss)
    loss_bound_k_weight = 1  # Weight for the k loss term (boundary loss)
    # Data Module
    data_module = TurbulenceDataModule(
        test_dataset_path=test_dataset_path,
        batch_size=batch_size,  # Number of samples per batch
        num_workers=8,  # Number of subprocesses for data loading
    )

    # Number of time steps for cosine annealing
    data_module.setup("test")  # Prepare data for the fitting process

    # Versions of the model to be tested
    versions = [
        "/Users/alexis/Cranfield/Group Project/Repos1/TurbulenceModelPINN/tb_logs/ChannelTurbulenceModelPINN/version_355/checkpoints/epoch=58886-step=2119932.ckpt",
        "/Users/alexis/Cranfield/Group Project/Repos1/TurbulenceModelPINN/tb_logs/ChannelTurbulenceModelPINN/version_357/checkpoints/epoch=16290-step=586476.ckpt",
        "/Users/alexis/Cranfield/Group Project/Repos1/TurbulenceModelPINN/tb_logs/ChannelTurbulenceModelPINN/version_359/checkpoints/epoch=20332-step=731988.ckpt",
        "/Users/alexis/Cranfield/Group Project/Repos1/TurbulenceModelPINN/tb_logs/ChannelTurbulenceModelPINN/version_361/checkpoints/epoch=54851-step=1974672.ckpt",
    ]

    # Initialize a TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="ChannelTurbulenceModelPINN")

    # Initialize a PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,  # Maximum number of training epochs
        accelerator="cpu",  # Use CPU for training
        devices="auto",  # Automatically select devices
        logger=logger,  # No logger for the trainer
    )

    # Create a dataframe to store the results
    results = pd.DataFrame(
        columns=[
            "version",
            "mse_total",
            "rmse_total",
            "r2_total",
            "mse_U",
            "mse_dUdy",
            "mse_P",
            "mse_k",
            "mse_uu",
            "mse_vv",
            "mse_ww",
            "mse_uv",
            "r2_U",
            "r2_dUdy",
            "r2_P",
            "r2_k",
            "r2_uu",
            "r2_vv",
            "r2_ww",
            "r2_uv",
            "rmse_U",
            "rmse_dUdy",
            "rmse_P",
            "rmse_k",
            "rmse_uu",
            "rmse_vv",
            "rmse_ww",
            "rmse_uv",
        ]
    )

    for version in versions:
        # Model initialization with specified architecture parameters
        model = TurbulenceModelPINN.load_from_checkpoint(
            version,
            batch_size=batch_size,  # Number of samples per batch
            lr=learning_rate,  # Learning rate
            input_dim=input_dim,  # Dimensionality of input features
            hidden_dim=hidden_dim,  # Size of the model's hidden layers
            output_dim=output_dim,  # Dimensionality of the model's output
            hidden_depth=hidden_depth,  # Number of hidden layers
            activation=activation,  # Activation function for hidden layers
            loss_phys_momentum_weight=loss_phys_momentum_weight,  # Weight for the momentum loss term (physical loss)
            loss_phys_k_weight=loss_phys_k_weight,  # Weight for the k loss term (physical loss)
            loss_bound_U_weight=loss_bound_U_weight,  # Weight for the U loss term (boundary loss)
            loss_bound_dUdy_weight=loss_bound_dUdy_weight,  # Weight for the dUdy loss term (boundary loss)
            loss_bound_P_weight=loss_bound_P_weight,  # Weight for the P loss term (boundary loss)
            loss_bound_k_weight=loss_bound_k_weight,  # Weight for the k loss term (boundary loss)
            loss_bound_stress_weight=loss_bound_stress_weight,  # Weight for the stress loss term (boundary loss)
        )

        # Compute the metrics for the model
        metrics = trainer.test(model, datamodule=data_module)  # Test the model

        # Convert the metrics to a pandas DataFrame
        metrics = pd.DataFrame(metrics)

        # Concatenate the version with the metrics
        results = pd.concat([results, pd.concat([pd.DataFrame([version], columns=["version"]), metrics], axis=1)])

    # Save the results to a CSV file
    results.to_csv("metrics.csv", index=False)
