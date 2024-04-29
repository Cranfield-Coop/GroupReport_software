import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

# Importing necessary modules from the custom TurbulenceModel package and utilities
from model.TurbulenceDataModule import TurbulenceDataModule
from model.TurbulenceModelPINN import TurbulenceModelPINN

# Main execution block
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    torch.manual_seed(123)

    # Model initialization with specified architecture parameters
    batch_size = 32 # Number of samples per batch
    train_dataset_path = "data/channel/train_dataset.csv"  # Training dataset
    val_dataset_path = "data/channel/val_dataset.csv"  # Validation dataset
    test_dataset_path = "data/channel/test_dataset.csv"  # Test dataset
    input_dim = 5  # Dimensionalityx of input features
    output_dim = 5  # Dimensionality of the model's output
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
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        test_dataset_path=test_dataset_path,
        batch_size=batch_size,  # Number of samples per batch
        num_workers=8,  # Number of subprocesses for data loading
    )

    # Number of time steps for cosine annealing
    data_module.setup("fit")  # Prepare data for the fitting process

    # Model initialization with specified architecture parameters
    model = TurbulenceModelPINN(
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

    # Logger setup for TensorBoard
    logger = TensorBoardLogger("tb_logs", name="ChannelTurbulenceModelPINN")

    # Early stopping and checkpointing callbacks
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="val_mse"),
    ]

    # Trainer initialization with configurations for training process
    trainer = L.Trainer(
        max_epochs=max_epochs,  # Maximum number of epochs for training
        accelerator="cpu",  # Specifies the training will be on CPU
        devices="auto",  # Automatically selects the available devices
        logger=logger,  # Integrates the TensorBoard logger for tracking experiments
        callbacks=callbacks,  # Adds the specified callbacks to the training process
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
    )

    # Training phase
    trainer.fit(
        model,
        datamodule=data_module,
    )  # Start training the model

    # Testing phase
    data_module.setup("test")  # Prepare data for testing
    trainer.test(model, datamodule=data_module)  # Evaluate the model on the test set
