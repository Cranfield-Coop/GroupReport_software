from tqdm import tqdm
import torch
import lightning as L
import csv
import numpy as np

# Importing modules for data handling and the neural network model
from utils.pinns.TurbulenceModel.TurbulenceDataModule import TurbulenceDataModule
from utils.pinns.TurbulenceModel.TurbulenceModelPINN import TurbulenceModelPINN


def run_test_inference(
    noise, model_checkpoint_path, test_dataset_path, prediction_output_path
):
    """
    Runs inference on the test dataset using the specified model checkpoint and saves predictions.

    Args:
        model_checkpoint_path (str): Path to the trained model checkpoint.
        prediction_dataset_path (str): Path to the prediction dataset CSV file.
        prediction_output_path (str): Path to save the prediction output CSV file.
    """
    # Initialize the data module with the test dataset path
    data_module = TurbulenceDataModule(
        batch_size=64,  # Adjust based on your system's memory capacity
        num_workers=0,  # Adjust based on your system's capabilities
        test_dataset_path=test_dataset_path,
    )

    # Load the model from the checkpoint
    model = TurbulenceModelPINN.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path,
    )

    # Prepare the data module specifically for testing
    data_module.setup(noise, stage="test")

    # Initialize a PyTorch Lightning trainer
    trainer = L.Trainer(
        accelerator="cpu",  # Use CPU for training
        devices="auto",  # Automatically select devices
    )

    # Compute the MSE over the test dataset
    metrics = trainer.test(model, datamodule=data_module)  # Test the model

    # Open a file to save the predictions
    with open(prediction_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(
            [
                "y/delta",
                "y^+",
                "u_tau",
                "nu",
                "Re_tau",
                "u'u'_target",
                "v'v'_target",
                "w'w'_target",
                "u'v'_target",
                "U_target",
                "dU/dy_target",
                "P_target",
                "k_target",
                "u'u'_pred",
                "v'v'_pred",
                "w'w'_pred",
                "u'v'_pred",
                "U_pred",
                "dU/dy_pred",
                "P_pred",
                "k_pred",
            ]
        )

        # Evaluate the model on the test dataset
        with torch.inference_mode():
            for batch in tqdm(
                data_module.test_dataloader(), desc="Inference in Progress"
            ):
                features = batch["features"]
                targets = batch["targets"]
                outputs = model(features)

                # Convert tensors to numpy arrays for easier manipulation and writing
                features_np = features.cpu().numpy()
                targets_np = targets.cpu().numpy()
                outputs_np = outputs.cpu().numpy()

                # Iterate over the batch and write each instance to the CSV
                for feature, target, output in zip(features_np, targets_np, outputs_np):
                    # Combine the feature and output arrays for writing
                    row = np.concatenate((feature, target, output)).tolist()
                    writer.writerow(row)

        return metrics


if __name__ == "__main__":
    run_test_inference(
        model_checkpoint_path="epoch=20332-step=731988.ckpt",  # Trained model
        # input csv path generated by prepare_prediction_csv method
        test_dataset_path="/Users/alexis/Cranfield/Group Project/Repos1/TurbulenceModelPINN/data/channel/test_dataset.csv",
        prediction_output_path="test_prediction_5200_output.csv",  # output csv path
    )
