import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.PINNs.TurbulenceModel.NNModel import NNModel


# Define the PyTorch Lightning module for training the PINN
class TurbulenceModelPINN(L.LightningModule):
    """
    Implements a PyTorch Lightning module for training a Physics-Informed Neural Network (PINN)
    that models turbulence using the Navier-Stokes equations. This class integrates the PINNModel
    to simulate fluid dynamics and turbulence phenomena under various conditions.

    Attributes:
        lr (float): Learning rate for the Adam optimizer.
        input_dim (int): The number of input features (e.g., spatial coordinates, time).
        hidden_dim (int): The number of neurons in each hidden layer of the PINNModel.
        output_dim (int): The number of output variables predicted by the model (e.g., velocity, pressure).
        model (PINNModel): The instantiated physics-informed neural network model.
    """

    def __init__(
        self,
        batch_size=32,
        max_steps=100000,
        lr=1e-4,
        input_dim=4,
        hidden_dim=70,
        output_dim=8,
        hidden_depth=10,
        activation="elu",
        loss_phys_momentum_weight=1,
        loss_phys_k_weight=1,
        loss_bound_U_weight=1,
        loss_bound_dUdy_weight=1,
        loss_bound_P_weight=1,
        loss_bound_k_weight=1,
        loss_bound_stress_weight=1,
        rho=1.225,
    ):
        """
        Initializes the turbulence model with specific parameters for the underlying PINNModel and
        training configuration.

        Args:
            lr (float): Learning rate for the optimizer.
            input_dim (int): Number of input features for the model.
            hidden_dim (int): Number of neurons in each hidden layer.
            output_dim (int): Number of outputs from the model.
            hidden_depth (int): Number of hidden layers in the model.
            activation (str): Activation function used in hidden layers.
        """
        super(TurbulenceModelPINN, self).__init__()

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.lr = lr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_depth = hidden_depth
        self.activation = activation
        self.loss_phys_momentum_weight = loss_phys_momentum_weight
        self.loss_phys_k_weight = loss_phys_k_weight
        self.loss_bound_U_weight = loss_bound_U_weight
        self.loss_bound_dUdy_weight = loss_bound_dUdy_weight
        self.loss_bound_P_weight = loss_bound_P_weight
        self.loss_bound_k_weight = loss_bound_k_weight
        self.loss_bound_stress_weight = loss_bound_stress_weight
        self.rho = rho

        # Initialize the PINNModel with specified architecture and activation function
        self.model = NNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            hidden_depth=self.hidden_depth,
            activation=self.activation,
        )

        self.save_hyperparameters()  # Automatically logs and saves hyperparameters for reproducibility

    def forward(self, x):
        return self.model(x)

    def physics_informed_loss(self, predictions, x):
        """
        Computes a custom loss based on the deviation from the  Navier-Stokes equations,
        enabling the model to learn both from data and the underlying physics of fluid dynamics.

        Args:
            predictions (torch.Tensor): The model's predictions for fluid properties.
            x (torch.Tensor): The input features to the model, including physical parameters.

        Returns:
            torch.Tensor: The computed physics-informed loss.
        """
        # Extract relevant variables from predictions
        (
            uu_pred,
            vv_pred,
            ww_pred,
            uv_pred,
            U_pred,
            dUdy_pred,
            P_pred,
            k_pred,
        ) = predictions.chunk(8, dim=1)

        # Retrieve the value of nu
        nu = x[:, 3]

        # Compute the gradients of U with respect to y^+
        grad_U = torch.autograd.grad(
            U_pred,
            x,
            grad_outputs=torch.ones_like(U_pred),
            create_graph=True,
        )[0][:, 0]

        # Compute the second-order derivatives of U with respect to y^+
        grad2_U = torch.autograd.grad(
            grad_U,
            x,
            grad_outputs=torch.ones_like(grad_U),
            create_graph=True,
        )[0][:, 0]

        # Compute the gradients of the predicted velocity components
        grad_uv = torch.autograd.grad(
            uv_pred,
            x,
            create_graph=True,
            grad_outputs=torch.ones_like(uv_pred),
        )[0][:, 0]
        grad_vv = torch.autograd.grad(
            vv_pred,
            x,
            create_graph=True,
            grad_outputs=torch.ones_like(vv_pred),
        )[0][:, 0]
        grad_P = torch.autograd.grad(
            P_pred,
            x,
            create_graph=True,
            grad_outputs=torch.ones_like(P_pred),
        )[0][:, 0]

        # Compute the loss between the predicted and true momentum equations
        x_momentum_eq = -nu * grad2_U + grad_uv
        y_momentum_eq = (1 / self.rho) * grad_P + grad_vv
        loss_x_momentum = F.mse_loss(x_momentum_eq, torch.zeros_like(x_momentum_eq))
        loss_y_momentum = F.mse_loss(y_momentum_eq, torch.zeros_like(y_momentum_eq))
        momentum_loss = loss_x_momentum + loss_y_momentum
        self.log(
            "train_phys_momentum_loss", momentum_loss, on_step=False, on_epoch=True
        )

        # Compute the turbulence kinetic energy consistency loss
        k_calculated = 0.5 * (uu_pred + vv_pred + ww_pred)
        loss_tke = F.mse_loss(k_pred, k_calculated)
        self.log("train_phys_tke_loss", loss_tke, on_step=False, on_epoch=True)

        # Overall physics-informed loss
        physics_loss = (
            +self.loss_phys_momentum_weight * momentum_loss
            + self.loss_phys_k_weight * loss_tke
        )

        return physics_loss

    def data_loss(self, predictions, data):
        """
        Computes the data loss (e.g., MSE) between the predictions and true data.

        Args:
            predictions (torch.Tensor): The model's predictions.
            data (torch.Tensor): The true data.

        Returns:
            torch.Tensor: The computed data loss.
        """
        (
            uu_data,
            vv_data,
            ww_data,
            uv_data,
            U_data,
            dUdy_data,
            P_data,
            k_data,
        ) = data.chunk(8, dim=1)
        (
            uu_pred,
            vv_pred,
            ww_pred,
            uv_pred,
            U_pred,
            dUdy_pred,
            P_pred,
            k_pred,
        ) = predictions.chunk(8, dim=1)

        # Compute the MSE loss between the predictions and the target data
        loss_bound_U = F.mse_loss(U_pred, U_data)
        self.log("train_bound_U_loss", loss_bound_U, on_step=False, on_epoch=True)

        loss_bound_dUdy = F.mse_loss(dUdy_pred, dUdy_data)
        self.log("train_bound_dUdy_loss", loss_bound_dUdy, on_step=False, on_epoch=True)

        loss_bound_P = F.mse_loss(P_pred, P_data)
        self.log("train_bound_P_loss", loss_bound_P, on_step=False, on_epoch=True)

        loss_bound_k = F.mse_loss(k_pred, k_data)
        self.log("train_bound_k_loss", loss_bound_k, on_step=False, on_epoch=True)

        loss_bound_stress = F.mse_loss(
            torch.cat([uu_pred, vv_pred, ww_pred, uv_pred], dim=1),
            torch.cat([uu_data, vv_data, ww_data, uv_data], dim=1),
        )
        self.log(
            "train_bound_stress_loss",
            loss_bound_stress,
            on_step=False,
            on_epoch=True,
        )

        return (
            loss_bound_U * self.loss_bound_U_weight
            + loss_bound_dUdy * self.loss_bound_dUdy_weight
            + loss_bound_P * self.loss_bound_P_weight
            + loss_bound_k * self.loss_bound_k_weight
            + loss_bound_stress * self.loss_bound_stress_weight
        )

    def training_step(self, batch, batch_idx):
        x, targets = batch["features"], batch["targets"]
        x.requires_grad = True

        predictions = self.forward(x)
        loss_pde = self.physics_informed_loss(predictions, x)
        loss_data = self.data_loss(predictions, targets)
        loss = loss_pde + loss_data
        # Log the losses
        self.log(
            "train_pde_loss",
            loss_pde,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_data_loss",
            loss_data,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_total_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes the validation step for a single batch.

        Args:
            batch (dict): The batch of data from the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The validation loss for the batch.
        """
        features, targets = batch["features"], batch["targets"]  # Unpack the batch
        predictions = self(features)  # Predict the targets

        # Calculate the loss
        val_loss = F.mse_loss(input=predictions, target=targets)
        self.log("val_mse", val_loss, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Executes the test step for a single batch.

        Args:
            batch (dict): The batch of data from the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            dict: A dictionary containing the test step metrics.
        """
        features, targets = batch["features"], batch["targets"]  # Unpack the batch
        predictions = self.forward(features)  # Predict the targets

        (uu_data, vv_data, ww_data, uv_data, U_data, dUdy_data, P_data, k_data) = (
            targets.chunk(8, dim=1)
        )
        (uu_pred, vv_pred, ww_pred, uv_pred, U_pred, dUdy_pred, P_pred, k_pred) = (
            predictions.chunk(8, dim=1)
        )

        # Compute the MSE loss between the predictions and the target data
        loss_bound_U = F.mse_loss(U_pred, U_data)
        loss_bound_dUdy = F.mse_loss(dUdy_pred, dUdy_data)
        loss_bound_P = F.mse_loss(P_pred, P_data)
        loss_bound_k = F.mse_loss(k_pred, k_data)
        loss_bound_uu = F.mse_loss(uu_pred, uu_data)
        loss_bound_vv = F.mse_loss(vv_pred, vv_data)
        loss_bound_ww = F.mse_loss(ww_pred, ww_data)
        loss_bound_uv = F.mse_loss(uv_pred, uv_data)
        self.log("mse_U", loss_bound_U, on_step=False, on_epoch=True)
        self.log("mse_dUdy", loss_bound_dUdy, on_step=False, on_epoch=True)
        self.log("mse_P", loss_bound_P, on_step=False, on_epoch=True)
        self.log("mse_k", loss_bound_k, on_step=False, on_epoch=True)
        self.log("mse_uu", loss_bound_uu, on_step=False, on_epoch=True)
        self.log("mse_vv", loss_bound_vv, on_step=False, on_epoch=True)
        self.log("mse_ww", loss_bound_ww, on_step=False, on_epoch=True)
        self.log("mse_uv", loss_bound_uv, on_step=False, on_epoch=True)

        # Compute the R^2 score 
        r2_U = 1 - (U_pred - U_data).pow(2).sum() / (U_data - U_data.mean()).pow(2).sum()
        r2_dUdy = 1 - (dUdy_pred - dUdy_data).pow(2).sum() / (dUdy_data - dUdy_data.mean()).pow(2).sum()
        r2_P = 1 - (P_pred - P_data).pow(2).sum() / (P_data - P_data.mean()).pow(2).sum()
        r2_k = 1 - (k_pred - k_data).pow(2).sum() / (k_data - k_data.mean()).pow(2).sum()
        r2_uu = 1 - (uu_pred - uu_data).pow(2).sum() / (uu_data - uu_data.mean()).pow(2).sum()
        r2_vv = 1 - (vv_pred - vv_data).pow(2).sum() / (vv_data - vv_data.mean()).pow(2).sum()
        r2_ww = 1 - (ww_pred - ww_data).pow(2).sum() / (ww_data - ww_data.mean()).pow(2).sum()
        r2_uv = 1 - (uv_pred - uv_data).pow(2).sum() / (uv_data - uv_data.mean()).pow(2).sum()
        self.log("r2_U", r2_U, on_step=False, on_epoch=True)
        self.log("r2_dUdy", r2_dUdy, on_step=False, on_epoch=True)
        self.log("r2_P", r2_P, on_step=False, on_epoch=True)
        self.log("r2_k", r2_k, on_step=False, on_epoch=True)
        self.log("r2_uu", r2_uu, on_step=False, on_epoch=True)
        self.log("r2_vv", r2_vv, on_step=False, on_epoch=True)
        self.log("r2_ww", r2_ww, on_step=False, on_epoch=True)
        self.log("r2_uv", r2_uv, on_step=False, on_epoch=True)

        # Compute the RMSE
        rmse_U = torch.sqrt(loss_bound_U)
        rmse_dUdy = torch.sqrt(loss_bound_dUdy)
        rmse_P = torch.sqrt(loss_bound_P)
        rmse_k = torch.sqrt(loss_bound_k)
        rmse_uu = torch.sqrt(loss_bound_uu)
        rmse_vv = torch.sqrt(loss_bound_vv)
        rmse_ww = torch.sqrt(loss_bound_ww)
        rmse_uv = torch.sqrt(loss_bound_uv)
        self.log("rmse_U", rmse_U, on_step=False, on_epoch=True)
        self.log("rmse_dUdy", rmse_dUdy, on_step=False, on_epoch=True)
        self.log("rmse_P", rmse_P, on_step=False, on_epoch=True)
        self.log("rmse_k", rmse_k, on_step=False, on_epoch=True)
        self.log("rmse_uu", rmse_uu, on_step=False, on_epoch=True)
        self.log("rmse_vv", rmse_vv, on_step=False, on_epoch=True)
        self.log("rmse_ww", rmse_ww, on_step=False, on_epoch=True)
        self.log("rmse_uv", rmse_uv, on_step=False, on_epoch=True)

        # Total loss
        mse_total = F.mse_loss(predictions, targets)
        rmse_total = torch.sqrt(mse_total)
        r2_total = 1 - (predictions - targets).pow(2).sum() / (targets - targets.mean()).pow(2).sum()
        self.log("mse_total", mse_total, on_step=False, on_epoch=True)
        self.log("rmse_total", rmse_total, on_step=False, on_epoch=True)
        self.log("r2_total", r2_total, on_step=False, on_epoch=True)


        return {
            "mse_total": mse_total,
            "rmse_total": rmse_total,
            "r2_total": r2_total,
            "mse_U": loss_bound_U,
            "mse_dUdy": loss_bound_dUdy,
            "mse_P": loss_bound_P,
            "mse_k": loss_bound_k,
            "mse_uu": loss_bound_uu,
            "mse_vv": loss_bound_vv,
            "mse_ww": loss_bound_ww,
            "mse_uv": loss_bound_uv,
            "r2_U": r2_U,
            "r2_dUdy": r2_dUdy,
            "r2_P": r2_P,
            "r2_k": r2_k,
            "r2_uu": r2_uu,
            "r2_vv": r2_vv,
            "r2_ww": r2_ww,
            "r2_uv": r2_uv,
            "rmse_U": rmse_U,
            "rmse_dUdy": rmse_dUdy,
            "rmse_P": rmse_P,
            "rmse_k": rmse_k,
            "rmse_uu": rmse_uu,
            "rmse_vv": rmse_vv,
            "rmse_ww": rmse_ww,
            "rmse_uv": rmse_uv
        }

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training the model.

        Returns:
            dict: The dictionary containing the optimizer, learning rate scheduler, and monitoring metric.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        """ optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.lr,
            max_iter=100,
            history_size=200,
        )  """

        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=10, min_lr=1e-8
        )
        

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_steps, eta_min=1e-6, 
        )
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=85
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 85,
                "monitor": "val_mse",
            },
            "monitor": "val_mse",
        }

        return optimizer
