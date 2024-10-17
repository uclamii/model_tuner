import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import optuna


class LitModel(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dims: list = [128, 128],
        num_layers: int = 2,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.layers = self._build_layers(input_dim)

    def _build_layers(self, input_dim):
        layers = []
        for num_layer in range(self.num_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dims[num_layer]))
            layers.append(nn.ReLU())
            input_dim = self.hidden_dims[num_layer]
        layers.append(nn.Linear(self.hidden_dim[-1], 1))  # Binary classification output
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def objective(
    trial: optuna.Trial,
    train_loader,
    val_loader,
    metric_name="val_loss",
    checkpoint_callback=None,
):

    num_layers = trial.suggest_int("num_layers", 1, 10)
    # hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    # Suggest a distinct number of hidden units for each layer
    hidden_dims = [
        trial.suggest_int(f"hidden_dim_layer_{i}", 32, 256) for i in range(num_layers)
    ]

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    num_epochs = trial.suggest_int("num_epochs", 5, 20)

    # Setup model
    model = LitModel(
        input_dim=train_loader.dataset[0][0].shape[0],
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        learning_rate=learning_rate,
    )

    # Use Trainer for training
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=False,
        callbacks=[checkpoint_callback],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Fit the model
    trainer.fit(model, train_loader, val_loader)

    # Retrieve and return the chosen metric
    metric_value = trainer.callback_metrics.get(metric_name)
    return metric_value.item() if metric_value is not None else float("inf")


import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y


from torch.utils.data import DataLoader


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    # Create datasets
    train_dataset = NumpyDataset(X_train, y_train)
    val_dataset = NumpyDataset(X_val, y_val)
    test_dataset = NumpyDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
