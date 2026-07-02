"""Masked Autoencoder (MAE) pre-training for ChromoNet 1D genomic signals."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os


class MaskedAutoencoder1D(nn.Module):
    """1D Masked Autoencoder for genomic waveform pre-training.

    Parameters
    ----------
    input_len : int
        Number of genes in the sorted expression vector.
    mask_ratio : float
        Fraction of positions to mask (default 0.30).
    encoder_dim : int
        Hidden dimension of the encoder MLP.
    latent_dim : int
        Dimension of the latent bottleneck.
    """

    def __init__(self, input_len, mask_ratio=0.30, encoder_dim=256, latent_dim=128):
        super().__init__()
        self.input_len = input_len
        self.mask_ratio = mask_ratio

        self.encoder = nn.Sequential(
            nn.Linear(input_len, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, input_len),
        )

    def forward(self, x):
        """Forward pass with random masking.

        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed full signal.
        mask : torch.Tensor
            Boolean mask indicating masked positions.
        """
        batch_size = x.size(0)
        device = x.device

        # Generate random mask per sample
        mask = torch.rand(batch_size, self.input_len, device=device) < self.mask_ratio

        # Apply mask (set masked positions to zero)
        x_masked = x.clone()
        x_masked[mask] = 0.0

        # Encode → Decode
        latent = self.encoder(x_masked)
        reconstructed = self.decoder(latent)

        return reconstructed, mask

    def mae_loss(self, x, reconstructed, mask):
        """MSE loss computed only on masked positions."""
        diff = (reconstructed - x) ** 2
        loss = diff[mask].mean()
        return loss


def pretrain_mae(model, X, epochs=100, batch_size=256, lr=1e-3, device=None, save_path=None):
    """Pre-train the MAE on unlabeled expression data.

    Parameters
    ----------
    model : MaskedAutoencoder1D
    X : np.ndarray or torch.Tensor
        Expression matrix (cells x genes), already sorted and residual-corrected.
    epochs : int
    batch_size : int
    lr : float
    device : torch.device
    save_path : str or None
        If provided, saves the encoder state_dict to this path.

    Returns
    -------
    model : MaskedAutoencoder1D
        Trained MAE model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.train()

    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(device)

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f'[MAE Pre-training] epochs={epochs}, batch={batch_size}, mask_ratio={model.mask_ratio:.0%}')

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (bx,) in loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            reconstructed, mask = model(bx)
            loss = model.mae_loss(bx, reconstructed, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f'  Epoch {epoch + 1}/{epochs} — MAE loss: {avg_loss:.6f}')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path)
        print(f'MAE encoder saved to: {save_path}')

    return model
