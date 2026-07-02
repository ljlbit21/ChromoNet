import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


def train_model(model, X_train, y_train, device, epochs=100, batch_size=256, lr=1e-3,
                weight_decay=1e-4, save_path=None):
    """Train the ChromoNet model with AdamW + cosine annealing.

    Parameters
    ----------
    model : nn.Module
    X_train : np.ndarray or torch.Tensor
    y_train : np.ndarray or torch.Tensor
    device : torch.device
    epochs : int
    batch_size : int
    lr : float
        Peak learning rate.
    weight_decay : float
    save_path : str or None
    """
    model.to(device)
    model.train()

    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with linear warmup (5 epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=5
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - 5, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
    )

    print(f'Training: epochs={epochs}, batch={batch_size}, lr={lr}, weight_decay={weight_decay}')

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx).squeeze()
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}, lr: {current_lr:.2e}')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to: {save_path}')

    return model
