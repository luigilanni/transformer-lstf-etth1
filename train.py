#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla Transformer baseline for long-sequence time-series forecasting on ETTh1,
using the Informer-style generative decoder input (Zhou et al., 2021).

Author: Luigi Lanni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Data ------------------------------------------

df = pd.read_csv("ETTh1.csv")
df = df[["OT"]]                    # univariate: oil temperature only
data = df.values                   # (N, 1) numpy array

seq_len = 96                       # encoder input length
pred_len = 24                      # forecast horizon
token_len = 48                     # start-token length fed to decoder

# Sliding windows
X, Y = [], []
for i in range(len(data) - seq_len - pred_len):
    X.append(data[i : i + seq_len])
    Y.append(data[i + seq_len : i + seq_len + pred_len])
X = np.array(X)
Y = np.array(Y)

# Chronological split 70 / 15 / 15
train_size = int(0.70 * len(X))
val_size   = int(0.15 * len(X))

X_train, Y_train = X[:train_size],                       Y[:train_size]
X_val,   Y_val   = X[train_size:train_size + val_size],  Y[train_size:train_size + val_size]
X_test,  Y_test  = X[train_size + val_size:],            Y[train_size + val_size:]

# Standardize using training statistics only
mean = X_train.mean()
std  = X_train.std()
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std
Y_train = (Y_train - mean) / std
Y_val   = (Y_val   - mean) / std
Y_test  = (Y_test  - mean) / std

# To tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
Y_val   = torch.tensor(Y_val,   dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
Y_test  = torch.tensor(Y_test,  dtype=torch.float32)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = TimeSeriesDataset(X_train, Y_train)
val_dataset   = TimeSeriesDataset(X_val,   Y_val)
test_dataset  = TimeSeriesDataset(X_test,  Y_test)

# NOTE: shuffle=True on training so batches are not seen in fixed temporal order.
# Validation and test stay shuffle=False so metrics are reproducible.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)


# ----------------------------- Model -----------------------------------------

class TransformerModel(nn.Module):
    """
    Vanilla PyTorch Transformer (encoder-decoder) with an Informer-style
    generative decoder input: [known token_len values | zeros for pred_len steps]
    -> predicts all pred_len future steps in a single forward pass.
    """
    def __init__(self, d_model=128, nhead=4, num_enc=2, num_dec=1, pred_len=24):
        super().__init__()
        self.pred_len = pred_len

        self.enc_embed = nn.Linear(1, d_model)
        self.dec_embed = nn.Linear(1, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            batch_first=True,
        )

        self.projection = nn.Linear(d_model, 1)

    def forward(self, x_enc, x_dec):
        enc = self.enc_embed(x_enc)
        dec = self.dec_embed(x_dec)
        tgt_mask = self.transformer.generate_square_subsequent_mask(dec.size(1)).to(dec.device)
        out = self.transformer(enc, dec, tgt_mask=tgt_mask)
        out = self.projection(out)
        return out[:, -self.pred_len:, :]   # keep only forecast horizon


def build_decoder_input(batch_X, token_len=48, pred_len=24):
    """Concatenate the last `token_len` known values with `pred_len` zeros."""
    B = batch_X.size(0)
    x_dec = torch.zeros(B, token_len + pred_len, 1, device=batch_X.device)
    x_dec[:, :token_len, :] = batch_X[:, -token_len:, :]
    return x_dec


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = TransformerModel(pred_len=pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)


# ----------------------------- Training loop with early stopping -------------

num_epochs = 8
patience = 3                # stop if val loss does not improve for `patience` epochs
best_val = float("inf")
epochs_no_improve = 0
best_state = None

for epoch in range(num_epochs):
    # Train
    model.train()
    total_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        x_dec = build_decoder_input(batch_X, token_len, pred_len)

        optimizer.zero_grad()
        output = model(batch_X, x_dec)
        loss = criterion(output, batch_Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            x_dec = build_decoder_input(batch_X, token_len, pred_len)
            output = model(batch_X, x_dec)
            val_loss += criterion(output, batch_Y).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1:>2} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    # Early stopping bookkeeping
    if val_loss < best_val:
        best_val = val_loss
        epochs_no_improve = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after epoch {epoch + 1} (no val improvement for {patience} epochs).")
            break

# Restore best weights and save
if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), "transformer_model.pth")


# ----------------------------- Test ------------------------------------------

model.eval()
test_loss_mse = 0.0
test_loss_mae = 0.0
with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        x_dec = build_decoder_input(batch_X, token_len, pred_len)
        output = model(batch_X, x_dec)
        test_loss_mse += ((output - batch_Y) ** 2).mean().item()
        test_loss_mae += (output - batch_Y).abs().mean().item()

test_loss_mse /= len(test_loader)
test_loss_mae /= len(test_loader)
print(f"Test MSE: {test_loss_mse:.4f} | Test MAE: {test_loss_mae:.4f}")


# ----------------------------- Prediction plot -------------------------------
# 2x2 grid of forecasts on random test windows, denormalized to original units.
# A single window would over- or under-sell the model; a small sample is fairer.

import random
random.seed(0)                       # reproducible sample
n_rows, n_cols = 2, 2
sample_idx = random.sample(range(len(X_test)), n_rows * n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
axes = axes.flatten()

model.eval()
with torch.no_grad():
    for ax, idx in zip(axes, sample_idx):
        x_sample = X_test[idx:idx + 1].to(device)      # (1, 96, 1)
        y_true   = Y_test[idx:idx + 1].to(device)      # (1, 24, 1)
        x_dec    = build_decoder_input(x_sample, token_len, pred_len)
        y_pred   = model(x_sample, x_dec)              # (1, 24, 1)

        x_hist_i = (x_sample.cpu().numpy().squeeze() * std) + mean
        y_true_i = (y_true.cpu().numpy().squeeze()   * std) + mean
        y_pred_i = (y_pred.cpu().numpy().squeeze()   * std) + mean

        ax.plot(range(seq_len), x_hist_i, label="History", color="#333333")
        ax.plot(range(seq_len, seq_len + pred_len), y_true_i, label="Ground truth", color="#1f77b4")
        ax.plot(range(seq_len, seq_len + pred_len), y_pred_i, label="Forecast",
                color="#E86A24", linestyle="--")
        ax.axvline(x=seq_len - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Test window #{idx}")
        ax.set_ylabel("OT")

axes[0].legend(loc="upper left", fontsize=9)
for ax in axes[-n_cols:]:
    ax.set_xlabel("Time step (hours)")

fig.suptitle(
    f"ETTh1 — 96h history, 24h forecast   |   "
    f"Test MSE: {test_loss_mse:.4f}   Test MAE: {test_loss_mae:.4f}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("forecast_example.png", dpi=150)
plt.show()
print("Saved forecast_example.png")
