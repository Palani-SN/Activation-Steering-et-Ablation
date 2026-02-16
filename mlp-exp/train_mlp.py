import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import numpy as np
import random

from mlp.mlp_definition import FinalSpatialMLP
from dataset.data_generator import OneHotSpatialDataset
from torch.utils.data import Dataset, DataLoader

# --- 3. TRAINING ---
def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(OneHotSpatialDataset(60000), batch_size=512, shuffle=True)
    val_loader = DataLoader(OneHotSpatialDataset(5000), batch_size=512)

    model = FinalSpatialMLP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print("Training with One-Hot Coordinates...")

    for epoch in range(100):
        model.train()
        for g, c, y in train_loader:
            g, c, y = g.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.MSELoss()(model(g, c), y)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss = sum(nn.MSELoss()(model(g.to(DEVICE), c.to(DEVICE)), y.to(DEVICE)).item() for g, c, y in val_loader) / len(val_loader)
        scheduler.step(v_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Val MSE: {v_loss:.6f}")
        if v_loss < 0.01: break

    # Verification
    model.eval()
    print("\nFinal Precise Results:")
    test_loader = DataLoader(OneHotSpatialDataset(5), batch_size=1)
    for g, c, y in test_loader:
        pred = model(g.to(DEVICE), c.to(DEVICE))
        print(f"Actual: {y.item():.0f} | Prediction: {pred.item():.4f}")

if __name__ == "__main__":
    train()