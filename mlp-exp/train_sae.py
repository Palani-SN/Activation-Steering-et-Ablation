import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sae.sae_definition import SparseAutoencoder


def train_sae_from_payload(payload_path, epochs=100):
    # Load the dictionary saved by the harvester
    payload = torch.load(payload_path)
    activations = payload["activations"]  # [8000, 512]

    loader = DataLoader(TensorDataset(activations),
                        batch_size=128, shuffle=True)

    sae = SparseAutoencoder(input_dim=256, dict_size=2048).cuda()
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)

    print("\n" + "="*70)
    print("  PHASE II: TRAINING SPARSE AUTOENCODER (SAE)")
    print("="*70)
    print("  Input Dimension: 512")
    print("  Dictionary Size: 2048")
    print(f"  Sparsity (k): {128}")
    print(f"  Total Epochs: {epochs}")
    print("  Batch Size: 128")
    print("="*70 + "\n")

    for epoch in range(epochs):
        total_mse = 0
        for (batch,) in loader:
            batch = batch.cuda()
            optimizer.zero_grad()

            reconstructed, _ = sae(batch)

            # With Top-K, we focus primarily on Reconstruction MSE
            loss = nn.MSELoss()(reconstructed, batch)

            loss.backward()
            optimizer.step()
            total_mse += loss.item()

        if (epoch + 1) % 10 == 0:
            pct = ((epoch + 1) / epochs) * 100
            bar_len = 30
            filled = int(bar_len * (epoch + 1) / epochs)
            bar = "█" * filled + "░" * (bar_len - filled)
            avg_mse = total_mse / len(loader)
            print(f"  [{bar}] Epoch {epoch+1:3d}/{epochs} | MSE: {avg_mse:.6f} | {pct:5.1f}%")

    print("\n" + "="*70)
    print("  [OK] SAE Training Complete!")
    print("="*70 + "\n")

    torch.save(sae.state_dict(), "sae/universal_sae.pth")


if __name__ == "__main__":

    train_sae_from_payload("harvested_data.pt", epochs=100)