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

    sae = SparseAutoencoder(input_dim=512, dict_size=2048, k=20).cuda()
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)

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
            print(
                f"SAE Epoch [{epoch+1}/{epochs}] | MSE: {total_mse/len(loader):.6f}")

    torch.save(sae.state_dict(), "sae/universal_sae.pth")


if __name__ == "__main__":

    train_sae_from_payload("harvested_data.pt", epochs=100)