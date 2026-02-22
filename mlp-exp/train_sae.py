import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sae.sae_definition import SparseAutoencoder

# --- 2. The Training Function ---
def train_sae_from_file(file_path, epochs=100, l1_coeff=1e-4):
    # Load the harvested activations
    activations_tensor = torch.load(file_path)
    print(f"Loaded activations: {activations_tensor.shape}")

    # Dataset & Loader
    dataset = TensorDataset(activations_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize SAE
    input_dim = activations_tensor.shape[1] # Should be 512
    sae = SparseAutoencoder(input_dim=input_dim, dict_size=2048)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    
    # Loss tracking
    for epoch in range(epochs):
        model_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            
            reconstructed, hidden = sae(batch)
            
            # Reconstruction accuracy
            mse_loss = nn.MSELoss()(reconstructed, batch)
            
            # L1 Sparsity: Force features to compete for activation
            # We calculate mean over batch to keep scale consistent
            l1_loss = l1_coeff * torch.norm(hidden, p=1, dim=1).mean()
            
            loss = mse_loss + l1_loss
            loss.backward()
            optimizer.step()
            model_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = model_loss / len(loader)
            print(f"SAE Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")

    torch.save(sae.state_dict(), "sae/sae_model.pth")
    print("SAE training complete. Weights saved.")
    return sae

# --- Execution ---
if __name__ == "__main__":
    # Ensure mlp_activations.pt exists in the same directory
    sae_model = train_sae_from_file("mlp_activations.pt", epochs=100)