import torch
# import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sae.sae_definition import SparseAutoencoder
import os

def train_sae():
    # --- Hyperparameters ---
    ACT_DIM = 256
    DICT_SIZE = 1024
    L1_COEFF = 0.001  # Start much lower to "wake up" the neurons
    LR = 1e-3
    EPOCHS = 50
    BATCH_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load and PRE-PROCESS Data
    if not os.path.exists("harvested_data.pt"):
        print("Error: harvested_data.pt not found.")
        return

    raw_data = torch.load("harvested_data.pt", weights_only=False)
    acts = raw_data if torch.is_tensor(raw_data) else raw_data['acts']
    acts = acts.to(torch.float32).to(DEVICE)

    # CRITICAL: Normalize data so it has a standard scale
    # This prevents L1 from killing neurons that have small raw values
    acts_mean = acts.mean(dim=0)
    acts_std = acts.std(dim=0) + 1e-6
    norm_acts = (acts - acts_mean) / acts_std

    dataset = TensorDataset(norm_acts)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Setup
    model = SparseAutoencoder(ACT_DIM, DICT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Starting Training | Device: {DEVICE}")
    
    for epoch in range(EPOCHS):
        total_mse, total_l1, total_l0 = 0, 0, 0
        
        for (x_batch,) in loader:
            # Forward
            recon, f_acts, mse, l1 = model(x_batch, l1_coeff=L1_COEFF)
            loss = mse + l1
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Constraint
            model.make_decoder_weights_unit_norm()
            
            # Stats
            total_mse += mse.item()
            total_l1 += l1.item()
            total_l0 += (f_acts > 0).float().sum(dim=1).mean().item()

        avg_l0 = total_l0 / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | MSE: {total_mse/len(loader):.5f} | L1: {total_l1/len(loader):.5f} | L0: {avg_l0:.1f}")

    # 3. Save weights AND the normalization stats
    # We need mean/std to use the SAE during inference later!
    torch.save({
        'model_state_dict': model.state_dict(),
        'acts_mean': acts_mean.cpu(),
        'acts_std': acts_std.cpu(),
        'dict_size': DICT_SIZE,
        'act_dim': ACT_DIM
    }, "sae/sae_weights.pth")
    print("\nTraining complete. Saved sae/sae_weights.pth")

if __name__ == "__main__":
    train_sae()