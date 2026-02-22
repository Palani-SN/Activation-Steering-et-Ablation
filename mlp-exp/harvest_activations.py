import torch
from mlp.mlp_definition import InterpretabilityMLP
from dataset.data_loader import load_excel_to_dataloader

def harvest_activations(model_path, dataloader):
    # 1. Setup Model
    model = InterpretabilityMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_acts = []
    
    print("Harvesting activations...")
    with torch.no_grad():
        for batch_x, _ in dataloader:
            # Forward pass triggers the internal dictionary storage
            _ = model(batch_x)
            # We take 'layer2' which corresponds to the 512-dim hidden1 output
            acts = model.activations['layer2']
            all_acts.append(acts.cpu())
            
    # Concatenate into one massive tensor [8000, 512]
    final_tensor = torch.cat(all_acts, dim=0)
    torch.save(final_tensor, "mlp_activations.pt")
    print(f"Success! Saved tensor of shape: {final_tensor.shape}")
    return final_tensor

if __name__ == "__main__":

    # --- Execution ---
    # Use your training loader (8000 samples)
    train_loader = load_excel_to_dataloader("dataset/mlp_train.xlsx", batch_size=64)
    activations = harvest_activations("mlp/perfect_mlp.pth", train_loader)