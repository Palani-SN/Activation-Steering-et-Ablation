import torch
import os
from mlp.mlp_definition import InterpretabilityMLP
from dataset.data_loader import load_excel_to_dataloader, CONCEPT_MAP


def harvest_activations(model_path, dataloader, device="cuda"):

    # 1. Setup Model
    model = InterpretabilityMLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_acts = []
    all_labels = []

    print("\n" + "="*80)
    print("  HARVESTING ACTIVATIONS FROM TRAINED MLP")
    print("="*80)
    print(f"  Device: {device}")
    print("  Expected Samples: ~8000")
    print("="*80 + "\n")
    print(f"  -> Harvesting activations on {device}...")

    with torch.no_grad():

        # Unpack the concept tags (bg) along with inputs (bx)
        for bx, _, bg in dataloader:

            bx = bx.to(device)

            # Forward pass
            _ = model(bx)

            # Capture hidden layer (512-dim)
            acts = model.activations['hidden2']

            all_acts.append(acts.cpu())
            all_labels.append(bg.cpu())

    # 2. Save both Activations and Metadata
    final_acts = torch.cat(all_acts, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    # Saving as a dictionary is cleaner for downstream SAE scripts
    payload = {
        "activations": final_acts,
        "labels": final_labels,
        "concept_map": CONCEPT_MAP
    }

    torch.save(payload, "temp/harvested_data.pt")
    print(
        f"\n  [OK] Successfully saved {final_acts.shape[0]} activations with metadata.")
    print("="*80 + "\n")
    return payload


if __name__ == "__main__":

    # Ensure you use include_concepts=True to match your new balanced loader
    train_loader = load_excel_to_dataloader(
        "dataset/mlp_train.xlsx",
        batch_size=64,
        include_concepts=True
    )

    harvest_activations("mlp/perfect_mlp.pth", train_loader)
