import torch
import torch.nn as nn
import torch.optim as optim
from dataset.data_loader import load_excel_to_dataloader, CONCEPT_MAP
from mlp.mlp_definition import InterpretabilityMLP


def train_to_perfection(epochs=500):

    mse = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterpretabilityMLP().to(device)

    # Use include_concepts=True to handle the new balanced dataset format
    train_loader = load_excel_to_dataloader(
        "dataset/mlp_train.xlsx", batch_size=256, include_concepts=True)
    val_loader = load_excel_to_dataloader(
        "dataset/mlp_val.xlsx", batch_size=256, include_concepts=True)
    test_loader = load_excel_to_dataloader(
        "dataset/mlp_test.xlsx", batch_size=256, include_concepts=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    criterion = nn.MSELoss()

    print("\n" + "="*80)
    print("  PHASE I: TRAINING MLP TO INTERPRETABLE PERFECTION")
    print("="*80)
    print(f"  Device: {device}")
    print(f"  Total Epochs: {epochs}")
    print("  Batch Size: 256")
    print("="*80 + "\n")

    for epoch in range(epochs):

        model.train()

        for batch_x, batch_y, _ in train_loader:  # Unpack and ignore groups during training
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % 50 == 0:

            model.eval()

            with torch.no_grad():

                v_loss = sum(criterion(model(bx.to(device)), by.to(device))
                             for bx, by, _ in val_loader) / len(val_loader)
                pct = ((epoch + 1) / epochs) * 100
                bar_len = 30
                filled = int(bar_len * (epoch + 1) / epochs)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"  [{bar}] Epoch {epoch+1:3d}/{epochs} | Val MSE: {v_loss:.6f} | {pct:5.1f}%")
                mse.append((int(epoch+1), float(v_loss)))

    # --- EVALUATION: PER-CONCEPT ACCURACY ---
    print("\n" + "="*80)
    print("  FINAL PERFORMANCE ANALYSIS")
    print("="*80)
    model.eval()

    # Initialize trackers for each concept group
    inv_map = {v: k for k, v in CONCEPT_MAP.items()}
    group_losses = {name: [] for name in CONCEPT_MAP.keys()}

    with torch.no_grad():

        for bx, by, bg in test_loader:

            bx, by = bx.to(device), by.to(device)
            preds = model(bx)

            # Calculate individual squared errors for this batch
            errors = (preds - by)**2

            # Attribute errors to their respective concept groups
            for i in range(len(bg)):

                group_name = inv_map[bg[i].item()]
                group_losses[group_name].append(errors[i].item())

    # Final report
    overall_mse = 0
    print("\n  Per-Concept Metrics:")
    print("  " + "-"*66)

    for name, losses in group_losses.items():

        avg_mse = sum(losses) / len(losses)
        overall_mse += avg_mse
        name_display = name.replace('_', ' ').title()
        print(
            f"  → {name_display:20} | MSE: {avg_mse:.6f} | Samples: {len(losses):4d}")

    print("  " + "-"*66)
    print(f"  ✓ Total Test MSE: {overall_mse / 4:.6f}")
    print("="*80 + "\n")

    # Save the model
    torch.save(model.state_dict(), "mlp/perfect_mlp.pth")

    return "mlp/perfect_mlp.pth", mse


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    file, data = train_to_perfection(epochs=500)

    # Unpack the tuples into x and y coordinates using zip
    x, y = zip(*data)
    # print(x, y)

    # Plot the data
    # Use 'o' for markers and '-' for a line
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(file)
    plt.grid(True)
    plt.savefig("images/mlp_training.png")
    plt.close()
