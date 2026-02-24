import torch
import torch.nn as nn
import torch.optim as optim
from dataset.data_loader import load_excel_to_dataloader, CONCEPT_MAP
from mlp.mlp_definition import InterpretabilityMLP


def train_to_perfection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterpretabilityMLP().to(device)

    # Use include_concepts=True to handle the new balanced dataset format
    train_loader = load_excel_to_dataloader(
        "dataset/mlp_train.xlsx", batch_size=64, include_concepts=True)
    val_loader = load_excel_to_dataloader(
        "dataset/mlp_val.xlsx", batch_size=64, include_concepts=True)
    test_loader = load_excel_to_dataloader(
        "dataset/mlp_test.xlsx", batch_size=64, include_concepts=True)

    epochs = 500
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    criterion = nn.MSELoss()

    print(f"Starting training on {device}...")

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
                print(f"Epoch {epoch+1} | Val MSE: {v_loss:.6f}")

    # --- EVALUATION: PER-CONCEPT ACCURACY ---
    print("\n--- Final Performance Analysis ---")
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
    for name, losses in group_losses.items():
        avg_mse = sum(losses) / len(losses)
        overall_mse += avg_mse
        print(
            f"Concept: {name:10} | MSE: {avg_mse:.6f} | Count: {len(losses)}")

    print(f"Total Test MSE: {overall_mse / 4:.6f}")

    # Save the model
    torch.save(model.state_dict(), "mlp/perfect_mlp.pth")

if __name__ == "__main__":
    train_to_perfection()
