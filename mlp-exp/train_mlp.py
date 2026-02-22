import torch
import torch.nn as nn
import torch.optim as optim
from dataset.data_loader import load_excel_to_dataloader
from mlp.mlp_definition import InterpretabilityMLP

# --- 3. Training Script ---
def train_model():
    model = InterpretabilityMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_loader = load_excel_to_dataloader("dataset/mlp_train.xlsx")
    val_loader = load_excel_to_dataloader("dataset/mlp_val.xlsx")

    for epoch in range(250):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
        
        # Simple Validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(bx), by) for bx, by in val_loader) / len(val_loader)
            print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "mlp/trained_mlp.pth")
    return model

# --- 3. Training & Testing Script ---
def train_and_test_model():
    model = InterpretabilityMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Load all three sets
    train_loader = load_excel_to_dataloader("dataset/mlp_train.xlsx")
    val_loader   = load_excel_to_dataloader("dataset/mlp_val.xlsx")
    test_loader  = load_excel_to_dataloader("dataset/mlp_test.xlsx")

    # --- Training Loop ---
    for epoch in range(250):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation Check
        model.eval()
        with torch.no_grad():
            v_loss = sum(criterion(model(bx), by) for bx, by in val_loader) / len(val_loader)
            print(f"Epoch {epoch+1} | Val MSE: {v_loss:.4f}")

    # --- FINAL TEST STEP ---
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model(bx)
            test_loss += criterion(preds, by).item()
    
    avg_test_loss = test_loss / len(test_loader)
    print("\n--- Final Results ---")
    print(f"Final Test MSE: {avg_test_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "mlp/trained_mlp.pth")
    return model

def train_to_perfection():
    model = InterpretabilityMLP()
    train_loader = load_excel_to_dataloader("dataset/mlp_train.xlsx", batch_size=64)
    val_loader = load_excel_to_dataloader("dataset/mlp_val.xlsx", batch_size=64)

    epochs = 500 # Pushing further for total convergence
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # OneCycleLR helps "jump" into the correct indexing logic
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )
    
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                v_loss = sum(criterion(model(bx), by) for bx, by in val_loader) / len(val_loader)
                print(f"Epoch {epoch+1} | Val MSE: {v_loss:.6f}")

    torch.save(model.state_dict(), "mlp/perfect_mlp.pth")
    return model

if __name__ == "__main__":
    trained_model = train_to_perfection()