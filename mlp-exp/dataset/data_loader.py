import pandas as pd
import torch
import ast
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Data Loading Utility ---
def load_excel_to_dataloader(filename, batch_size=32):
    df = pd.read_excel(filename)
    X = torch.tensor([ast.literal_eval(x) for x in df['input_list']], dtype=torch.float32)
    y = torch.tensor([ast.literal_eval(y) for y in df['output_list']], dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)