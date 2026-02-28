import pandas as pd
import torch
import ast
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Map Concepts to Integers for PyTorch ---
CONCEPT_MAP = {
    "+00 < pos <= +05": 0,
    "+05 < pos <= +10": 1,
    "-05 <= neg < +00": 2,
    "-10 <= neg < -05": 3
}


def load_excel_to_dataloader(filename, batch_size=32, include_concepts=True):
    df = pd.read_excel(filename)

    # Standard input/output conversion
    X = torch.tensor([ast.literal_eval(x)
                     for x in df['input_list']], dtype=torch.float32)
    y = torch.tensor([ast.literal_eval(y)
                     for y in df['output_list']], dtype=torch.float32)

    if include_concepts and 'concept' in df.columns:
        # Convert string labels ('pos_odd') to integers for the DataLoader
        groups = torch.tensor([CONCEPT_MAP[g]
                              for g in df['concept']], dtype=torch.long)
        dataset = TensorDataset(X, y, groups)
    else:
        dataset = TensorDataset(X, y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
