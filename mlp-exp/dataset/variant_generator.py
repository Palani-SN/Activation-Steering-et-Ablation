import random
import pandas as pd
import numpy as np
from data_generator import MLPExcelGenerator


class OODDataGenerator(MLPExcelGenerator):
    def __init__(self, input_shape=(5, 2), min_val=10, max_val=20, dtype=int):
        super().__init__(input_shape)
        self.min_val = min_val
        self.max_val = max_val
        self.dtype = dtype

    def _generate_single_sample(self):
        """Generates a sample using the OOD range [min_val, max_val]."""
        x_dim, y_dim = self.input_shape

        if self.dtype == float:
            # Generate floats for data, but keep index as int
            # 1. Create the input matrix using OOD range
            # We keep the index (last element) within the valid range (0 to x_dim-2)
            inp = np.array([
                (
                    [float(round(random.uniform(self.min_val, self.max_val), 2))
                     for _ in range(x_dim - 1)]
                    +
                    [random.randint(0, x_dim - 2)]
                ) for _ in range(y_dim)
            ])
        else:
            # Standard integer generation
            # 1. Create the input matrix using OOD range
            # We keep the index (last element) within the valid range (0 to x_dim-2)
            inp = np.array([
                (
                    [random.randint(self.min_val, self.max_val)
                     for _ in range(x_dim - 1)]
                    +
                    [random.randint(0, x_dim - 2)]
                ) for _ in range(y_dim)
            ])

        if self.dtype == float:
            # 2. Logic: Index into the rows using the last element
            val1 = inp[0][int(inp[0][-1])]
            val2 = inp[-1][int(inp[-1][-1])]
            out_val = float(round(val1 - val2, 2))
        else:
            # 2. Logic: Index into the rows using the last element
            val1 = inp[0][inp[0][-1]]
            val2 = inp[-1][inp[-1][-1]]
            out_val = int(val1 - val2)

        # 3. Determine Concept Group
        is_pos = out_val > 0
        is_neg = out_val < 0

        if self.dtype == float:
            is_odd = abs(round(out_val)) % 2 != 0
            is_even = abs(round(out_val)) % 2 == 0
        else:
            is_odd = abs(out_val) % 2 != 0
            is_even = abs(out_val) % 2 == 0

        group = None
        if is_pos and is_odd:
            group = "pos_odd"
        elif is_pos and is_even:
            group = "pos_even"
        elif is_neg and is_odd:
            group = "neg_odd"
        elif is_neg and is_even:
            group = "neg_even"

        return inp.flatten().tolist(), [out_val], group

    def save_ood_test_set(self, count=1000, filename="mlp_test_ood.xlsx"):
        """Generates and saves a balanced OOD test set."""
        balanced_records = self.generate_balanced_data(count)
        df = pd.DataFrame(balanced_records)
        df.to_excel(filename, index=False)
        print(
            f"    [OK] {filename:20} | {count:5d} samples | Range: [{self.min_val}-{self.max_val}]")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  OOD VARIANT DATASET GENERATION")
    print("="*70 + "\n")
    
    # Create the generator for the Interpolation Test (Range 0-4)
    # This subset range was seen by the model during its original training.
    print("  → Generating Interpolation Test (In-Distribution)...")
    ood_generator = OODDataGenerator(min_val=0, max_val=4)
    ood_generator.save_ood_test_set(count=1000, filename="interp_test.xlsx")

    # Create the generator for the Extrapolation Test (Range 10-19)
    # This subset range was not seen by the model during its original training.
    print("  → Generating Extrapolation Test (Out-of-Distribution)...")
    ood_generator = OODDataGenerator(min_val=10, max_val=19)
    ood_generator.save_ood_test_set(count=1000, filename="extrap_test.xlsx")

    # Create the generator for the Scaling Test (Range 100-109)
    # This subset range was not seen by the model during its original training.
    print("  → Generating Scaling Test (Magnitude Shift)...")
    ood_generator = OODDataGenerator(min_val=100, max_val=109)
    ood_generator.save_ood_test_set(count=1000, filename="scaling_test.xlsx")

    # Create the generator for the Precision Test (Range 0.0-9.0)
    # This subset range was not seen by the model during its original training.
    print("  → Generating Precision Test (Float Values)...")
    ood_generator = OODDataGenerator(min_val=0.0, max_val=9.0, dtype=float)
    ood_generator.save_ood_test_set(count=1000, filename="precision_test.xlsx")
    
    print("\n" + "="*70)
    print("  ✓ ALL OOD VARIANT DATASETS GENERATED")
    print("="*70 + "\n")
