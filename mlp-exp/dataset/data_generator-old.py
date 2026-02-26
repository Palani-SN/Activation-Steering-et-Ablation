import random
import pandas as pd
import numpy as np


class MLPExcelGenerator:
    def __init__(self, input_shape=(5, 2)):
        self.input_shape = input_shape  # (x_dim, y_dim) -> 5 columns, 2 rows

    def _generate_single_sample(self):
        """Generates one random sample and calculates its output and labels."""
        x_dim, y_dim = self.input_shape

        # 1. Create the input matrix
        # Each row (y_dim=2) gets 4 random ints (0-9) and 1 index (0-3)
        inp = np.array([
            (
                [random.randint(0, 9) for _ in range(x_dim - 1)]
                +
                [random.randint(0, x_dim - 2)]
            ) for _ in range(y_dim)
        ])

        # 2. Logic: Index into the rows using the last element
        val1 = inp[0][inp[0][-1]]
        val2 = inp[-1][inp[-1][-1]]
        out_val = int(val1 - val2)

        # 3. Determine Concept Group
        is_pos = out_val > 0
        is_neg = out_val < 0
        is_odd = 0 <= abs(out_val) < 5
        is_even = 5 <= abs(out_val) < 10

        # Determine specific quadrant for balancing
        # Note: we exclude 0 to maintain 4 clean quadrants for steering
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

    def generate_balanced_data(self, total_rows: int):
        """Generates data ensuring equal proportions of the 4 concept combinations."""
        target_per_group = total_rows // 4
        buckets = {
            "pos_odd": [],
            "pos_even": [],
            "neg_odd": [],
            "neg_even": []
        }

        print(f"\n  ⚙ Generating {total_rows} balanced samples ({target_per_group} per group)...")

        attempts = 0
        while any(len(b) < target_per_group for b in buckets.values()):
            inp_list, out_list, group = self._generate_single_sample()
            attempts += 1

            # If it falls into a group and that group isn't full, keep it
            if group and len(buckets[group]) < target_per_group:
                buckets[group].append({
                    "input_list": inp_list,
                    "output_list": out_list,
                    "concept": group  # Storing this helps with SAE analysis later
                })

            if attempts % 5000 == 0:
                counts = {k: len(v) for k, v in buckets.items()}
                progress_bar = " | ".join([f"{k}: {v}/{target_per_group}" for k, v in counts.items()])
                print(f"    Progress: {progress_bar}")

        # Combine, shuffle and return
        all_data = []
        for b in buckets.values():
            all_data.extend(b)

        random.shuffle(all_data)
        return all_data

    def save_all_splits(self, train=8000, val=1000, test=1000):
        """Generates and saves three separate Excel files with balanced data."""
        splits = {
            "mlp_train.xlsx": train,
            "mlp_val.xlsx": val,
            "mlp_test.xlsx": test
        }

        print("\n" + "="*70)
        print("  DATASET GENERATION PIPELINE")
        print("="*70 + "\n")

        for filename, count in splits.items():
            balanced_records = self.generate_balanced_data(count)
            df = pd.DataFrame(balanced_records)
            df.to_excel(filename, index=False)
            print(f"    [OK] {filename:20} | {count:5d} samples generated")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    generator = MLPExcelGenerator()
    generator.save_all_splits(train=8000, val=1000, test=1000)
