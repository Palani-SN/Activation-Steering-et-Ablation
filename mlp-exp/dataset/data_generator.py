import random
import pandas as pd
import numpy as np

class MLPExcelGenerator:
    def __init__(self, input_shape=(5, 2)):
        self.input_shape = input_shape

    def stream_data(self, num_rows: int):
        """Yields (input_list, output_list) pairs using index-based logic."""
        # x is rows (5), y is cols (2)
        x_dim, y_dim = self.input_shape 
        
        for _ in range(num_rows):
            # 1. Create the input matrix
            # Each row gets random ints (0-10), last element is a valid index (0 to x-2)
            inp = np.array([
                (
                    [random.randint(0, 9) for _ in range(x_dim - 1)]
                    +
                    [random.randint(0, x_dim - 2)]
                ) for _ in range(y_dim)
            ])
            
            # --- YOUR MATH MODEL LOGIC ---
            # out = abs( value_at_index_from_first_col - value_at_index_from_second_col )
            # Logic: Using the last element of each column as an index for that column
            val1 = inp[0][inp[0][-1]]
            val2 = inp[-1][inp[-1][-1]]
            out = np.array([abs(val1 - val2)])
            
            # Yield as Python lists
            yield inp.flatten().tolist(), out.tolist()

    def save_all_splits(self, train=8000, val=1000, test=1000):
        """Generates and saves three separate Excel files."""
        splits = {
            "mlp_train.xlsx": train,
            "mlp_val.xlsx": val,
            "mlp_test.xlsx": test
        }
        
        for filename, count in splits.items():
            print(f"Generating {count} rows for {filename}...")
            data_records = []
            
            for inp_list, out_list in self.stream_data(count):
                data_records.append({
                    "input_list": str(inp_list), 
                    "output_list": str(out_list)
                })
            
            df = pd.DataFrame(data_records)
            df.to_excel(filename, index=False)
            print(f"Successfully saved {filename}")

# --- Execution ---
if __name__ == "__main__":
    # Note: Your input_shape is (5, 2), matching your indexing logic
    generator = MLPExcelGenerator(input_shape=(5, 2))
    generator.save_all_splits(train=8000, val=1000, test=1000)