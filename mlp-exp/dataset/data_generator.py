import random
import pandas as pd
import numpy as np

class MLPExcelGenerator:
    def __init__(self, input_shape=(5, 2)):
        self.input_shape = input_shape

    def stream_data(self, set: str, num_rows: int):
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
            
            pre1, pre2 = inp[0][inp[0][-1]], inp[-1][inp[-1][-1]]
            # --- YOUR MATH MODEL LOGIC ---
            if set == "pos":
                post1, post2 = max(pre1, pre2), min(pre1, pre2) 
            if set == "neu":
                normalised = ( pre1 + pre2 ) // 2
                post1, post2 = normalised, normalised
            if set == "neg":
                post1, post2 = min(pre1, pre2), max(pre1, pre2)
            # out = abs( value_at_index_from_first_col - value_at_index_from_second_col )
            # Logic: Using the last element of each column as an index for that column
            
            if set != "neu":
                if (post1 - post2) == 0:
                    if -1 < post1 < 5:
                        if set == "pos":
                            post1 += random.randint(1, 5)
                        else:
                            post2 += random.randint(1, 5)
                    else:
                        if set == "pos":
                            post2 -= random.randint(1, 5)
                        else:
                            post1 -= random.randint(1, 5)
                            
            out = np.array([post1 - post2])
            inp[0][inp[0][-1]], inp[-1][inp[-1][-1]] = post1, post2
            # Yield as Python lists
            yield inp.flatten().tolist(), out.tolist()

    def save_all_splits(self, train=8000, val=1000, test=1000):
        """Generates and saves three separate Excel files."""
        set_cnt = {"train": train, "test": test, "val": val}
        var_type = ("pos", "neu", "neg")

        for _set, _cnt in set_cnt.items():
            for _var in var_type:
                records = []
                filename = f"mlp_{_set}_{_var}.xlsx"
                for inp_list, out_list in self.stream_data(_var, _cnt):
                    records.append({
                        "input_list": str(inp_list), 
                        "output_list": str(out_list)
                    })

                df = pd.DataFrame(records)
                df.to_excel(filename, index=False)
                print(f"Successfully saved {filename}")


# --- Execution ---
if __name__ == "__main__":
    # Note: Your input_shape is (5, 2), matching your indexing logic
    generator = MLPExcelGenerator(input_shape=(5, 2))
    generator.save_all_splits(train=80, val=10, test=10)