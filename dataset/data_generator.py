import math
import random
import pandas as pd
import numpy as np


class MinimalGenerator:
    def __init__(self, dim: tuple = (5, 2), range: tuple = (0, 9), dtype: type = int, categories: dict = None):
        self.input_shape = dim
        self.range = range
        self.dtype = dtype
        self.categories = categories or {}
        self.leaf_categories = self._extract_leaf_categories()
        self.value_bounds = self._compute_target_bounds()
        self.feasible_leaves, self.infeasible_leaves = self._filter_feasible_leaves()
        self._warned_infeasible = False

    def stream_data(self, num_rows: int):
        """Yields (input_list, output_list) pairs using index-based logic."""
        for _ in range(num_rows):
            inp_list, out_list, _ = self._generate_single_example()
            yield inp_list, out_list

    def _generate_single_example(self):
        """Generates one data point and returns (input_list, output_list, target_value)."""
        x_dim, y_dim = self.input_shape
        x_min, x_max = self.range

        if self.dtype == float:
            inp = np.array([
                (
                    [round(random.uniform(x_min, x_max), 2)
                     for _ in range(x_dim - 1)]
                    +
                    [random.randint(0, x_dim - 2)]
                ) for _ in range(y_dim)
            ])

        if self.dtype == int:
            inp = np.array([
                (
                    [random.randint(x_min, x_max) for _ in range(x_dim - 1)]
                    +
                    [random.randint(0, x_dim - 2)]
                ) for _ in range(y_dim)
            ])

        # Logic: Using the last element of each column as an index for that column
        if self.dtype == float:
            val1 = inp[0][int(inp[0][-1])]
            val2 = inp[-1][int(inp[-1][-1])]
        if self.dtype == int:
            val1 = inp[0][inp[0][-1]]
            val2 = inp[-1][inp[-1][-1]]
        # Keep the signed delta so downstream category filters can use the full range.
        target_value = val1 - val2
        if self.dtype == int:
            out = np.array([target_value])
        if self.dtype == float:
            out = np.array([round(target_value, 2)])

        return inp.flatten().tolist(), out.tolist(), target_value

    def _compute_target_bounds(self):
        x_min, x_max = self.range
        lower = float(x_min - x_max)
        upper = float(x_max - x_min)
        return lower, upper

    def _extract_leaf_categories(self):
        """Unrolls the category tree into leaf nodes with accumulated conditions."""
        categories = []

        if not self.categories:
            return categories

        def dfs(node_key, node_dict, path, cond_chain):
            if not isinstance(node_dict, dict):
                raise ValueError(
                    "Each category node must be a dictionary with a 'cond' callable.")

            node_cond = node_dict.get("cond")
            updated_chain = cond_chain + ([node_cond] if node_cond else [])
            children = [(k, v) for k, v in node_dict.items() if k != "cond"]
            current_path = path + [node_key]

            if not children:
                # label = " :: ".join(current_path)
                # Use only the leaf node's key as the label
                label = current_path[-1]
                categories.append({
                    "label": label,
                    "conds": tuple(updated_chain)
                })
                return

            for child_key, child_node in children:
                dfs(child_key, child_node, current_path, updated_chain)

        for root_key, root_node in self.categories.items():
            dfs(root_key, root_node, [], [])

        return categories

    def _filter_feasible_leaves(self):
        """Splits leaves into feasible vs infeasible for the configured range."""
        feasible = []
        infeasible = []
        lower, upper = self.value_bounds

        for leaf in self.leaf_categories:
            if self._leaf_is_feasible(leaf, lower, upper):
                feasible.append(leaf)
            else:
                infeasible.append(leaf["label"])

        return tuple(feasible), tuple(infeasible)

    def _value_matches_leaf(self, value, leaf):
        return all(cond(value) for cond in leaf["conds"])

    def _leaf_is_feasible(self, leaf, lower, upper):
        if lower > upper:
            return False

        if self.dtype == int:
            start = int(math.ceil(lower))
            end = int(math.floor(upper))
            for candidate in range(start, end + 1):
                if self._value_matches_leaf(candidate, leaf):
                    return True
            return False

        # For float ranges, probe a grid of values plus the boundaries to detect feasibility.
        sample_points = np.linspace(lower, upper, num=25)
        probe_values = {lower, upper, 0.0}
        probe_values.update(sample_points.tolist())

        for candidate in probe_values:
            if self._value_matches_leaf(candidate, leaf):
                return True

        return False

    def _generate_balanced_records(self, total_rows: int):
        if not self.leaf_categories:
            raise ValueError(
                "Category definitions are required to balance the dataset.")

        leaves = list(self.feasible_leaves)
        if not leaves:
            raise RuntimeError(
                "No feasible category leaves for the current configuration. Adjust the range or category conditions.")

        if self.infeasible_leaves and not self._warned_infeasible:
            skipped = ", ".join(self.infeasible_leaves)
            print(
                f"Skipping infeasible categories for this configuration: {skipped}")
            self._warned_infeasible = True

        leaf_count = len(leaves)
        if total_rows % leaf_count != 0:
            raise ValueError(
                f"Total rows ({total_rows}) must be divisible by the number of leaf categories ({leaf_count})."
            )

        rows_per_leaf = total_rows // leaf_count
        records = []
        max_attempts = rows_per_leaf * 1000

        for leaf in leaves:
            produced = 0
            attempts = 0

            while produced < rows_per_leaf:
                inp_list, out_list, target_value = self._generate_single_example()
                if self._value_matches_leaf(target_value, leaf):
                    records.append({
                        "input_list": str(inp_list),
                        "output_list": str(out_list),
                        "concept": leaf["label"]
                    })
                    produced += 1
                    attempts = 0
                    continue

                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        f"Unable to satisfy category '{leaf['label']}' with the provided conditions."
                    )

        random.shuffle(records)
        return records

    def save_all_splits(self, train=8000, val=1000, test=1000):
        """Generates and saves three separate Excel files."""
        splits = {
            "mlp_train.xlsx": train,
            "mlp_val.xlsx": val,
            "mlp_test.xlsx": test
        }

        for filename, count in splits.items():
            print(f"Generating {count} balanced rows for {filename}...")
            data_records = self._generate_balanced_records(count)

            df = pd.DataFrame(data_records)
            df.to_excel(filename, index=False)
            print(f"Successfully saved {filename}")

    def save_custom_set(self, filename, count):
        """Generates and saves a custom Excel file."""
        print(f"Generating {count} balanced rows for {filename}...")
        data_records = self._generate_balanced_records(count)

        df = pd.DataFrame(data_records)
        df.to_excel(filename, index=False)
        print(f"Successfully saved {filename}")


# --- Execution ---
if __name__ == "__main__":

    # Note: Your input_shape is (5, 2), matching your indexing logic
    train_set = MinimalGenerator(
        dim=(5, 2),
        range=(0, 10),
        dtype=int,
        categories={
            "pos": {
                "cond": lambda x: x > 0,
                "+00 < pos <= +05": {"cond": lambda x: ((x > 0) and (0 < x <= 5))},
                "+05 < pos <= +10": {"cond": lambda x: ((x > 0) and (5 < x <= 10))}
            },
            "neg": {
                "cond": lambda x: x < 0,
                "-05 <= neg < +00": {"cond": lambda x: ((x < 0) and (-5 <= x < 0))},
                "-10 <= neg < -05": {"cond": lambda x: ((x < 0) and (-10 <= x < -5))}
            }
        }
    )

    train_set.save_all_splits(
        train=8000,
        val=1000,
        test=1000
    )

    # Interpolation set with a narrower range to ensure more overlap between categories
    interpolation = MinimalGenerator(
        dim=(5, 2),
        range=(1, 9),
        dtype=int,
        categories={
            "pos": {
                "cond": lambda x: x > 0,
                "+00 < pos <= +05": {"cond": lambda x: ((x > 0) and (0 < x <= 5))},
                "+05 < pos <= +10": {"cond": lambda x: ((x > 0) and (5 < x <= 10))}
            },
            "neg": {
                "cond": lambda x: x < 0,
                "-05 <= neg < +00": {"cond": lambda x: ((x < 0) and (-5 <= x < 0))},
                "-10 <= neg < -05": {"cond": lambda x: ((x < 0) and (-10 <= x < -5))}
            }
        }
    )
    interpolation.save_custom_set(
        filename="interp_test.xlsx",
        count=1000
    )

    # Extrapolation set with a narrower range to ensure more overlap between categories
    extrapolation = MinimalGenerator(
        dim=(5, 2),
        range=(10, 20),
        dtype=int,
        categories={
            "pos": {
                "cond": lambda x: x > 0,
                "+00 < pos <= +05": {"cond": lambda x: ((x > 0) and (0 < x <= 5))},
                "+05 < pos <= +10": {"cond": lambda x: ((x > 0) and (5 < x <= 10))}
            },
            "neg": {
                "cond": lambda x: x < 0,
                "-05 <= neg < +00": {"cond": lambda x: ((x < 0) and (-5 <= x < 0))},
                "-10 <= neg < -05": {"cond": lambda x: ((x < 0) and (-10 <= x < -5))}
            }
        }
    )
    extrapolation.save_custom_set(
        filename="extrap_test.xlsx",
        count=1000
    )

    # Scaling set with a narrower range to ensure more overlap between categories
    scaling = MinimalGenerator(
        dim=(5, 2),
        range=(100, 110),
        dtype=int,
        categories={
            "pos": {
                "cond": lambda x: x > 0,
                "+00 < pos <= +05": {"cond": lambda x: ((x > 0) and (0 < x <= 5))},
                "+05 < pos <= +10": {"cond": lambda x: ((x > 0) and (5 < x <= 10))}
            },
            "neg": {
                "cond": lambda x: x < 0,
                "-05 <= neg < +00": {"cond": lambda x: ((x < 0) and (-5 <= x < 0))},
                "-10 <= neg < -05": {"cond": lambda x: ((x < 0) and (-10 <= x < -5))}
            }
        }
    )
    scaling.save_custom_set(
        filename="scaling_test.xlsx",
        count=1000
    )

    # Precision set with a narrower range to ensure more overlap between categories
    precision = MinimalGenerator(
        dim=(5, 2),
        range=(0.0, 10.0),
        dtype=float,
        categories={
            "pos": {
                "cond": lambda x: x > 0,
                "+00 < pos <= +05": {"cond": lambda x: ((x > 0) and (0 < x <= 5))},
                "+05 < pos <= +10": {"cond": lambda x: ((x > 0) and (5 < x <= 10))}
            },
            "neg": {
                "cond": lambda x: x < 0,
                "-05 <= neg < +00": {"cond": lambda x: ((x < 0) and (-5 <= x < 0))},
                "-10 <= neg < -05": {"cond": lambda x: ((x < 0) and (-10 <= x < -5))}
            }
        }
    )
    precision.save_custom_set(
        filename="precision_test.xlsx",
        count=1000
    )
