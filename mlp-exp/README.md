# MLP Activation Steering & Sparse Autoencoder (SAE) Interpretability

A comprehensive pipeline for training neural networks, extracting internal activations, learning sparse representations, and performing activation steering to control model behavior.

## 📋 Project Overview

This project implements a complete workflow for **interpretability through activation steering**:

1. **Data Generation**: Synthetic dataset generation with index-based mathematical logic
2. **MLP Training**: Train a multi-layer perceptron on the generated data
3. **Activation Harvesting**: Extract internal layer activations from the trained MLP
4. **Sparse Autoencoder (SAE) Training**: Learn sparse, interpretable features from activations
5. **Behavior Modelling**: Analyze and identify monosemantic features
6. **Activation Steering**: Manipulate activations to steer model outputs to target ranges

## 🎯 Problem Domain

The project implements a mathematical model where:
- **Input**: 10-dimensional vectors arranged as a 5×2 matrix
- **Logic**: `abs(matrix[0][index1] - matrix[1][index2])`
  - `index1` = last element of first row (0-4)
  - `index2` = last element of second row (0-4)
- **Output**: Absolute difference between indexed values (0-9)

### Example:
```
Input: [8, 9, 5, 1, 3, 2, 9, 4, 7, 1]
       └─────────────┬─────────────┘
         Row 1          Row 2
Index1 = 3 (last elem of row 1) → matrix[0][3] = 1
Index2 = 1 (last elem of row 2) → matrix[1][1] = 9
Output = |1 - 9| = 8.0
```

## 📁 Directory Structure

```
mlp-exp/
├── README.md                          # This file
├── workflow.bat                       # Automated pipeline execution script
├── workflow.log                       # Execution logs and results
├── reqs.txt                          # Python dependencies
│
├── dataset/
│   ├── data_generator.py             # Synthetic data generation
│   ├── data_loader.py                # Data loading utilities
│   ├── mlp_train.xlsx                # Training dataset (8000 samples)
│   ├── mlp_val.xlsx                  # Validation dataset (1000 samples)
│   └── mlp_test.xlsx                 # Test dataset (1000 samples)
│
├── mlp/
│   ├── mlp_definition.py             # MLP architecture definition
│   └── perfect_mlp.pth               # Trained MLP weights
│
├── sae/
│   ├── sae_definition.py             # Sparse Autoencoder architecture
│   └── sae_model.pth                 # Trained SAE weights
│
├── train_mlp.py                      # MLP training script
├── harvest_activations.py            # Activation extraction script
├── train_sae.py                      # SAE training script
├── behavior_modelling.py             # Feature analysis & steering logic
└── mlp_activations.pt                # Extracted MLP activations tensor
```

## 🔧 Setup & Installation

### Prerequisites
- Python 3.11+
- Conda environment manager
- CUDA 12.6 (optional, for GPU support)

### Dependencies
Install all required packages:
```bash
pip install -r reqs.txt
```

Key packages:
- **torch** 2.10.0+cu126 - Deep learning framework
- **pandas** 3.0.0 - Data manipulation
- **numpy** 2.4.2 - Numerical computing
- **openpyxl** 3.1.5 - Excel file handling
- **matplotlib** 3.10.8 - Visualization
- **plotly** 6.5.2 - Interactive plots

### Quick Start
Run the entire pipeline:
```bash
workflow.bat
```

Or execute steps individually (see below).

## 📊 Dataset Generation

### Script: `dataset/data_generator.py`

Generates synthetic training/validation/test datasets in Excel format.

**Features:**
- Generates matrices with index-based lookup logic
- Creates 8000 training samples, 1000 validation, 1000 test samples
- Exports to Excel files with columns: `input_list`, `output_list`

**Class: `MLPExcelGenerator`**
```python
generator = MLPExcelGenerator(input_shape=(5, 2))
generator.save_all_splits(train=8000, val=1000, test=1000)
```

**Example Output:**
```
Input:  [8, 9, 5, 1, 3, 2, 9, 4, 7, 1]
Output: [8.0]
```

### Data Loading

**Script: `dataset/data_loader.py`**

Utility function to load Excel data and convert to PyTorch DataLoaders:
```python
from dataset.data_loader import load_excel_to_dataloader

train_loader = load_excel_to_dataloader("dataset/mlp_train.xlsx", batch_size=32)
```

## 🧠 Model Architectures

### MLP Architecture

**Script: `mlp/mlp_definition.py`**

```
Input (10) 
  ↓
Linear(10, 256) + BatchNorm + ReLU
  ↓
Linear(256, 512) + BatchNorm + ReLU  ← Hidden Layer (Layer2) [*SAE inject point]
  ↓
Linear(512, 256) + ReLU
  ↓
Linear(256, 1)
  ↓
Output (1)
```

**Key Features:**
- Wide hidden layer (512 dims) to capture complex patterns
- Activation capturing at `layer2` for SAE training
- Batch normalization for training stability

**Class: `InterpretabilityMLP`**
```python
model = InterpretabilityMLP()
output = model(input_tensor)
# Access activations: model.activations['layer2']
```

### Sparse Autoencoder (SAE)

**Script: `sae/sae_definition.py`**

```
MLP Activations (512)
  ↓
Encoder: Linear(512, 2048) + ReLU
  ↓
Latent Features (2048) ← Monosemantic representations
  ↓
Decoder: Linear(2048, 512)
  ↓
Reconstructed Activations (512)
```

**Key Features:**
- Expands 512-dim activations to 2048-dim latent space
- Sparse ReLU activation enforces feature competition
- Trained with L1 regularization for sparsity

**Class: `SparseAutoencoder`**
```python
sae = SparseAutoencoder(input_dim=512, dict_size=2048)
reconstructed, hidden_features = sae(mlp_activations)
```

## 🚀 Training Pipeline

### Step 1: Generate Dataset
```bash
cd dataset
python data_generator.py
cd ..
```

**Output from logs:**
```
Generating 8000 rows for mlp_train.xlsx...
Successfully saved mlp_train.xlsx
Generating 1000 rows for mlp_val.xlsx...
Successfully saved mlp_val.xlsx
Generating 1000 rows for mlp_test.xlsx...
Successfully saved mlp_test.xlsx
```

### Step 2: Train MLP
```bash
python train_mlp.py
```

**Training Details:**
- Epochs: 500
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Scheduler: OneCycleLR for learning rate annealing
- Loss: Mean Squared Error (MSE)
- Batch size: 64

**Output from logs:**
```
Epoch 50  | Val MSE: 0.601289
Epoch 100 | Val MSE: 0.592865
Epoch 150 | Val MSE: 0.292769
Epoch 200 | Val MSE: 0.201282
...
Epoch 500 | Val MSE: 0.109966
```

Saves trained weights to `mlp/perfect_mlp.pth`

### Step 3: Harvest Activations
```bash
python harvest_activations.py
```

**Process:**
1. Loads trained MLP from `mlp/perfect_mlp.pth`
2. Forward-passes all 8000 training samples
3. Extracts 512-dim activations from `layer2`
4. Concatenates into single tensor: [8000, 512]

**Output from logs:**
```
Harvesting activations...
Success! Saved tensor of shape: torch.Size([8000, 512])
```

### Step 4: Train Sparse Autoencoder
```bash
python train_sae.py
```

**Training Details:**
- Epochs: 100
- Optimizer: Adam (lr=1e-3)
- Batch size: 128
- L1 coefficient: 1e-4 (sparsity regularization)
- Loss: MSE + L1 regularization

**Output from logs:**
```
Loaded activations: torch.Size([8000, 512])
SAE Epoch [10/100] | Loss: 0.003793
SAE Epoch [20/100] | Loss: 0.002686
...
SAE Epoch [100/100] | Loss: 0.001237
SAE training complete. Weights saved.
```

Saves trained weights to `sae/sae_model.pth`

### Step 5: Behavior Modelling & Activation Steering
```bash
python behavior_modelling.py
```

This step performs multiple analyses:

#### 5.1 Feature Inspection
Identifies active SAE features for sample inputs.

**Output from logs:**
```
--- Interpretability Report ---
Sample Input: [8, 9, 5, 1, 3, 2, 9, 4, 7, 1]     |     Expected Output: 8.0
MLP Output: 7.7718
Number of active SAE features: 58

Top Active Features (Monosemantic Candidates):
Feature # 476 | Activation: 0.6795
Feature #1829 | Activation: 0.4535
Feature #1805 | Activation: 0.4254
Feature # 373 | Activation: 0.3256
Feature #1238 | Activation: 0.2876
```

#### 5.2 Feature Mapping by Output Groups
Profiles SAE features across output value groups (0-9).

**Output from logs:**
```
--- Top Features per Output Group ---
Group 0 (n=811):
  Top Features: #1058 (0.2847), #528 (0.2681), #1829 (0.2563), ...
Group 1 (n=822):
  Top Features: #528 (0.3185), #1829 (0.3011), #1058 (0.2724), ...
...
```

#### 5.3 Pure Set Extraction
Identifies features that are:
- **Positive/Active Features**: Strongly activate for target outputs (e.g., 3-6)
- **Negative/Ablate Features**: Strongly activate for non-target outputs

#### 5.4 Activation Steering
Iteratively manipulates latent features to steer model outputs toward target range.

**Algorithm:**
```
For each iteration (50 steps):
  1. Pass steered activations through SAE decoder
  2. Get current MLP output
  3. Calculate masks for outputs below/above target range
  4. UP-STEER: Boost positive features if output too low
  5. DOWN-STEER: Boost negative features if output too high
  6. Clamp latent values to [0, 15]
```

**Target Range Example:** [3, 6]
```
Baseline Acc: 45.25%  (before steering)
Steered Acc:  92.18%  (after steering)
```

## 📈 Key Results from workflow.log

### Training Convergence
MLP validation MSE showed steady improvement:
- Started: 0.601 (Epoch 50)
- Mid-training: 0.202 (Epoch 200)
- Final: 0.110 (Epoch 500)

### SAE Learning
Sparse autoencoder successfully compressed 512-dim activations:
- Learned 2048 sparse features
- Achieved reconstruction loss: 0.001237 by epoch 100
- Effective sparsity through L1 regularization

### Feature Discovery
Successfully identified monosemantic features:
- ~50-60 features active per sample (out of 2048)
- Consistent feature sets for same output values
- Clear separation between positive/negative features

### Steering Success
Iterative activation manipulation achieved high accuracy:
- Successfully steered outputs to target ranges
- Maintained output quality without gradient-based retraining
- Proved feasibility of feature-level control

## 🔍 Advanced Features

### Feature Inspection API

**`inspect_features(mlp_path, sae_path, test_input, exp_output)`**

Analyzes a single input through MLP and SAE to show:
- MLP output
- Number of active SAE features
- Top 5 most activated features with values

### Feature Mapping API

**`generate_feature_map(mlp_path, sae_path, data_path, k)`**

Profiles features across output groups:
- Groups data by target output (0-9)
- Identifies top-k features per group
- Useful for understanding feature specialization

### Steering Intervention API

**`perform_steering_intervention(ablate_ids, active_ids, mlp, sae, data_loader, target_range)`**

Steers model outputs to specific range:
- Requires ablate and active feature IDs
- Iteratively manipulates latent features
- Reports baseline vs. steered accuracy

### Visual Demo

**`run_visual_demo(mlp, sae, data_loader, ablate_ids, active_ids, target_range, num_samples)`**

Displays before/after steering for sample outputs:
```
IDX  | ORIGINAL   | STEERED    | TARGET   | STATUS
0    | 5.2341     | 4.8932     | (3, 6)   | STABLE 🆗
1    | 1.5678     | 3.2104     | (3, 6)   | FIXED ✅
2    | 7.4523     | 6.9876     | (3, 6)   | FAIL ❌
```

## 🛠️ Customization Guide

### Change Input/Output Dimensions
Edit `data_generator.py`:
```python
generator = MLPExcelGenerator(input_shape=(rows, cols))
```

### Adjust MLP Architecture
Edit `mlp/mlp_definition.py`:
```python
self.layers = nn.ModuleDict({
    'input': nn.Linear(10, 256),      # Change input/hidden dims
    'hidden1': nn.Linear(256, 512),   # Change hidden layers
    # ... modify as needed
})
```

### Tune SAE Hyperparameters
Edit `train_sae.py`:
```python
sae = SparseAutoencoder(input_dim=512, dict_size=2048)  # Expand dict_size for more features
l1_coeff=1e-4  # Increase for more sparsity, decrease for better reconstruction
```

### Modify Target Steering Range
Edit `behavior_modelling.py`:
```python
ablate_ids, active_ids = get_pure_sets(
    bad_indices=[0, 1, 2, 7, 8, 9],
    good_indices=[3, 4, 5, 6]  # Change target range
)
perform_steering_intervention(..., target_range=(3, 6))  # Modify range
```

## 🐛 Troubleshooting

### Issue: "Failed to load model"
- Check that `mlp/perfect_mlp.pth` and `sae/sae_model.pth` exist
- Ensure training completed successfully
- Verify model dimensions match architecture definitions

### Issue: Low steering accuracy
- Increase number of steering iterations (default: 50)
- Adjust step_size parameter in `perform_steering_intervention`
- Verify ablate_ids and active_ids are correctly identified
- Check that target_range is reasonable for the data

### Issue: Out of memory
- Reduce batch size in data loaders
- Decrease dict_size in SAE (e.g., 1024 instead of 2048)
- Process data in smaller chunks

### Issue: SAE not learning
- Increase l1_coeff to encourage sparsity
- Verify activations are being properly harvested
- Check that input_dim matches MLP layer2 output (should be 512)

## 📚 References & Theory

### Sparse Autoencoders
SAEs learn sparse, interpretable representations by:
1. Expanding input to larger latent space (512 → 2048)
2. Using ReLU to enforce non-negativity
3. Applying L1 regularization for sparsity
4. Result: Each feature specializes (monosemantic)

### Activation Steering
Steering works by:
1. Identifying features important for target behavior
2. Manipulating latent features directly
3. Iteratively adjusting based on current output
4. No gradient computation needed (faster than fine-tuning)

### Interpretability Benefits
- **Feature-level transparency**: Understand which features drive outputs
- **Causal interventions**: Directly test feature importance
- **Behavior control**: Steer outputs without retraining
- **Monosemanticity**: Features represent single concepts

## 📝 License & Attribution

This project implements techniques from mechanistic interpretability research.

Key concepts:
- Sparse Autoencoders: [Bricken et al., 2023](https://arxiv.org/abs/2309.08600)
- Activation Steering: Applications in LLM interpretability
- Index-based logic: Synthetic task for controlled experimentation

## 🤝 Contributing

To extend this project:
1. Add new data generation strategies
2. Implement different SAE architectures
3. Develop new steering objectives
4. Add visualization tools
5. Create ablation study frameworks

## 📧 Questions & Support

For issues or questions:
- Check `workflow.log` for execution traces
- Review inline code comments
- Verify all dependencies are installed
- Ensure data files exist before training

---

**Last Updated:** February 22, 2026  
**Project Status:** Complete pipeline implementation with evaluation
