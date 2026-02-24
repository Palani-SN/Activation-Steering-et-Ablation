
# Activation Steering & Ablation: Experiment Pipeline

## Granular Codebase Reference & Experiment Walkthrough

This README provides a highly detailed, file-referenced guide to the pipeline, referencing specific scripts, classes, and functions for maximal transparency and reproducibility.

### 1. Dataset Generation & Loading
- **[dataset/data_generator.py](dataset/data_generator.py):** Implements `MLPExcelGenerator`, generating balanced samples with concept group assignment (pos_odd, pos_even, neg_odd, neg_even) via quadrant logic. Each sample is mapped to a group for steering and ablation.
- **[dataset/variant_generator.py](dataset/variant_generator.py):** Extends `MLPExcelGenerator` as `OODDataGenerator` to create out-of-distribution (OOD) variants, including float and integer extrapolation, for compliance testing.
- **[dataset/data_loader.py](dataset/data_loader.py):** Defines `CONCEPT_MAP` and `load_excel_to_dataloader`, converting Excel datasets to PyTorch DataLoader, ensuring concept tags are available for interpretability. Also provides `get_grouped_activations` for grouped latent analysis.

### 2. Model Definitions
- **[mlp/mlp_definition.py](mlp/mlp_definition.py):** Contains `InterpretabilityMLP`, a custom, wide MLP architecture with explicit layer naming and activation capture (`self.activations['layer2']`). Designed for SAE injection and interpretability.
- **[sae/sae_definition.py](sae/sae_definition.py):** Implements `SparseAutoencoder`, using top-K sparsification in latent space. The `forward` method returns both reconstruction and sparse hidden features, enabling feature discovery and manipulation.

### 3. Training Scripts
- **[train_mlp.py](train_mlp.py):** Function `train_to_perfection()` loads datasets, trains the MLP for 1000 epochs, logs progress, and saves weights. Uses AdamW optimizer and OneCycleLR scheduler for robust convergence.
- **[train_sae.py](train_sae.py):** Function `train_sae_from_payload()` loads harvested activations, trains SAE for 100-200 epochs, logs MSE, and saves the sparse dictionary. Focuses on reconstruction loss and top-K sparsity.

### 4. Activation Harvesting
- **[harvest_activations.py](harvest_activations.py):** Function `harvest_activations()` loads trained MLP, runs forward passes on dataset, captures layer2 activations and concept tags, and saves them as `harvested_data.pt` for SAE training. Logs sample counts and metadata.

### 5. Feature Probing & Steering Basis Extraction
- **[feature_probe.py](feature_probe.py):**
	- `get_universal_vectors()` computes steering basis vectors (v_sign, v_parity) by averaging SAE latents across concept groups.
	- `UniversalSteeringController` class loads MLP, SAE, and steering basis, and implements `steer_input()` for causal interventions in latent space.
	- `run_surgical_ablation()` performs targeted ablation, logs baseline and shifted outputs, and quantifies causal impact.
	- `get_top_k_features_by_group()` analyzes feature activations by group, supporting interpretability.

### 6. Compliance & Consistency Validation
- **[consistency_compliance.py](consistency_compliance.py):**
	- `SteeringValidator` class loads models and basis, implements `run_intervention()` for causal steering, `validate_dataset()` for compliance rate calculation, and `run_alpha_sweep()` for parameter sweeps and heatmap generation. Logs results and exports Excel reports.

### 7. Visualization Suite
- **[feature_reports.py](feature_reports.py):**
	- `load_trained_models()` loads MLP and SAE for visualization.
	- `plot_elegant_dual_compass()` visualizes steering basis geometry.
	- `plot_steering_performance_unified()` generates heatmaps and Pareto frontiers for compliance rates.
	- `plot_unified_logit_lens()` maps SAE features to logic categories, visualizing causal attribution and overlap.
	- All images are exported to [images/](images/) and referenced in logs.

---

## Experiment Logic & Scientific Rigor

Each script is modular and references explicit classes/functions for transparency. The pipeline enables:
- Discovery and manipulation of interpretable features
- Causal steering and ablation in latent space
- Compliance validation across in-distribution and OOD samples
- Quantitative and qualitative logging for reproducibility
- Scientific visualization for evidence-based claims

---

For further details, see inline comments and docstrings in each referenced file. This README is designed for top-tier AI research reproducibility and interpretability.

## Research-Level Guidance & Reproducibility

This pipeline is designed for maximum scientific rigor and reproducibility:

- **Modular Scripts:** Each phase is encapsulated in a dedicated script, with clear logging and error handling.
- **Automated Workflow:** The `workflow.bat` script ensures end-to-end execution, capturing start/end times and summarizing results.
- **Comprehensive Outputs:** All logs, images, and reports are saved for documentation and further analysis.
- **Interpretability:** The use of SAE and steering basis vectors enables direct causal attribution and manipulation.
- **Generalization:** Compliance testing across OOD variants validates robustness and scientific claims.

### For Future Work
- Extend to larger models or real-world datasets
- Integrate additional causal probes or ablation strategies
- Refine visualization and reporting for broader interpretability

---

For questions, scientific collaboration, or reproducibility requests, please contact the repository maintainer.
# Activation Steering & Ablation: Experiment Pipeline
# ...existing code...

## Output Images & Scientific Interpretation

All visualizations are generated automatically and exported to the `images/` folder. These include:

- **Steering Basis Compass:**
	- Visualizes the geometry of universal steering vectors (sign, parity) in latent space.
- **Performance Heatmaps & Pareto Frontier:**
	- Show compliance rates and trade-offs across datasets and steering strengths.
- **Unified Logit-Lens:**
	- Provides causal attribution and feature overlap, mapping SAE features to discovered logic categories.

Each image is accompanied by detailed captions and is referenced in the experiment logs. Regeneration is possible by rerunning `feature_reports.py`.

These visualizations provide scientific evidence for the causal structure and compliance of the steering and ablation interventions.
# Activation Steering & Ablation: Experiment Pipeline
# ...existing code...

## Log Outputs & Experiment Results

The pipeline produces detailed log outputs at each phase, demonstrating experiment progress and validation:

- **Training Logs:**
	- MLP and SAE training scripts print epoch progress, loss metrics, and completion status.
		- Example: `[██████░░░░░░░░░░░░░░░░░░░░░░░░] Epoch 10/100 | MSE: 0.000123 | 10.0%`
- **Activation Harvesting:**
	- Logs number of activations harvested and metadata saved.
		- Example: `[OK] Successfully saved 8000 activations with metadata.`
- **Feature Probing:**
	- Prints steering basis analysis, orthogonality checks, and ablation results.
		- Example: `STEERING BASIS VECTORS ANALYSIS` and `Causal Shift: +0.1234`
- **Compliance Validation:**
	- Reports heatmap generation and compliance rates for each dataset.
		- Example: `✓ Heatmap report generated: alpha_sweep_results.xlsx`
- **Visualization Generation:**
	- Confirms export of images and visualizations.
		- Example: `Successfully generated unified heatmap and Pareto frontier in /images.`

These logs provide transparency, reproducibility, and evidence for causal claims made in the experiment.
# Activation Steering & Ablation: Experiment Pipeline
# ...existing code...

## Experiment Logic: Steering, Ablation & Attribution

The experiment is structured to probe the causal structure of neural activations:

- **Feature Discovery:**
	- The SAE learns a sparse dictionary of features from MLP activations, enabling interpretable latent space manipulation.
- **Steering Basis Extraction:**
	- Universal steering vectors (sign, parity) are extracted, representing directions in latent space that causally control output logic.
- **Causal Steering:**
	- By intervening in SAE latent space (using `UniversalSteeringController`), the experiment demonstrates direct control over model output, validating causal relationships.
- **Surgical Ablation:**
	- Targeted ablation of features tests the necessity and sufficiency of discovered logic, with log outputs quantifying causal shifts.
- **Compliance Testing:**
	- The pipeline validates steering and ablation across in-distribution and OOD datasets, ensuring robustness and generalization.

All interventions are logged, with quantitative and qualitative results supporting causal claims.
# Activation Steering & Ablation: Experiment Pipeline
# ...existing code...

## Workflow Process & Scripts

The pipeline is orchestrated via `workflow.bat`, which automates all phases:

### Pipeline Phases
1. **Environment Activation:**
	- Activates the Python environment for reproducibility.
2. **Dataset Generation:**
	- Runs `data_generator.py` and `variant_generator.py` to create primary and OOD datasets.
3. **MLP Training:**
	- `train_mlp.py` trains a custom MLP to near-perfect accuracy on the generated dataset.
4. **Activation Harvesting:**
	- `harvest_activations.py` extracts hidden layer activations and concept tags, saving them for SAE training.
5. **Sparse Autoencoder Training:**
	- `train_sae.py` trains the SAE on harvested activations, discovering sparse, interpretable features.
6. **Feature Probing & Steering Basis Extraction:**
	- `feature_probe.py` analyzes SAE features, extracts universal steering vectors (sign, parity), and validates their orthogonality.
	- Conducts surgical ablation experiments to test causal impact.
7. **Consistency & Compliance Validation:**
	- `consistency_compliance.py` runs compliance checks across interpolation, extrapolation, and scaling datasets, generating heatmaps and Excel reports.
8. **Visualization Suite Generation:**
	- `feature_reports.py` produces comprehensive visualizations (compass, heatmaps, logit-lens) and exports them to the `images/` folder.

Each script is modular, with clear logging and output, enabling reproducible and interpretable experiments.
# Activation Steering & Ablation: Experiment Pipeline
# ...existing code...

## Problem Set & Dataset Logic

The core problem addressed is the **causal steering and ablation of neural activations** to understand and control model behavior. The dataset is generated to support this goal:

- **Primary Dataset Generation:**
	- `dataset/data_generator.py` creates a balanced set of integer and float samples, each tagged with logical concepts (e.g., sign, parity).
	- Each sample is mapped to a concept group via `CONCEPT_MAP`, enabling targeted interventions.
- **Variant Generation:**
	- `dataset/variant_generator.py` produces OOD variants for robust compliance testing, including interpolation, extrapolation, and scaling sets.
- **Data Loading:**
	- `dataset/data_loader.py` provides utilities to load Excel datasets into PyTorch dataloaders, ensuring concept tags are included for downstream interpretability.

The dataset structure is designed to facilitate:
- Discovery of interpretable features
- Controlled steering and ablation experiments
- Validation across both in-distribution and OOD samples
# Activation Steering & Ablation: Experiment Pipeline

## Overview

This repository implements a rigorous pipeline for **Activation Steering and Ablation** in neural networks, designed to probe, manipulate, and interpret learned representations in a controlled setting. The project leverages a custom Multi-Layer Perceptron (MLP) and Sparse Autoencoder (SAE) to discover, steer, and ablate latent features, with a focus on causal attribution and compliance testing.

### Research Context

The experiment is inspired by recent advances in mechanistic interpretability, aiming to:
- Identify and isolate interpretable features in hidden layers
- Steer model outputs via targeted interventions in latent space
- Validate causal relationships and compliance across in-distribution and out-of-distribution (OOD) samples
- Visualize and document the effects of steering and ablation with comprehensive log outputs and generated images

This README provides a detailed walkthrough of the problem set, workflow, scripts, experiment logic, log outputs, and visualizations, suitable for top-tier AI research standards.
