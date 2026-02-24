# Generated Visualizations

This folder contains all automatically generated visualization outputs from the **Activation Steering and Ablation** pipeline.

## Image Files

### Steering Basis Visualization
- **concept_compass.png** - Geometric visualization of steering basis vectors (v_sign, v_parity) and their orthogonality in 2D PCA space

### Performance Metrics
- **heatmap_Interpolation.png** - Steering success rates for in-distribution data
- **heatmap_Extrapolation.png** - Steering success rates for out-of-distribution (extrapolation) data
- **heatmap_Scaling.png** - Steering success rates for scaled magnitude data
- **heatmap_Precision.png** - Steering success rates for float precision data
- **pareto_frontier.png** - Trade-off curve between steering intensity (alpha) and total accuracy

### Causal Attribution Analysis
- **logit_lens_Distinct_Positive_Sign.png** - Causal impact of positive sign features on final output
- **logit_lens_Distinct_Negative_Sign.png** - Causal impact of negative sign features on final output
- **logit_lens_Distinct_Odd_Parity.png** - Causal impact of odd parity features on final output
- **logit_lens_Distinct_Even_Parity.png** - Causal impact of even parity features on final output

## Generation

These images are automatically generated when running `feature_reports.py` as part of the main pipeline. Each image is saved in PNG format with appropriate resolution for documentation purposes.

### Regenerating Images

To regenerate all visualizations:
```bash
python feature_reports.py
```

This will overwrite existing images with fresh outputs based on the current model state and sweep results.
