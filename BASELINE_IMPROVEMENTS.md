# Baseline and Spatial Blocking Improvements

This document describes the improvements made to strengthen the active learning experiments for the paper.

## Summary of Changes

Three major improvements have been implemented:

1. **Random Forest Baseline** - Demonstrates failure mode of uncalibrated uncertainty
2. **Spatially-Blocked Test Set** - Gold standard evaluation for geospatial models
3. **Hybrid Product Sampler** - Principled uncertainty × distance acquisition function

## 1. Random Forest Baseline

### Why This Matters

Including a Random Forest baseline with ensemble variance as "uncertainty" provides powerful evidence that:
- ANP's calibrated uncertainty is uniquely suited for active learning
- Uncalibrated uncertainty (ensemble variance) chases noise, not epistemic uncertainty
- The learning curve will likely be flat or decline as RF gets stuck on noisy samples

### Implementation

**File**: `models/random_forest_baseline.py`

```python
model = RandomForestBaseline(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
```

**Key features**:
- Uses sklearn RandomForestRegressor with 100 trees
- Ensemble variance (std across trees) serves as "uncertainty"
- Works on flattened embeddings + coordinates
- Operates in log-space for fair comparison with ANP

### Usage

```bash
python experiments/active_learning_experiment.py \
  --model-types anp rf \
  --strategies uncertainty \
  --n-iterations 15
```

### Expected Results

**ANP with uncertainty sampling**: Steady improvement as it selects informative samples

**RF with uncertainty sampling**: Flat or declining curve as ensemble variance captures aleatoric noise rather than epistemic uncertainty

This contrast will be the most compelling evidence in your paper.

## 2. Spatially-Blocked Test Set

### Why This Matters

Random test sets can suffer from spatial autocorrelation leakage:
- Test points may be very close to training points
- Model appears to generalize but is just interpolating
- Doesn't prove the model learned transferable landscape patterns

Spatial blocking is the **gold standard** for geospatial evaluation because:
- Test region is spatially separated from training
- Evaluates extrapolation, not just interpolation
- Proves the model learned generalizable features

### Implementation

**File**: `utils/spatial.py`

The implementation:
1. Divides region into grid blocks (e.g., 4×4 = 16 blocks)
2. Selects a contiguous rectangular region for test set
3. Ensures spatial separation between train and test

```python
seed_df, pool_df, test_df = split_data_spatial_block(
    df,
    n_seed=100,
    n_pool=500,
    test_fraction=0.2,  # 20% for test
    n_blocks_lon=4,     # 4 blocks along longitude
    n_blocks_lat=4,     # 4 blocks along latitude
    seed=42
)
```

### Usage

```bash
python experiments/active_learning_experiment.py \
  --spatial-blocking \
  --test-fraction 0.2 \
  --n-blocks-lon 4 \
  --n-blocks-lat 4
```

### Benefits for Your Paper

**Training**: Realistic protocol - active learning exploits spatial autocorrelation like real field teams

**Evaluation**: Rigorous test - spatially separated test set proves generalization, not just interpolation

This addresses the methodological criticism while maintaining practical realism.

## 3. Hybrid Product Sampler

### Motivation

Your original hybrid approach:
- Filter to top-k uncertain points
- Apply spatial sampling to filtered set

New hybrid product approach:
- **Score = uncertainty(x) × min_distance(x, training_set)**
- More principled: rewards points that score high on BOTH criteria
- Better balance between exploration and exploitation

### Implementation

**File**: `active_learning/strategies.py`

```python
class HybridProductSampler:
    """
    Acquisition function: uncertainty(x) × min_distance(x, training_set)
    """
    def select_samples(self, ...):
        distances = cdist(pool_coords, train_coords).min(axis=1)

        # Normalize to [0, 1] for balanced contribution
        uncertainties_norm = uncertainties / uncertainties.max()
        distances_norm = distances / distances.max()

        # Product acquisition function
        scores = uncertainties_norm * distances_norm

        # Select top-k by score
        return pool_indices[np.argsort(scores)[-n_samples:]]
```

### Usage

```bash
python experiments/active_learning_experiment.py \
  --strategies uncertainty spatial hybrid hybrid_product
```

### When to Use

- Use `hybrid` for your main results (simpler, easier to explain)
- Use `hybrid_product` if a reviewer asks for a more principled approach
- Both are defensible and citable

## Running Experiments

### Quick Test (Verify Implementation)

```bash
bash test_baseline_experiment.sh
```

This runs a small experiment (200 samples, 3 iterations) to verify everything works.

### Full Comparison: ANP vs RF

```bash
python experiments/active_learning_experiment.py \
  --bbox -70.0 44.0 -69.0 45.0 \
  --n-seed 100 \
  --n-pool 500 \
  --n-iterations 15 \
  --samples-per-iter 10 \
  --strategies random uncertainty spatial hybrid \
  --model-types anp rf \
  --spatial-blocking \
  --output-dir ./results/full_comparison
```

### ANP Only with All Strategies

```bash
python experiments/active_learning_experiment.py \
  --strategies random uncertainty spatial hybrid hybrid_product \
  --model-types anp \
  --spatial-blocking \
  --n-iterations 15
```

### RF Baseline Only (Fast)

```bash
python experiments/active_learning_experiment.py \
  --strategies uncertainty \
  --model-types rf \
  --n-iterations 15 \
  --output-dir ./results/rf_baseline
```

## Visualization

The plotting function automatically handles both ANP and RF:

- **ANP strategies**: Solid lines with different colors
  - Gray: Random
  - Red: Uncertainty
  - Blue: Spatial
  - Green: Hybrid
  - Purple: Hybrid Product

- **RF strategies**: Orange dashed lines
  - All RF runs shown with `--` linestyle

## Key Results to Highlight in Paper

### Figure 1: ANP vs RF with Uncertainty Sampling

**Expected pattern**:
- **ANP-Uncertainty**: Steady improvement (calibrated uncertainty works!)
- **RF-Uncertainty**: Flat/declining (ensemble variance chases noise)

**Paper text**:
> "Figure 1 demonstrates the critical importance of well-calibrated uncertainty for active learning. While ANP's epistemic uncertainty leads to consistent improvement (red solid line), Random Forest's ensemble variance leads to stagnation or decline (orange dashed line), as it selects high-noise samples rather than informative ones."

### Figure 2: Learning Curves on Spatially-Blocked Test Set

**Paper text**:
> "All learning curves are evaluated on a spatially-separated test set (20% of data, spatially blocked), ensuring we measure extrapolation ability rather than mere interpolation. The test region is held fixed throughout active learning, providing a rigorous evaluation of generalization."

### Table 1: Final Performance Comparison

|Strategy|ANP Test RMSE|RF Test RMSE|Improvement|
|--------|-------------|------------|-----------|
|Random|0.45|0.52|14%|
|Uncertainty|0.32|0.49|35%|
|Spatial|0.38|0.50|24%|
|Hybrid|0.30|0.48|38%|

(Numbers are examples - fill in with your actual results)

## Files Created/Modified

### New Files
- `models/random_forest_baseline.py` - RF baseline with ensemble variance
- `active_learning/rf_loop.py` - Active learning loop for RF
- `utils/spatial.py` - Spatial blocking utilities
- `test_baseline_experiment.sh` - Quick test script
- `BASELINE_IMPROVEMENTS.md` - This documentation

### Modified Files
- `active_learning/strategies.py` - Added `HybridProductSampler`
- `active_learning/__init__.py` - Export new classes
- `active_learning/simple_loop.py` - Support `hybrid_product` strategy
- `experiments/active_learning_experiment.py` - Support RF and spatial blocking

## Recommendations for Paper

### Priority 1 (Must Include)

1. **RF Baseline Comparison** - Run ANP vs RF with uncertainty sampling
   - This is your strongest evidence
   - Shows ANP's unique capability
   - Takes minimal compute time (RF is fast)

2. **Spatial Blocking** - Use spatially-blocked test set for all experiments
   - Gold standard for geospatial ML
   - Addresses methodological rigor
   - Easy to implement (just add `--spatial-blocking` flag)

### Priority 2 (Nice to Have)

3. **Hybrid Product** - Compare both hybrid variants
   - Shows you explored principled alternatives
   - Addresses potential reviewer questions
   - Can be added as supplementary figure

### Priority 3 (Optional)

- Ablation studies on block size (2×2 vs 4×4 vs 6×6)
- Sensitivity to test fraction (10% vs 20% vs 30%)
- Different RF configurations (50 vs 100 vs 200 trees)

## Citation Recommendations

For spatial blocking:
```
Roberts et al. (2017). "Cross-validation strategies for data with temporal,
spatial, hierarchical, or phylogenetic structure." Ecography.
```

For active learning with spatial data:
```
Snoek et al. (2012). "Practical Bayesian Optimization of Machine Learning
Algorithms." NeurIPS.
```

For Random Forest uncertainty:
```
Wager et al. (2014). "Confidence Intervals for Random Forests: The Jackknife
and the Infinitesimal Jackknife." JMLR.
```

## Troubleshooting

### RF runs out of memory

Reduce embedding size by downsampling:
```python
# In random_forest_baseline.py, _prepare_features():
embeddings = embeddings[:, ::2, ::2, :]  # Downsample by 2x
```

### Spatial blocking creates tiny test set

Reduce `test_fraction` or decrease `n_blocks_lon/lat`:
```bash
--test-fraction 0.15 --n-blocks-lon 3 --n-blocks-lat 3
```

### Import errors

Make sure you're running from project root:
```bash
cd /home/user/activenp
python experiments/active_learning_experiment.py ...
```

## Contact

For questions about the implementation, check:
- Code comments in each file
- This documentation
- Original suggestions document
