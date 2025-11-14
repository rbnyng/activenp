# Active Learning Experiments

This directory contains scripts for running active learning experiments with GEDI Neural Process models.

## Quick Start

Run the full active learning experiment (Maine forests):

```bash
python experiments/active_learning_experiment.py \
    --bbox -70.0 44.0 -69.0 45.0 \
    --year 2022 \
    --n-seed 100 \
    --n-test 1000 \
    --n-iterations 15 \
    --samples-per-iter 10 \
    --strategies random uncertainty spatial hybrid \
    --output-dir ./results/active_learning \
    --device cuda
```

Or run with default arguments (Maine forests, 2022 embeddings):

```bash
python experiments/active_learning_experiment.py --device cuda
```

## Arguments

- `--bbox`: Bounding box for GEDI query (lon_min lat_min lon_max lat_max, default: Maine -70 44 -69 45)
- `--year`: Year for GeoTessera embeddings (default: 2022). GEDI data is queried from year±1 for temporal consistency.
- `--agbd-max`: Maximum AGBD threshold to filter outliers (default: 500 Mg/ha)
- `--n-seed`: Size of initial seed set (default: 100)
- `--n-test`: Size of test set (default: 1000)
- `--n-iterations`: Number of active learning iterations (default: 15)
- `--samples-per-iter`: Samples to acquire per iteration (default: 10)
- `--strategies`: List of strategies to compare (default: random uncertainty spatial hybrid)
- `--output-dir`: Directory to save results (default: ./results/active_learning)
- `--cache-dir`: Cache directory for GEDI/embeddings (default: ./cache)
- `--sample-limit`: Limit total samples for testing (optional)
- `--device`: Device to use (default: cuda if available, else cpu)

## Data Quality

The experiment includes several data quality controls:
- **Temporal matching**: GEDI data is queried from embedding year ±1 year for consistency
- **Outlier filtering**: AGBD values > 500 Mg/ha are filtered out (removes sensor errors)
- **Failed embeddings**: Samples with failed GeoTessera extractions are automatically removed

## Model Configuration

The experiment uses improved hyperparameters to prevent overfitting in few-shot regimes:
- **Architecture**: Full ANP (attention + latent paths) for better uncertainty in few-shot
- **Learning rate**: 5e-4 (reduced for stability)
- **Epochs per iteration**: 30 (reduced from 50 to prevent overfitting)
- **Hidden dim**: 512
- **Latent dim**: 128

## Sampling Strategies

### Random
Baseline random sampling from the pool.

### Uncertainty
Select samples with highest predictive uncertainty (standard deviation).

### Spatial
Maximize geographic diversity using maxmin sampling - iteratively select points farthest from existing training set.

### Hybrid
Combine uncertainty and spatial diversity: from top-k uncertain points, select samples that maximize spatial diversity.

## Output

The experiment produces:

```
results/active_learning/
├── config.json              # Experiment configuration
├── results.json             # Full results and histories
├── learning_curves.png      # Comparison plot
├── seed_df.pkl             # Initial seed data
├── pool_df.pkl             # Pool data
├── test_df.pkl             # Test data
├── random/
│   └── checkpoint_iter_*.pt
├── uncertainty/
│   └── checkpoint_iter_*.pt
├── spatial/
│   └── checkpoint_iter_*.pt
└── hybrid/
    └── checkpoint_iter_*.pt
```

## Expected Results

The hypothesis is that uncertainty-guided and hybrid sampling should achieve lower test RMSE with fewer samples compared to random sampling, demonstrating that active learning can reduce fieldwork requirements for conservation projects.

## Small Test Run

For quick testing with fewer samples:

```bash
python experiments/active_learning_experiment.py \
    --bbox -73.0 2.9 -72.9 3.0 \
    --sample-limit 500 \
    --n-seed 50 \
    --n-test 100 \
    --n-iterations 5 \
    --samples-per-iter 10 \
    --strategies random uncertainty \
    --output-dir ./results/test \
    --device cpu
```
