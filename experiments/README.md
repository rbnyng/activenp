# Active Learning Experiments

This directory contains scripts for running active learning experiments with GEDI Neural Process models.

## Quick Start

Run the full active learning experiment:

```bash
python experiments/active_learning_experiment.py \
    --bbox -73.0 2.9 -72.9 3.0 \
    --n-seed 100 \
    --n-test 1000 \
    --n-iterations 15 \
    --samples-per-iter 10 \
    --strategies random uncertainty spatial hybrid \
    --output-dir ./results/active_learning \
    --device cuda
```

## Arguments

- `--bbox`: Bounding box for GEDI query (lon_min lat_min lon_max lat_max)
- `--n-seed`: Size of initial seed set (default: 100)
- `--n-test`: Size of test set (default: 1000)
- `--n-iterations`: Number of active learning iterations (default: 15)
- `--samples-per-iter`: Samples to acquire per iteration (default: 10)
- `--strategies`: List of strategies to compare (default: random uncertainty spatial hybrid)
- `--output-dir`: Directory to save results (default: ./results/active_learning)
- `--cache-dir`: Cache directory for GEDI/embeddings (default: ./cache)
- `--sample-limit`: Limit total samples for testing (optional)
- `--device`: Device to use (default: cuda if available, else cpu)

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
