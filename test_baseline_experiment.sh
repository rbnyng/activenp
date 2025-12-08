#!/bin/bash
# Quick test script to verify the baseline implementation works

echo "Testing RF baseline with spatial blocking and all strategies..."

python experiments/active_learning_experiment.py \
  --bbox -70.0 44.0 -69.0 45.0 \
  --n-seed 25 \
  --n-pool 100 \
  --n-iterations 3 \
  --samples-per-iter 5 \
  --strategies uncertainty hybrid_product \
  --model-types anp rf \
  --spatial-blocking \
  --test-fraction 0.2 \
  --sample-limit 200 \
  --output-dir ./results/test_baseline \
  --seed 42

echo ""
echo "Test complete! Check ./results/test_baseline for outputs."
