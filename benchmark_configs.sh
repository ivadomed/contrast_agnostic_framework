#!/usr/bin/env bash
# Benchmark different configurations for 3 epochs each
# Run each in sequence and measure total time

echo "=========================================="
echo "SPEED BENCHMARKING - 3 epochs each"
echo "=========================================="
echo ""

CONFIG_OVERRIDES=(
    # 1. BASELINE: No augmentation, no cache
    "training.generator.gpu_aug.enabled=false data.cache_rate=0.0"
    
    # 2. WITH AUGMENTATION (enabled)
    "training.generator.gpu_aug.enabled=true data.cache_rate=0.0"
    
    # 3. WITH CACHE (full cache, with aug)
    "training.generator.gpu_aug.enabled=true data.cache_rate=1.0"
    
    # 4. DISABLE COMPILE
    "training.generator.gpu_aug.enabled=true data.cache_rate=1.0 training.generator.compile_model=false"
    
    # 5. DISABLE DETERMINISTIC
    "training.generator.gpu_aug.enabled=true data.cache_rate=1.0 training.generator.compile_model=false training.deterministic=false"
    
    # 6. DISABLE BENCHMARK MODE
    "training.generator.gpu_aug.enabled=true data.cache_rate=1.0 training.generator.compile_model=false training.deterministic=false training.benchmark=false"
)

NAMES=(
    "NO AUG, NO CACHE"
    "WITH AUG, NO CACHE"
    "WITH AUG + CACHE"
    "NO AUG + CACHE + NO COMPILE"
    "NO AUG + CACHE + NO COMPILE + NO DETERMINISTIC"
    "NO AUG + CACHE + NO COMPILE + NO DETERMINISTIC + NO BENCHMARK"
)

echo "Each run: 3 epochs"
echo ""

for i in "${!CONFIG_OVERRIDES[@]}"; do
    NAME="${NAMES[$i]}"
    OVERRIDES="${CONFIG_OVERRIDES[$i]}"
    
    echo "=========================================="
    echo "TEST $((i+1)): $NAME"
    echo "Overrides: $OVERRIDES"
    echo "=========================================="
    
    START=$(date +%s.%N)
    
    timeout 1800 .venv/bin/python scripts/train.py \
        training.max_epochs.generator=3 \
        training.limit_train_batches=1.0 \
        training.num_sanity_val_steps=0 \
        training.log_every_n_steps=50 \
        training.generator.enable_image_logging=false \
        $OVERRIDES \
        2>&1 | tail -5
    
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    PER_EPOCH=$(echo "scale=1; $ELAPSED / 3" | bc)
    
    echo "Total: ${ELAPSED}s | Per epoch: ${PER_EPOCH}s"
    echo ""
done

echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
