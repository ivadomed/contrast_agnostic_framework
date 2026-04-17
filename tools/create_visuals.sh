
GPU=2
set_slot $GPU CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python tools/generate_visualizations.py \
    --checkpoint checkpoints/v19/generator/t1w/run1/best_loss-epoch=022-train_loss=0.0000.ckpt \
    --num-samples 10 
set_slot $GPU CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python tools/generate_visualizations.py \
    --checkpoint checkpoints/v19/generator/t2w/run1/best_loss-epoch=022-train_loss=0.0000.ckpt \
    --num-samples 10