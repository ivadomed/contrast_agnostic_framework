
GPU=2
set_slot $GPU CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python tools/generate_visualizations.py \
    --checkpoint checkpoints/v18_6/generator/t1w/run4/best_loss-epoch=024-train_loss=0.0000.ckpt \
    --num-samples 10 
set_slot $GPU CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python tools/generate_visualizations.py \
    --checkpoint checkpoints/v18_6/generator/t2w/run3/best_loss-epoch=024-train_loss=0.0000.ckpt \
    --num-samples 10