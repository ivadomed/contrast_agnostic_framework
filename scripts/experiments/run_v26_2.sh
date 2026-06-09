#!/bin/bash
source ~/.bashrc
set_slot 0 .venv/bin/python scripts/generate_synthetic_guidance.py --generator v26_2 --lhc --rank 0 --world-size 4 --device cuda:0 > /tmp/gen0.log 2>&1 & P0=$!
set_slot 1 .venv/bin/python scripts/generate_synthetic_guidance.py --generator v26_2 --lhc --rank 1 --world-size 4 --device cuda:1 > /tmp/gen1.log 2>&1 & P1=$!
set_slot 2 .venv/bin/python scripts/generate_synthetic_guidance.py --generator v26_2 --lhc --rank 2 --world-size 4 --device cuda:2 > /tmp/gen2.log 2>&1 & P2=$!
set_slot 3 .venv/bin/python scripts/generate_synthetic_guidance.py --generator v26_2 --lhc --rank 3 --world-size 4 --device cuda:3 > /tmp/gen3.log 2>&1 & P3=$!

wait $P0 $P1 $P2 $P3
echo "Generation done!"
