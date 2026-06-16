#!/bin/bash

python trainer.py -cn cir_msiglip \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=3 \
    ++trainer.precision=16-mixed \
    \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-4 \
    \
    +lora=default \
    +ckpt_path="'outputs/2026-06-16/00-30-10/checkpoints/epoch=42-val_score=50.23.ckpt'"
