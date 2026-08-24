#!/bin/bash

python trainer.py -cn cir_msiglip \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=1 \
    ++trainer.precision=bf16-mixed \
    \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-4 \
    \
    +lora=default \
    #+ckpt_path="'./trained_models/on_vn3k/warmupdone_ep5_score=38.00.ckpt'"
