#!/bin/bash

DATA_DIR=./data
SEED=1
EPOCHS=30

for METHOD in phsim baseline
do
  echo "==== Running $METHOD seed${SEED} ===="

  for E in $(seq 1 $EPOCHS)
  do
    CKPT="checkpoints/upstream/${METHOD}/seed${SEED}/epoch${E}/simclr_${METHOD}_resnet18_epoch${E}_seed${SEED}.pt"

    if [ -f "$CKPT" ]; then
      echo "Processing epoch $E..."

      python eval_upstream_gamma_adv.py \
        --ckpt "$CKPT" \
        --data_dir "$DATA_DIR" \
        --eps_px 8 --steps 5 --alpha_px 2 \
        --per_class 50 --batch_size 256 \
        --out_csv "logs/upstream/${METHOD}/seed${SEED}/${METHOD}_seed${SEED}_gamma_adv.csv"

    else
      echo "Checkpoint not found: $CKPT"
    fi
  done
done

echo "==== Done ===="