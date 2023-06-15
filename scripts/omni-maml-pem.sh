#!/bin/bash
cd ..
out='./results/un_maml_pem/omni'
python train_un_maml_pem.py ./data --dataset omniglot --num-ways 5 --num-shots 1 \
      --num-shots-test 5 --num-epochs 20 --use-cuda --output-folder $out --n-warmup 5

python test.py $out'/config.json' --use-cuda --num-steps 50 --num-batches 125 --num-shots 1
python test.py $out'/config.json' --use-cuda --num-steps 50 --num-batches 125 --num-shots 5