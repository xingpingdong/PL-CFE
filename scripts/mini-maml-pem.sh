#!/bin/bash
cd ..
out='./results/un_maml_pem/mini'
python train_un_maml_pem.py ./data --dataset miniimagenet --num-ways 5 --num-shots 1 \
      --num-shots-test 5 --num-epochs 100 --use-cuda --output-folder $out

python test.py $out'/config.json' --use-cuda --num-steps 50 --num-batches 125 --num-shots 1
python test.py $out'/config.json' --use-cuda --num-steps 50 --num-batches 125 --num-shots 5
python test.py $out'/config.json' --use-cuda --num-steps 50 --num-batches 125 --num-shots 20
python test.py $out'/config.json' --use-cuda --num-steps 50 --num-batches 125 --num-shots 50