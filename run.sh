#!/usr/bin/env bash

python CS.py \
      -acfg configs/adapter/imagenet/DCF.yaml \
      -dcfg configs/dataset/imagenet.yaml \
      -ocfg configs/order/imagenet/0.yaml \
      -mcfg configs/models/resnet50.yaml \
      SEED 0 \
      TEST.BATCH_SIZE 64 \
      CORRUPTION.SEVERITY "[5]" \
      NOTE "test" \
      bash_file_name "$(basename "$0")"
