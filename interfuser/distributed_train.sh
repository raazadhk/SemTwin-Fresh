#!/bin/bash

NUM_PROC=$1
shift

if [ "$NUM_PROC" -gt 1 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"
else
    python3 train.py "$@"
fi



