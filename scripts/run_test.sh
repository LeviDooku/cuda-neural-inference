#!/bin/bash

OUTPUT="data/results.dat"

echo "# samples cpu_ms gpu_total_ms gpu_kernel_ms cpu_acc gpu_acc memory_mb" > "$OUTPUT"

TRAINING=10000
EPOCH=1000
LR=0.01

for SAMPLES in 100 1000 10000 100000 1000000 5000000 10000000 20000000
do
    ./test $TRAINING $EPOCH $LR $SAMPLES >> $OUTPUT
done