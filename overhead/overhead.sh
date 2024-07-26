#!/bin/bash

# Compile the latest program
make -C ../matmul matmul_xgpu

GPU_COUNT=1
ITERATIONS=10
KERNEL_LAUNCH_ITERATIONS=1000
TARGET_FILE_1="time_baseline_${GPU_COUNT}gpu.txt"
TARGET_FILE_2="time_overhead_${GPU_COUNT}gpu.txt"
PROGRAM_1="../matmul/matmul_xgpu $GPU_COUNT --iterations $KERNEL_LAUNCH_ITERATIONS"
PROGRAM_2="hpcrun -e gpu=level0,pc ../matmul/matmul_xgpu $GPU_COUNT --iterations $KERNEL_LAUNCH_ITERATIONS"

# Clear content of the target files
> $TARGET_FILE_1
> $TARGET_FILE_2

# Run the baseline program
for i in {1..$ITERATIONS}
do
    echo "Run $i:" >> $TARGET_FILE_1
    { time $PROGRAM_1 > /dev/null ; } 2>> $TARGET_FILE_1
    echo "" >> $TARGET_FILE_1
done

# Run the overhead program
for i in {1..$ITERATIONS}
do
    echo "Run $i:" >> $TARGET_FILE_2
    { time $PROGRAM_2 > /dev/null ; } 2>> $TARGET_FILE_2
    echo "" >> $TARGET_FILE_2
done
