#!/bin/bash

# Usage: ./run_sweep.sh --num_runs=<value>

# default

num_runs=0

for arg in "$@"
do
    case $arg in
        --num_runs=*)
        num_runs="${arg#*=}"
        shift
        ;;
        *)
        echo "Unknown argument: $arg"
        exit 1
        ;;
    esac
done

# validate
if ! [[ $num_runs =~ ^[0-9]+$ ]]
then
    echo "Invalid num_runs: $num_runs"
    exit 1
fi

# run number of runs
for (( run_num=0; run_num<=$num_runs; run_num++ ))
do
    echo "Run number: $run_num"
    python scripts/rsl_rl/sweep_inference.py --num_envs=4096 --run_num=$run_num --num_runs=$num_runs --headless
done