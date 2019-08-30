#!/bin/bash
#PBS -lwalltime=00:30:00
#PBS -lselect=1:ncpus=8:mem=96gb:ompthreads=8
#PBS -N DLCM


module load anaconda3/personal
python3 ./run_generation.py