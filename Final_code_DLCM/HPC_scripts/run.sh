#!/bin/bash
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=32:mem=124gb:ompthreads=48
#PBS -N DLCM_run

cd $PBS_O_WORKDIR
module load anaconda3/personal
python3 ./run_generation.py