#!/bin/bash
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=48:mem=124gb:ompthreads=48
#PBS -N DLCM_run_04

cd $PBS_O_WORKDIR
module load anaconda3/personal
python3 ./run_generation_04.py