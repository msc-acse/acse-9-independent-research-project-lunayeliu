#PBS -N tetra_030
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=2:mpiprocs=1:mem=1gb


# Move back to the current directory.
cd $PBS_O_WORKDIR
module load anaconda3/personal

python3 ./run_generation_tetra_030.py