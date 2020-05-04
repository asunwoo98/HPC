#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=jacobi2d
#SBATCH --mail-type=END
#SBATCH --mail-user=as10506@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load openmpi/gnu/4.0.2
mpirun -np 16 jacobi2d-mpi 16 1000
# n = ln * sqrt(p)
