#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=jacobi2d
#SBATCH --mail-type=END
#SBATCH --mail-user=as10506@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load openmpi/gnu/4.0.2
mpirun -np 64 jacobi2d-mpi 200 1000
# n = ln * sqrt(p)
# 1600 = 100 * 16
# n = 1600