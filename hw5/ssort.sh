#!/bin/bash
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=ssort
#SBATCH --mail-type=END
#SBATCH --mail-user=as10506@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load openmpi/gnu/4.0.2
mpirun -np 64 ssort
mpirun -np 64 ssort -n 10e5
mpirun -np 64 ssort -n 10e6
