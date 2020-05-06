#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=ssort
#SBATCH --mail-type=END
#SBATCH --mail-user=as10506@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load openmpi/gnu/4.0.2
mpirun -np 64 ssort -n 1000000
