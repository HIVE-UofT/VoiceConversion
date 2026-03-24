#!/bin/bash
#SBATCH --job-name=compare_spksim
#SBATCH --account=def-zshakeri
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10

source .env/bin/activate

python compare_all_spksim.py
