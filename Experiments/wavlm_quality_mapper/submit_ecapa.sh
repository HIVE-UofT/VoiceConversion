#!/bin/bash
#SBATCH --job-name=ecapa_analyze
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10
module load gcc arrow/22.0.0
source ~/envs/myenv/bin/activate

echo "=========================================="
echo "  ECAPA-TDNN Content Invariance Analysis"
echo "=========================================="

python scripts/analyze_ecapa.py --surgery Tonsill

echo "Done!"
