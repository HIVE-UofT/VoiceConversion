#!/bin/bash
#SBATCH --job-name=eval_multisurg
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

# Multi-surgery baseline evaluation:
# Each method uses Tonsill train + Fess + Sept + Contr data to learn its
# transform, then tests on the same 5 Tonsill held-out patients.
# Goal: see if other surgery types' data improve Tonsill conversion.

EXPERIMENTS=/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments

echo "=============================="
echo "  kNN-VC (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/knn_vc
python scripts/run_eval_multisurg.py

echo "=============================="
echo "  Mean-Shift (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/mean_shift
python scripts/run_eval_multisurg.py

echo "=============================="
echo "  MKL-VC (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/mkl_vc
python scripts/run_eval_multisurg.py

echo "=============================="
echo "  LinearVC (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/linear_vc
python scripts/run_eval_multisurg.py

echo ""
echo "All multi-surgery baseline evaluations done!"
