#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

EXPERIMENTS=/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments

echo "=============================="
echo "  kNN-VC"
echo "=============================="
cd $EXPERIMENTS/knn_vc
python scripts/run_eval.py

echo "=============================="
echo "  Mean-Shift"
echo "=============================="
cd $EXPERIMENTS/mean_shift
python scripts/run_eval.py

echo "=============================="
echo "  MKL-VC"
echo "=============================="
cd $EXPERIMENTS/mkl_vc
python scripts/run_eval.py

echo "=============================="
echo "  Linear-VC"
echo "=============================="
cd $EXPERIMENTS/linear_vc
python scripts/run_eval.py

echo ""
echo "All baseline evaluations done!"
