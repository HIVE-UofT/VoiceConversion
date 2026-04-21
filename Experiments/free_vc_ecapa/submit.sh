#!/bin/bash
#SBATCH --job-name=freevc_ecapa
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EXP=/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc_ecapa

echo "================================================"
echo "  FreeVC + ECAPA shift + bridge — Training"
echo "================================================"
python "$EXP/scripts/train.py"

echo ""
echo "================================================"
echo "  FreeVC + ECAPA shift + bridge — Eval (4 strategies)"
echo "================================================"
python "$EXP/scripts/run_eval.py"

echo "Done!"
