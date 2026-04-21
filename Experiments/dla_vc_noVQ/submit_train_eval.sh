#!/bin/bash
#SBATCH --job-name=dla_vc_noVQ
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/dla_vc_noVQ

echo "=============================="
echo "  DLA-VC (no VQ) — Training"
echo "=============================="
python scripts/train_split.py \
    --surgery Tonsill \
    --output results_tonsill_split \
    --seed 42

echo ""
echo "=============================="
echo "  DLA-VC (no VQ) — Evaluation"
echo "=============================="
python scripts/run_eval.py \
    --checkpoint results_tonsill_split/best_model.pth

echo "Done!"
