#!/bin/bash
#SBATCH --job-name=dla_vc_aug
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

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/dla_vc

echo "================================================"
echo "  DLA-VC + Audio Augmentation — Training"
echo "  Augmentation: pitch shift (+-2 st), time stretch (0.92-1.08),"
echo "  gain (+-3 dB), Gaussian noise. Same params per (pre, post) pair."
echo "  Test patients (0045,0085,0110,0122,0132) excluded from training."
echo "================================================"
python scripts/train_split.py \
    --surgery Tonsill \
    --output results_tonsill_aug \
    --seed 42

echo ""
echo "================================================"
echo "  DLA-VC + Aug — Evaluation"
echo "================================================"
python scripts/run_eval.py \
    --checkpoint results_tonsill_aug/best_model.pth

echo "Done!"
