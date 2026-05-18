#!/bin/bash
#SBATCH --job-name=unet_vc_aug
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/unet_vc

echo "================================================"
echo "  UNet-VC + Audio Augmentation — 5-Fold CV + Train"
echo "  Augmentation: pitch shift (+-2 st), time stretch (0.92-1.08),"
echo "  gain (+-3 dB), Gaussian noise. 3 augmentation rounds."
echo "  Test patients (0045,0085,0110,0122,0132) are NOT augmented."
echo "================================================"

python scripts/train_kfold.py \
    --output checkpoints_kfold_aug \
    --k_folds 5 \
    --seed 42 \
    --n_aug 3

echo ""
echo "================================================"
echo "  UNet-VC (aug) — Evaluation on held-out test patients"
echo "================================================"
python scripts/run_eval.py \
    --checkpoint checkpoints_kfold_aug/best_model.pt

echo "Done!"
