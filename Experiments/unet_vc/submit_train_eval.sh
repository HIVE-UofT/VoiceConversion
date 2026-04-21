#!/bin/bash
#SBATCH --job-name=unet_vc_train
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/unet_vc

echo "=============================="
echo "  UNet-VC — 5-Fold CV + Train"
echo "=============================="
python scripts/train_kfold.py \
    --output checkpoints_kfold \
    --k_folds 5 \
    --seed 42

echo ""
echo "=============================="
echo "  UNet-VC — Evaluation"
echo "=============================="
python scripts/run_eval.py \
    --checkpoint checkpoints_kfold/best_model.pt

echo "Done!"
