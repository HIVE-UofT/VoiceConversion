#!/bin/bash
#SBATCH --job-name=unet_adv_train
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/unet_adv_vc

echo "=============================="
echo "  UNet-Adv-VC — Training"
echo "=============================="
python scripts/train_split.py \
    --output checkpoints \
    --seed 42

echo ""
echo "=============================="
echo "  UNet-Adv-VC — Evaluation"
echo "=============================="
python scripts/run_eval.py \
    --checkpoint checkpoints/best_model.pt

echo "Done!"
