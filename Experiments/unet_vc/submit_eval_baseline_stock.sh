#!/bin/bash
#SBATCH --job-name=unet_vc_baseline_eval
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/unet_vc

echo "================================================"
echo "  UNet-VC baseline (no aug) — Eval with stock HiFi-GAN"
echo "================================================"
python scripts/run_eval.py \
    --checkpoint checkpoints_kfold_baseline/best_model.pt \
    --stock_vocoder

echo "Done!"
