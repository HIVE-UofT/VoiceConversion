#!/bin/bash
#SBATCH --job-name=unet_spk_train
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/unet_vc_spk

echo "=============================="
echo "  UNet-VC-SPK — Training"
echo "=============================="
python scripts/train_split.py \
    --surgery Tonsill \
    --output results_tonsill \
    --seed 42

echo ""
echo "=============================="
echo "  UNet-VC-SPK — Evaluation"
echo "=============================="
python scripts/run_eval.py \
    --checkpoint results_tonsill/best_model.pt

echo "Done!"
