#!/bin/bash
#SBATCH --job-name=unet_ecapa_train
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/unet_vc_ecapa

echo "=============================="
echo "  UNet-VC-ECAPA — Training"
echo "=============================="
python scripts/train_split_v2.py \
    --surgery Tonsill \
    --output results_tonsill_v2 \
    --seed 42

echo ""
echo "=============================="
echo "  UNet-VC-ECAPA — Evaluation"
echo "=============================="
python scripts/run_eval.py \
    --checkpoint results_tonsill_v2/best_model.pt

echo "Done!"
