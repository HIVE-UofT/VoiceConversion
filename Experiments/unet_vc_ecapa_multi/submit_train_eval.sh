#!/bin/bash
#SBATCH --job-name=unet_ecapa_multi_train
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

export FORCE_STOCK_VOCODER=1

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/unet_vc_ecapa_multi

echo "=============================="
echo "  UNet-VC-ECAPA Multi — Training"
echo "=============================="
python scripts/train_split_v2.py \
    --surgeries Tonsill,Fess,Sept \
    --output results_multi_v2 \
    --seed 42

echo ""
echo "=============================="
echo "  UNet-VC-ECAPA Multi — Evaluation"
echo "=============================="
python scripts/run_eval.py \
    --checkpoint results_multi_v2/best_model.pt

echo "Done!"
