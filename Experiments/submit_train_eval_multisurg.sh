#!/bin/bash
#SBATCH --job-name=multisurg_train
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

# Multi-surgery training experiment:
# Each model trains on Tonsill (23 patients, all audio types) PLUS
# all patients from Fess + Sept + Contr (all audio types).
# Validation and testing remain Tonsill-only.
# Goal: see if adding ~115 extra patients from other surgery types helps.

EXPERIMENTS=/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments

echo "=============================="
echo "  UNet-VC (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/unet_vc
python scripts/train_kfold.py \
    --extra_surgeries \
    --k_folds 5 \
    --seed 42

echo ""
echo "  UNet-VC (MultiSurg) — Evaluation"
python scripts/run_eval.py \
    --checkpoint checkpoints_kfold_multisurg/best_model.pt

echo "=============================="
echo "  UNet-Adv-VC (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/unet_adv_vc
python scripts/train_split.py \
    --extra_surgeries \
    --seed 42

echo ""
echo "  UNet-Adv-VC (MultiSurg) — Evaluation"
python scripts/run_eval.py \
    --checkpoint checkpoints_multisurg/best_model.pt

echo "=============================="
echo "  UNet-VC-SPK (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/unet_vc_spk
python scripts/train_split.py \
    --extra_surgeries \
    --seed 42

echo ""
echo "  UNet-VC-SPK (MultiSurg) — Evaluation"
python scripts/run_eval.py \
    --checkpoint results_tonsill_multisurg/best_model.pt

echo "=============================="
echo "  UNet-VC-ECAPA (MultiSurg)"
echo "=============================="
cd $EXPERIMENTS/unet_vc_ecapa
python scripts/train_split_v2.py \
    --extra_surgeries \
    --seed 42

echo ""
echo "  UNet-VC-ECAPA (MultiSurg) — Evaluation"
python scripts/run_eval.py \
    --checkpoint results_tonsill_v2_multisurg/best_model.pt

echo ""
echo "All multi-surgery training done!"
