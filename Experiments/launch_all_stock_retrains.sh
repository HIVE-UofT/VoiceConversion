#!/bin/bash
# Submit every method's retrain + eval with FORCE_STOCK_VOCODER=1 set so all
# methods use the same stock bshall/knn-vc HiFi-GAN. This produces a clean
# comparison table across the 5 held-out test patients.

set -e
EXP=/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments

echo "=== Submitting training-free eval bundle (kNN-VC, Mean-Shift, MKL-VC, LinearVC) ==="
sbatch "$EXP/submit_all_evals_stock.sh"

echo ""
echo "=== UNet-Adv-VC retrain ==="
sbatch --export=ALL,FORCE_STOCK_VOCODER=1 \
       --output="$EXP/unet_adv_vc/%x-%j.out" \
       --chdir="$EXP/unet_adv_vc" \
       "$EXP/unet_adv_vc/submit_train_eval.sh"

echo ""
echo "=== UNet-VC-ECAPA retrain ==="
sbatch --export=ALL,FORCE_STOCK_VOCODER=1 \
       --output="$EXP/unet_vc_ecapa/%x-%j.out" \
       --chdir="$EXP/unet_vc_ecapa" \
       "$EXP/unet_vc_ecapa/submit_train_eval.sh"

echo ""
echo "=== UNet-VC-SPK retrain ==="
sbatch --export=ALL,FORCE_STOCK_VOCODER=1 \
       --output="$EXP/unet_vc_spk/%x-%j.out" \
       --chdir="$EXP/unet_vc_spk" \
       "$EXP/unet_vc_spk/submit_train_eval.sh"

echo ""
echo "All jobs submitted. Monitor with: squeue -u sepharfi"
