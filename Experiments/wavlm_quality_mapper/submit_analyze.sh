#!/bin/bash
#SBATCH --job-name=wlm_analyze
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10
source /project/6086959/sepehr/VoiceConversion/.env/bin/activate

echo "=========================================="
echo "  WavLM All 24 Layers Analysis: Pre vs Post"
echo "=========================================="

python scripts/analyze_layers.py --surgery Tonsill

echo "Done!"
