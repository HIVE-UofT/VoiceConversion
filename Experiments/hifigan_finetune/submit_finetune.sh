#!/bin/bash
#SBATCH --job-name=hifigan_finetune
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

module load python/3.10

source ../../.env/bin/activate

python finetune.py \
    --data_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios \
    --out_dir ./output \
    --steps 10000 \
    --lr 1e-4 \
    --batch_size 8 \
    --val_interval 500 \
    --seed 42

echo "Done!"
