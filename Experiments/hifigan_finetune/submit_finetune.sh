#!/bin/bash
#SBATCH --job-name=hifigan_finetune
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

python finetune.py \
    --data_dir /home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios \
    --out_dir ./output \
    --steps 6000 \
    --lr 1e-4 \
    --batch_size 8 \
    --val_interval 500 \
    --seed 42

echo "Done!"
