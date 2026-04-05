#!/bin/bash
#SBATCH --job-name=snow_poles_eval
#SBATCH --output=runs/slurm/%j.out
#SBATCH --error=runs/slurm/%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=ie-idi   # Update with your IDUN account

module purge
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# UPDATE this path to your best trained model
BEST_MODEL="runs/train/snow_poles/weights/best.pt"

python src/evaluate.py \
    --model "$BEST_MODEL" \
    --config configs/road_poles_iphone.yaml \
    --split val \
    --device 0
