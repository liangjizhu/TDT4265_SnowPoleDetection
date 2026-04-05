#!/bin/bash
#SBATCH --job-name=snow_poles
#SBATCH --output=runs/slurm/%j.out
#SBATCH --error=runs/slurm/%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=ie-idi   # Update with your IDUN account

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module purge
module load Python/3.10.8-GCCcore-12.2.0

# Create and activate virtual environment (first time only)
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

pip install --quiet -r requirements.txt

# Symlink dataset if not already done
if [ ! -L "data/Poles2025" ] && [ ! -d "data/Poles2025" ]; then
    ln -s /cluster/projects/vc/courses/TDT17/ad/Poles2025 data/Poles2025
fi

mkdir -p runs/slurm

python src/train.py \
    --config configs/road_poles_iphone.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --name "snow_poles_$(date +%Y%m%d_%H%M%S)"

echo "Job finished at $(date)"
