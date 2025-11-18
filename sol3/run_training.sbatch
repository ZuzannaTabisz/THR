#!/bin/bash
#SBATCH --job-name=sol3_training
#SBATCH --partition=hgx
#SBATCH --account=mgr_mikro
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G 
#SBATCH --time=1-00:00:00 # 1 day
#SBATCH --output=logs/sol3_training_output_%j.out
#SBATCH --error=logs/sol3_training_error_%j.err
 
echo "=== STARTING JOB: $(date) ==="
# Exit immediately if a command exits with a non-zero status.
# This is crucial for catching errors early.
set -e

echo "Job is running on node: $(hostname)"

# Create a logs directory if it doesn't exist. This must be after the #SBATCH block.
mkdir -p logs

# --- STAGE 0: INITIALIZE CONDA ---
# This is crucial for non-interactive batch jobs to find your conda environments.
source /home/inf151561/miniconda3/etc/profile.d/conda.sh
echo "Conda initialized."

# --- STAGE 1: DATA PREPARATION ---
# Define a permanent location for preprocessed tiles and a temporary one on fast /raid storage.
PERMANENT_TILE_DIR="/home/$USER/sol3/train_tiles_permanent"
RAID_TILE_DIR="/raid/users/$USER/sol3_job_$SLURM_JOB_ID/train_tiles"

echo "Permanent tile storage: $PERMANENT_TILE_DIR"
echo "Temporary /raid tile storage for this job: $RAID_TILE_DIR"

# Activate the conda environment. All subsequent commands will run inside it.
conda activate mayo_env

 # --- Sanity Check: Verify GPU access ---
 echo "Verifying PyTorch can access the GPU..."
 python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA. Check environment or job submission.' ; print('CUDA is available. GPU Name:', torch.cuda.get_device_name(0))"

 # --- Sanity Check: Clean Python cache ---
 echo "Cleaning Python __pycache__ directories..."
 find . -type d -name "__pycache__" -exec rm -r {} +
 echo "Verification successful."

 # Preprocess images only if the permanent directory doesn't exist or is empty.
 if [ -d "$PERMANENT_TILE_DIR" ] && [ "$(ls -A $PERMANENT_TILE_DIR)" ]; then
    echo "Permanent tile directory found. Skipping preprocessing."
 else
    echo "Permanent tile directory not found or empty. Running preprocessing..."
    mkdir -p "$PERMANENT_TILE_DIR"
    python preprocess.py --image_dir ../data/train --csv_path ../data/train.csv --output_dir "$PERMANENT_TILE_DIR"
 fi
 
 # Copy the permanent tiles to the fast /raid disk for this training run.
 echo "Copying tiles to /raid for fast access during training..."
 mkdir -p "$(dirname "$RAID_TILE_DIR")"
 cp -r "$PERMANENT_TILE_DIR" "$RAID_TILE_DIR"
 
 # --- STAGE 2: TRAINING ---
 # This step fine-tunes the model on the generated tiles using a GPU.
 # The training script needs to know where the tiles are located.
 echo "Running training script, reading from $RAID_TILE_DIR..."
 python train.py --tile_dir "$RAID_TILE_DIR"
 
 echo "=== JOB FINISHED: $(date) ==="

# Clean up the temporary data on the /raid disk
echo "Cleaning up temporary directory: $(dirname "$RAID_TILE_DIR")"
rm -rf "$(dirname "$RAID_TILE_DIR")"
