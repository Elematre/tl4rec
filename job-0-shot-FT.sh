#!/bin/bash
#SBATCH --mail-type=NONE                           # Mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.out   # Output file (A = JobID, a = ArrayIndex)
#SBATCH --error=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.err    # Error file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:geforce_rtx_3090:1                # Allocate 1 GPU per job
#SBATCH --nodelist=tikgpu09                          # Specific node reservation
#SBATCH --array=0-3                                  # 4 datasets, each running in zero-shot mode

# User-specific variables
ETH_USERNAME=trachsele
PROJECT_NAME=tl4rec
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=ba_bugfix

# Fixed checkpoint for evaluation
CKPT="/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}/ckpts/pretrain/Beau_Epin.pth"

# Define an array of datasets for zero-shot evaluation
DATASETS=("Ml1m" "LastFM" "Epinions" "BookX")

# Determine dataset based on SLURM_ARRAY_TASK_ID
dataset_index=$SLURM_ARRAY_TASK_ID

if [ $dataset_index -ge ${#DATASETS[@]} ]; then
    echo "Invalid job index: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

DATASET=${DATASETS[$dataset_index]}

# Set evaluation mode to zero-shot (epochs=0)
EPOCH=0
MODE_NAME="zero-shot"

# Create jobs directory if it doesn't exist
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Setup a temporary directory that will be automatically removed at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo "Failed to create temp directory" >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change to the temporary directory
cd "${TMPDIR}" || exit 1

# Log system and job info
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Evaluating dataset: $DATASET"
echo "Using checkpoint: $CKPT"
echo "Mode: $MODE_NAME (Epoch: $EPOCH)"

# Load Conda and activate the environment
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && \
    eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
conda info --envs

# Change to the project directory
cd ${DIRECTORY}

# Run evaluation in zero-shot mode
python script/run.py -c config/recommender/slurm_cfg.yaml \
    --dataset "$DATASET" --epochs $EPOCH --bpe 1000 --gpus "[0]" --ckpt "$CKPT"

# Log completion
echo "Finished zero-shot evaluation for dataset: $DATASET with checkpoint: $CKPT"
echo "Finished at: $(date)"

exit 0
