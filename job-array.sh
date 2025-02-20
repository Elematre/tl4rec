#!/bin/bash
#SBATCH --mail-type=NONE  # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.out  # Output file (A = JobID, a = ArrayIndex)
#SBATCH --error=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.err  # Error file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:geforce_rtx_3090:1  # Allocate 1 GPU per job
#SBATCH --nodelist=tikgpu09  # Specific node reservation
#SBATCH --array=0-9%8  # Job array: 10 datasets (0-9), max 8 parallel jobs

# User-specific variables
ETH_USERNAME=trachsele
PROJECT_NAME=tl4rec
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=ba_bugfix

# Dataset list (ensure order matches job array indices)
DATASETS=("Yelp18" "Gowalla" "Ml1m" "Amazon_Beauty" "Epinions" "LastFM" "BookX" "Amazon_Fashion" "Amazon_Men" "Amazon_Games")

# Get dataset for this job using SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Exit if dataset is not set correctly
if [[ -z "$DATASET" ]]; then
    echo "Invalid dataset index: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Create jobs directory if not exists
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Temporary directory setup
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo "Failed to create temp directory" >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change to temp directory
cd "${TMPDIR}" || exit 1

# Log system info
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Processing dataset: $DATASET"

# Load Conda
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
conda info --envs

# Change to project directory
cd ${DIRECTORY}

# Train on the selected dataset
python script/run.py -c config/recommender/slurm_cfg.yaml --dataset "$DATASET" --epochs 8 --bpe 1000 --gpus "[0]" --ckpt null

# Log completion time
echo "Finished processing dataset: $DATASET"
echo "Finished at: $(date)"

exit 0
