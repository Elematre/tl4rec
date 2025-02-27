#!/bin/bash
#SBATCH --mail-type=NONE                           # Mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.out   # Output file (A = JobID, a = ArrayIndex)
#SBATCH --error=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.err    # Error file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:geforce_rtx_3090:1                # Allocate 1 GPU per job
#SBATCH --nodelist=tikgpu09                          # Specific node reservation
#SBATCH --array=0-19                                 # Total jobs = (10 datasets * 2 checkpoints)

# User-specific variables
ETH_USERNAME=trachsele
PROJECT_NAME=tl4rec
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=ba_bugfix

# Set the checkpoint directory full path
CKPT_DIR=${DIRECTORY}/ckpts/pretrain

# Define an array of datasets
DATASETS=("Epinions" "BookX" "Ml1m" "Gowalla" "Amazon_Beauty" "Amazon_Fashion" "Amazon_Men" "Amazon_Games" "Yelp18" "LastFM")

# Define an array of checkpoint file names (without path)
CKPTS=("Yelp18.pth" \
       "Gowalla.pth")

# Compute dataset and checkpoint index from the SLURM_ARRAY_TASK_ID
job_index=$SLURM_ARRAY_TASK_ID
num_ckpts=${#CKPTS[@]}
num_datasets=${#DATASETS[@]}

dataset_index=$(( job_index / num_ckpts ))
ckpt_index=$(( job_index % num_ckpts ))

# Ensure valid indices
if [ $dataset_index -ge $num_datasets ] || [ $ckpt_index -ge $num_ckpts ]; then
    echo "Invalid job index: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

DATASET=${DATASETS[$dataset_index]}
CKPT=${CKPTS[$ckpt_index]}

# Set the checkpoint full path
CKPT="${CKPT_DIR}/${CKPT}"

# Fixed settings for zero-shot evaluation
EPOCH=0
MODE_NAME="zero-shot"

# Create jobs directory if it doesn't exist
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

# Change to temporary directory
cd "${TMPDIR}" || exit 1

# Log system info
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Evaluating dataset: $DATASET"
echo "Evaluating checkpoint: $CKPT in $MODE_NAME mode (Epoch: $EPOCH)"

# Load Conda
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && \
    eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
conda info --envs

# Change to project directory
cd ${DIRECTORY}

# Run evaluation (zero-shot mode only)
python script/run.py -c config/recommender/slurm_cfg.yaml \
    --dataset "$DATASET" --epochs $EPOCH --bpe 1000 --gpus "[0]" --ckpt "$CKPT"

# Log completion time
echo "Finished evaluation for dataset: $DATASET with checkpoint: $CKPT in $MODE_NAME mode"
echo "Finished at: $(date)"

exit 0
