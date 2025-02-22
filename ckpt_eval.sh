#!/bin/bash
#SBATCH --mail-type=NONE                           # Mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.out   # Output file (A = JobID, a = ArrayIndex)
#SBATCH --error=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.err    # Error file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:geforce_rtx_3090:1                # Allocate 1 GPU per job
#SBATCH --nodelist=tikgpu09                          # Specific node reservation
#SBATCH --array=0-23                                # Two evaluations per ckpt (if 12 ckpts, 0–23)

# User-specific variables
ETH_USERNAME=trachsele
PROJECT_NAME=tl4rec
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=ba_bugfix

# Set the checkpoint directory full path (update if necessary)
CKPT_DIR=${DIRECTORY}/ckpts/pretrain

# Define an array of checkpoint file names (without path)
CKPTS=("Amazon_Beauty.pth" \
           "Amazon_Games.pth" \
           "Beauty_Games_tuned.pth" \
           "Men_Epin_Gowa_Book.pth" \
           "Amazon_Fashion.pth" \
           "Epinions.pth" \
           "Ml1m.pth" \
           "Gowalla.pth" \
           "BookX.pth" \
           "LastFM.pth" \
           "inionsBeautyMl1m.pth" \
           "Yelp18.pth")

             

# Prepend the full path for each checkpoint file
for i in "${!CKPTS[@]}"; do
    CKPTS[$i]="${CKPT_DIR}/${CKPTS[$i]}"
done

# Total number of checkpoint files
NUM_CKPTS=${#CKPTS[@]}

# Compute the mode and checkpoint index from the SLURM_ARRAY_TASK_ID
# Mode: 0 for zero-shot (epoch=0), 1 for fine-tuned (epoch=1)
job_index=$SLURM_ARRAY_TASK_ID
mode=$(( job_index % 2 ))
ckpt_index=$(( job_index / 2 ))

if [ $ckpt_index -ge $NUM_CKPTS ]; then
    echo "Invalid job index: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

CKPT=${CKPTS[$ckpt_index]}

# Set the epoch based on the mode
if [ $mode -eq 0 ]; then
    EPOCH=0
    MODE_NAME="zero-shot"
else
    EPOCH=1
    MODE_NAME="fine-tuned"
fi

DATASET="Amazon_Men"  # Fixed dataset

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

# Run evaluation
python script/run.py -c config/recommender/slurm_cfg.yaml \
    --dataset "$DATASET" --epochs $EPOCH --bpe 1000 --gpus "[0]" --ckpt "$CKPT"

# Log completion time
echo "Finished evaluation for checkpoint: $CKPT in $MODE_NAME mode"
echo "Finished at: $(date)"

exit 0
