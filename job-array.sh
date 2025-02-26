#!/bin/bash
#SBATCH --mail-type=NONE                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%j_%a.out   # Output file, where %j is JOBID and %a is the array index
#SBATCH --error=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%j_%a.err    # Error file, where %j is JOBID and %a is the array index
#SBATCH --mem=20G
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --nodelist=tikgpu09
#SBATCH --array=0-9                          # Array job: indices 0 to 9 for 10 datasets

ETH_USERNAME=trachsele
PROJECT_NAME=tl4rec
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=ba_bugfix

# Create jobs directory if it doesn't exist
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Set up a temporary directory for the job and clean up automatically
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change to the temporary directory
cd "${TMPDIR}" || exit 1

# Log some useful information
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

# Activate conda environment
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
conda info --envs

# Change back to the main directory
cd ${DIRECTORY}

# Define datasets array
DATASETS=("Epinions" "LastFM" "BookX" "Ml1m" "Gowalla" "Amazon_Beauty" "Amazon_Fashion" "Amazon_Men" "Amazon_Games" "Yelp18")

# Select the dataset based on the SLURM array index
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "Processing dataset: ${DATASET}"

# Run training and evaluation with epochs = 9 and bpe = 0
python script/run.py -c config/recommender/slurm_cfg.yaml --dataset ${DATASET} --epochs 8 --bpe 0 --gpus "[0]" --ckpt null

# Log the finish time
echo "Finished at: $(date)"

exit 0
