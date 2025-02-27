#!/bin/bash
#SBATCH --mail-type=NONE                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.out   # %A = job array ID, %a = task ID
#SBATCH --error=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%A_%a.err    # %A = job array ID, %a = task ID
#SBATCH --mem=20G
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:geforce_rtx_3090:1 
#SBATCH --nodelist=tikgpu09
#SBATCH --array=0-15                                # 16 jobs (4 datasets x 4 checkpoints)

ETH_USERNAME=trachsele
PROJECT_NAME=tl4rec
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=ba_bugfix
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Create a temporary directory that will be automatically removed at job termination
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

# Log key information
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

# Activate the conda environment
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && \
    eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
conda info --envs

# Change to the project directory
cd ${DIRECTORY}

# Define arrays for datasets and corresponding checkpoints
DATASETS=("Ml1m" "LastFM" "Epinions" "BookX")
CKPTS=("Ml1m.pth" "LastFM.pth" "Epinions.pth" "BookX.pth")

# Compute indices from the SLURM_ARRAY_TASK_ID (4 datasets x 4 checkpoints = 16 jobs)
job_index=$SLURM_ARRAY_TASK_ID
num_ckpts=${#CKPTS[@]}
num_datasets=${#DATASETS[@]}

dataset_index=$(( job_index / num_ckpts ))
ckpt_index=$(( job_index % num_ckpts ))

if [ $dataset_index -ge $num_datasets ]; then
    echo "Invalid job index: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

DATASET=${DATASETS[$dataset_index]}
CKPT_FILE=${CKPTS[$ckpt_index]}
# Construct the full checkpoint path (same folder as before)
CKPT="${DIRECTORY}/ckpts/pretrain/${CKPT_FILE}"

echo "Evaluating dataset: ${DATASET}"
echo "Using checkpoint: ${CKPT}"

# Run the evaluation (0-shot evaluation: epochs=0)
python script/run.py -c config/recommender/slurm_cfg.yaml \
    --dataset "${DATASET}" --epochs 0 --bpe 1000 --gpus "[0]" --ckpt "${CKPT}"

echo "Finished evaluation for dataset: ${DATASET} with checkpoint: ${CKPT} at: $(date)"
exit 0
