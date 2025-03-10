#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%j.out # Output file, where %j is the JOBID
#SBATCH --error=/itet-stor/trachsele/net_scratch/tl4rec/jobs/%j.err # Error file, where %j is the JOBID1~#SBATCH --mem=20G
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#CommentSBATCH --gres=gpu:titan_rtx:1 
#SBATCH --gres=gpu:geforce_rtx_3090:1 
#SBATCH --nodelist=tikgpu09
#CommentSBATCH --account=tik-internal


ETH_USERNAME=trachsele
PROJECT_NAME=tl4rec
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=ba_bugfix
mkdir -p ${DIRECTORY}/jobs
#TODO: change your ETH USERNAME and other stuff from above according + in the #SBATCH output and error the path needs to be double checked!

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
conda info --envs

cd ${DIRECTORY}
#DATASETS = [
   # "Epinions", "LastFM", "BookX", "Ml1m", "Gowalla",
   # "Amazon_Beauty", "Amazon_Fashion", "Amazon_Men", "Amazon_Games", "Yelp18"
#]
# Execute your code 62 bpe for light gcn and 1978 for simple on movielense
python script/run.py -c config/recommender/slurm_cfg.yaml --dataset Ml1m --epochs 1 --bpe 0 --gpus "[0]" --ckpt /itet-stor/trachsele/net_scratch/tl4rec/ckpts/pretrain/inionsBeautyMl1m.pth
#python script/hyperparam_search.py -c config/recommender/pretrain_all.yaml --gpus [0]
#python script/run.py -c config/recommender/slurm_cfg.yaml --dataset LastFM --epochs 8 --bpe 2000 --gpus "[0]" --ckpt null
#python script/pretrain.py -c config/recommender/pretrain_all.yaml --gpus [0]
#python script/run.py -c config/recommender/slurm_cfg.yaml --dataset Amazon_Fashion --epochs 1 --bpe 66909 --gpus "[0]" --ckpt null --seed 17

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
