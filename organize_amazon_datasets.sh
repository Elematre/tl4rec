#!/bin/bash

# Base source directory
SOURCE_DIR="/itet-stor/trachsele/net_scratch/tl4rec/CARCA_Code_and_Data/Data"

# Base destination directory
DEST_DIR="/itet-stor/trachsele/net_scratch/tl4rec/model_outputs/data"

# Array of dataset names and their corresponding file patterns
declare -A DATASETS=(
    ["amazon_fashion"]="Fashion"
    ["amazon_men"]="Men"
    ["amazon_games"]="Video_Games"
)

# Iterate through datasets and organize files
for dataset in "${!DATASETS[@]}"; do
    # Define file prefix
    PREFIX=${DATASETS[$dataset]}
    
    # Create necessary directories
    mkdir -p "${DEST_DIR}/${dataset}/raw"
    mkdir -p "${DEST_DIR}/${dataset}/processed"
    
    # Move and rename files
    cp "${SOURCE_DIR}/${PREFIX}.txt" "${DEST_DIR}/${dataset}/raw/${dataset}.txt"
    
    # Handle feature files (image or category features)
    if [ -f "${SOURCE_DIR}/${PREFIX}_imgs.dat" ]; then
        cp "${SOURCE_DIR}/${PREFIX}_imgs.dat" "${DEST_DIR}/${dataset}/raw/${dataset}_feat_cat.dat"
    elif [ -f "${SOURCE_DIR}/${PREFIX}_feat.dat" ]; then
        cp "${SOURCE_DIR}/${PREFIX}_feat.dat" "${DEST_DIR}/${dataset}/raw/${dataset}_feat_cat.dat"
    fi

    # Handle context dictionary file
    cp "${SOURCE_DIR}/CXTDictSasRec_${PREFIX}.dat" "${DEST_DIR}/${dataset}/raw/${dataset}_ctxt.dat"
    
    echo "Organized ${dataset} dataset."
done

echo "All datasets have been organized."
