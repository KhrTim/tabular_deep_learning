#!/bin/bash

# Replace these with your actual Google Drive file IDs
FILE_IDS=("161WhZ0A5_AVr_KnZ5hZjLbbuqsLk4INx" "1xjcc7nzcrRBb7CodS5GswMhz_rUhAVzU")
FILE_NAMES=("data.zip" "mlruns.zip")

for i in ${!FILE_IDS[@]}; do
    FILE_ID="${FILE_IDS[$i]}"
    FILE_NAME="${FILE_NAMES[$i]}"

    echo "Downloading $FILE_NAME..."
    gdown --id "$FILE_ID" -O "$FILE_NAME"

    echo "Extracting $FILE_NAME..."
    unzip "$FILE_NAME"

    echo "Removing $FILE_NAME..."
    rm "$FILE_NAME"

    echo "Done with $FILE_NAME"
    echo "-----------------------"
done

