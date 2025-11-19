#!/bin/bash

# Usage: ./run_batch_parallel.sh <CSV_DIR> <SOURCE_VIDEO_DIR> [MAX_JOBS]
# Example: ./run_batch_parallel.sh ./annotations ./raw_videos 4

CSV_DIR="/media/oeste/BeaGL2/CMORE/Sequence_Classifier_GT"
SOURCE_VIDEO_DIR="/home/oeste/Desktop/Jun/KingLabs/WORKINGDATA/AdditionalSubjects_05_23_2025"
MAX_JOBS=8 # Default to 4 parallel jobs if not specified

OUTPUT_DIR="/media/oeste/BeaGL2/CMORE/Sequence_Classifier_Data"
MODEL_PATH="keypoint_detector.pt"

# 1. Validate Input Arguments
if [ -z "$CSV_DIR" ] || [ -z "$SOURCE_VIDEO_DIR" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <CSV_DIR> <SOURCE_VIDEO_DIR> [MAX_JOBS]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Export variables so the sub-processes (spawned by xargs) can see them
export SOURCE_VIDEO_DIR
export OUTPUT_DIR
export MODEL_PATH

echo "Starting PARALLEL processing..."
echo "Job Limit: $MAX_JOBS concurrent processes"
echo "------------------------------------------------"

# 2. Define the Worker Function
# This contains the logic that used to be inside the loop.
# We function-ize it so we can pass it to xargs.
process_single_csv() {
    csv_path="$1"
    
    # Ensure file exists
    [ -e "$csv_path" ] || return

    filename=$(basename -- "$csv_path")
    
    # Suffix Removal (Standard spelling)
    video_name="${filename%_sequence_annotations.csv}"

    # Search Recursively for the Video
    # (We use head -n 1 to just grab the first match)
    video_path=$(find "$SOURCE_VIDEO_DIR" -type f -name "${video_name}.*" | head -n 1)

    if [ -z "$video_path" ]; then
        echo "[!] SKIPPING: Video not found for '$video_name'"
        return
    fi

    # Determine Handedness
    if [[ "$video_name" == *"Left"* ]] || [[ "$video_name" == *"left"* ]]; then
        HAND_FLAG="--L"
    elif [[ "$video_name" == *"Right"* ]] || [[ "$video_name" == *"right"* ]]; then
        HAND_FLAG="--R"
    else
        # Default to Right
        HAND_FLAG="--R"
    fi

    echo "[STARTED] $video_name ($HAND_FLAG)"

    # Run Python script
    # We capture stdout/stderr to a log file so output doesn't get jumbled on screen,
    # or you can let it print to screen (might get messy). 
    # Here we let it print but prepend the video name for clarity is hard in bash parallel.
    # We will just run it.
    python3 generate_sequences.py \
        --video "$video_path" \
        --csv "$csv_path" \
        --out_path "$OUTPUT_DIR" \
        --model "$MODEL_PATH" \
        $HAND_FLAG > /dev/null 2>&1

    # Note: > /dev/null 2>&1 hides the python output to keep the terminal clean.
    # If you want to see errors, remove "> /dev/null 2>&1"

    echo "[COMPLETE] $video_name"
}

# 3. Export the function so 'bash -c' can use it
export -f process_single_csv

# 4. Find files and Pipe to xargs
# -print0 / -0 handles filenames with spaces correctly
# -P $MAX_JOBS sets the number of parallel CPUs to use
# -I {} replaces {} with the filename
# bash -c runs our exported function
find "$CSV_DIR" -name "*_sequence_annotations.csv" -print0 | \
    xargs -0 -P "$MAX_JOBS" -I {} bash -c 'process_single_csv "$@"' _ "{}"

echo "------------------------------------------------"
echo "Parallel batch processing complete."