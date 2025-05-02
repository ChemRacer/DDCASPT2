#!/bin/bash

# Define the starting directory
start_dir="./ozone_168.50"
# Flag to indicate when to start submitting jobs
start_found=false

# Variable to hold the previous job ID for dependency
prev_jobid=""

# Loop over directories sorted numerically
find . -mindepth 1 -maxdepth 1 -type d | sort -V | while read -r dir; do
    # Check if we've reached the starting directory
    if [[ "$dir" == "$start_dir" ]]; then
        start_found=true
    fi

    # Skip directories until the starting point is found
    if ! $start_found; then
        continue
    fi

    echo "Processing directory: $dir"
    cd "$dir" || { echo "Failed to enter directory $dir"; exit 1; }

    # Submit the job
    if [[ -z "$prev_jobid" ]]; then
        # Submit the first job without dependency
        jobid=$(sbatch --parsable run.sh)
        echo "Submitted initial job for $dir with Job ID: $jobid"
    else
        # Submit subsequent jobs with dependency on the previous job
        jobid=$(sbatch --dependency=afterok:$prev_jobid --parsable run.sh)
        echo "Submitted job for $dir with dependency on Job ID: $prev_jobid (New Job ID: $jobid)"
    fi

    # Update the previous job ID
    prev_jobid=$jobid

    # Return to the parent directory
    cd - > /dev/null || exit 1
done

