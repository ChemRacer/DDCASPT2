#!/bin/bash

dir=$(find . -mindepth 1 -maxdepth 1 -type d | sort -h) # Collect all directories and sort them
subdir=$(basename "$(pwd)") # Get the base name of the current directory

jobid=""

for i in $dir; do
    cd "$i" || exit # Change to the directory or exit on failure
    dir_name=$(basename "$i") # Get the directory name without the `./`
    
    # Skip directories with names less than 1.42
    if (( $(echo "$dir_name < 2.96" | bc -l) )); then
        cd ../
        continue
    fi

    if [[ -z $jobid ]]; then
        # Submit the first job if no jobid exists yet
        echo "Submitting first job for point $dir_name"
        jobid=$(sbatch --parsable run.sh)
        echo "Submitted job for point $dir_name with Job ID: $jobid"
    else
        # Submit the next job with a dependency on the previous one
        echo "Submitting job for point $dir_name with dependency on Job ID: $jobid"
        jobid=$(sbatch --dependency=afterok:$jobid --parsable run.sh)
    fi
    cd ../ || exit # Return to the parent directory or exit on failure
done

