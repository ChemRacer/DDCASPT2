#!/bin/bash
dir=$(find . -mindepth 1 -maxdepth 1 -type d | sort -h)

subdir=$(basename $(pwd))
for i in $dir; do
     cd $i
     if [[ "$i" == "./1.02" ]]; then
          # Do something if $i is equal to "./0.1"
          echo "Matched $i"
          jobid=$(sbatch --parsable run.sh)
          #jobid=$(squeue -u $USER | grep "$subdir" | awk '{print $1}')
          echo "Submitted job for point $i with Job ID: $jobid"
     elif [[ "$i" != "./1.02" ]]; then
     # Submit the next job with a dependency on the previous one
          echo "Submitted job for point $i with dependency on Job ID: $jobid"
          jobid=$(sbatch --dependency=afterok:$jobid --parsable run.sh)
     fi
     cd ../
done
