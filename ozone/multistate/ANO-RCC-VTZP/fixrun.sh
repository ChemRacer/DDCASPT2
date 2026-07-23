#!/bin/bash
#
for i in ./ozo*; do
    cd "$i" || exit
    rad=$(basename "$i")  # Extract directory name

    # Modify run.sh to update SBATCH settings
    sed -i -E "s|^#SBATCH --account=.*|#SBATCH --account=ACCOUNTNAME|;
               s|^#SBATCH --job-name=.*|#SBATCH --job-name=$rad|;
               s|^#SBATCH --partition=.*|#SBATCH --partition=ACCOUNTNAME|;
               s|^#SBATCH --qos=.*|#SBATCH --qos=ACCOUNTNAME|" run.sh

    cd ../
done
