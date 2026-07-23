#!/bin/bash
#
for i in ./ozo*; do
    cd "$i" || exit
    rad=$(basename "$i")  # Extract directory name

    # Modify run.sh to update SBATCH settings
    sed -i -E "s|^#SBATCH --account=.*|#SBATCH --account=ACF-UTK0022|;
               s|^#SBATCH --job-name=.*|#SBATCH --job-name=$rad|;
               s|^#SBATCH --partition=.*|#SBATCH --partition=condo-kvogiatz|;
               s|^#SBATCH --qos=.*|#SBATCH --qos=condo-kvogiatz|" run.sh

    cd ../
done
