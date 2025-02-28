#!/bin/bash
#This file is a submission script to request the ISAAC resources from Slurm
#SBATCH --account=ACF-UTK0022             # The project account to be charged
#SBATCH --job-name=ironoxo_2.73		       #The name of the job
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=16          # cpus per node
#SBATCH --partition=condo-kvogiatz            # If not specified then default is "campus"
#SBATCH --qos=condo-kvogiatz
#SBATCH --time=1-00:00:00             # Wall time (days-hh:mm:ss)
#SBATCH --error=job.e%J	       # The file where run time errors will be dumped
#SBATCH --output=job.o%J	       # The file where the output of the terminal will be dumped



# PATCH!
# MOVE TO WORKING DIR
echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
pwd
# DEFINE WorkDir
mkdir $SLURM_SUBMIT_DIR/$SLURM_JOBID
export WorkDir=$SLURM_SUBMIT_DIR/$SLURM_JOBID
export MOLCAS_WORKDIR="/lustre/isaac/scratch/gjones39"
export MOLCAS="/lustre/isaac/proj/UTK0022/Grier2025/Test/build"
echo $WorkDir



module purge
module load intel-compilers/2021.2.0
module load hdf5/1.10.8-intel
module load cmake/3.23.2-intel

# DEFINE WorkDir
echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is $(pwd)"

# THE COMMAND
pymolcas --new --clean 2.73.input -oe 2.73.output
# CLEAN-UP AND EXIT
rm -r $SLURM_SUBMIT_DIR/$SLURM_JOBID
echo "Program finished with exit code $? at: $(date)"
