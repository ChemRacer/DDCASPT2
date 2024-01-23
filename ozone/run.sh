#!/bin/bash
#This file is a submission script to request the ISAAC resources from Slurm 
#SBATCH --account=ACF-UTK0022             # The project account to be charged
#SBATCH --job-name=O3		       #The name of the job
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=16          # cpus per node 
#SBATCH --partition=condo-kvogiatz            # If not specified then default is "condo"
#SBATCH --qos=condo
#SBATCH --time=2-00:00:00             # Wall time (days-hh:mm:ss)
#SBATCH --error=job.e%J	       # The file where run time errors will be dumped
#SBATCH --output=job.o%J	       # The file where the output of the terminal will be dumped



# PATCH!
# MOVE TO WORKING DIR
echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
echo SKATA
pwd
# DEFINE WorkDir
mkdir $SLURM_SUBMIT_DIR/$SLURM_JOBID
export WorkDir=$SLURM_SUBMIT_DIR/$SLURM_JOBID
export MOLCAS="/lustre/isaac/proj/UTK0022/GMJ/OpenMolcas/build"
echo $WorkDir
module load intel-compilers 
# DEFINE WorkDir
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

# THE COMMAND
#python3 Data_Script.py 
conda activate /lustre/isaac/proj/UTK0022/GMJ/modules/anaconda3/envs/ddcaspt2
python3 DDCASPT2.py 
# CLEAN-UP AND EXIT
rm -r $SLURM_SUBMIT_DIR/$SLURM_JOBID
echo "Program finished with exit code $? at: `date`"
