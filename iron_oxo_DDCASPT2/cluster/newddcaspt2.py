#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from DDCASPT2 import DDCASPT2
import pickle, os, shutil
from glob import glob
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from math import sin, cos, pi
from tqdm.notebook import tqdm
# Geom manipulate
from AaronTools.geometry import Geometry
from tqdm import tqdm
from time import perf_counter


# Take [Fe$^{IV}$(O)(H$_2$O)$_4$]$^{+2}$ from Phys. Chem. Chem. Phys., 2018,20, 28786-28795

# In[ ]:


radius_range=np.linspace(1,3,100)



train_ind,test_ind=radius_range[0::2],radius_range[1::2]
# train_test_split(radius_range, test_size=0.3, random_state=0)
print(len(train_ind),len(test_ind))
with open('train_ind.pickle', 'wb') as handle:
    pickle.dump(train_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_ind.pickle', 'wb') as handle:
    pickle.dump(test_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_ind.pickle', 'rb') as handle:
    test_ind = pickle.load(handle)

with open('train_ind.pickle', 'rb') as handle:
    train_ind = pickle.load(handle)
    
print(len(train_ind),len(test_ind))    


# In[ ]:


def gen_run(r,path):
    with open(os.path.join(path,f'run.sh'),'w') as g:
        g.write(f"""#!/bin/bash
#This file is a submission script to request the ISAAC resources from Slurm
#SBATCH --account=ACF-UTK0022             # The project account to be charged
#SBATCH --job-name=ironoxo_{r}		       #The name of the job
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
export MOLCAS="/lustre/isaac/proj/UTK0022/GMJ/OpenMolcas/build"
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

conda activate /lustre/isaac/proj/UTK0022/GMJ/modules/anaconda3/envs/ddcaspt2
pymolcas --new --clean {r}.input -oe {r}.output
# CLEAN-UP AND EXIT
rm -r $SLURM_SUBMIT_DIR/$SLURM_JOBID
echo "Program finished with exit code $? at: $(date)"
""")    


# In[ ]:


topdir = os.getcwd()



# In[ ]:


t0 = perf_counter()
geom = Geometry('structure.xyz')
oxo, iron = geom.atoms[0],geom.atoms[1]

for idxr, r in tqdm(enumerate(radius_range)):

# Create radius subdirectory
    radstr = f"{r:.2f}"
    rad_dir = os.path.join(os.getcwd(),radstr)
    if os.path.exists(rad_dir)==False:
        os.mkdir(rad_dir)
    
        
    # Write xyz
    geom.change_distance(iron,oxo,r)
    geom.write(os.path.join(rad_dir,radstr))
    if idxr==0:
        d = DDCASPT2(rad_dir,'ANO-RCC-VDZP',radstr,10,8,33,previous=None,symmetry=1,spin=4,UHF=True,charge=2,clean=False)()
    else:            
        
        previous=os.path.join(topdir,f"{radius_range[idxr-1]:.2f}",f"{radius_range[idxr-1]:.2f}.RasOrb")
        d = DDCASPT2(rad_dir,'ANO-RCC-VDZP',radstr,10,8,33,previous=previous,symmetry=1,spin=4,UHF=True,charge=2,clean=False)()
    
    gen_run(radstr,rad_dir)
print(perf_counter()-t0)


# In[ ]:




