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


# In[ ]:


# radius_range=np.arange(106,182,0.5)
radius_range=np.arange(106,182,0.25)
# radius_range=np.arange(106,182,1)

train_ind,test_ind=radius_range[0::2],radius_range[1::2]
# train_test_split(radius_range, test_size=0.3, random_state=0)
print(len(train_ind),len(test_ind))
with open('big_train_ind.pickle', 'wb') as handle:
    pickle.dump(train_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('big_test_ind.pickle', 'wb') as handle:
    pickle.dump(test_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('test_ind.pickle', 'rb') as handle:
#     test_ind = pickle.load(handle)

# with open('train_ind.pickle', 'rb') as handle:
#     train_ind = pickle.load(handle)
    
print(len(train_ind),len(test_ind))    


# In[ ]:


# for i in chains:
topdir = os.getcwd()

print(topdir)


# In[ ]:


def gen_run(name,path):
    with open(os.path.join(path,f'run.sh'),'w') as g:
        g.write(f"""#!/bin/bash
#This file is a submission script to request the ISAAC resources from Slurm
#SBATCH --account=ACF-UTK0011             # The project account to be charged
#SBATCH --job-name={name}		       #The name of the job
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=4          # cpus per node
#SBATCH --partition=campus            # If not specified then default is "campus"
#SBATCH --qos=campus
#SBATCH --time=0-04:00:00             # Wall time (days-hh:mm:ss)
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
pymolcas --new --clean {name}.input -oe {name}.output
# CLEAN-UP AND EXIT
rm -r $SLURM_SUBMIT_DIR/$SLURM_JOBID
echo "Program finished with exit code $? at: $(date)"
""")    


# In[ ]:


def run(dirname,basis_set):
    if os.path.exists(basis_set)==False:
        os.mkdir(basis_set)
        
    # os.chdir(os.path.join(topdir,dirname))    
    for idxr, r in tqdm(enumerate(radius_range)):
        
        # Loop radius
        name=f"{dirname}_{r:.2f}"
        print(name)
        # Create files
        subdirpath = os.path.join(topdir,basis_set,f'{name}')
        if os.path.exists(subdirpath)==False:
            os.mkdir(subdirpath)
        print(subdirpath)
        # if os.path.exists(os.path.join(subdirpath,f'{name}.csv'))==False:
        shutil.rmtree(os.path.join('tmp'),name)
        # gen_run(subdirpath,subdirpath)
        # Write xyz
        with open(os.path.join(subdirpath,f'{name}.xyz'),'w') as f:
            f.write(f'{3}\n\n')
            
            radius=1.278
            f.write(f"""O {radius*cos(((float(r)/2)*(pi/180))):>8f} {radius*sin((float(r)/2)*(pi/180)):>8f} {0.0000:>8f}
O {0:>8f} {0:>8f} {0:>8f}
O {radius*cos(-(float(r)/2)*(pi/180)):>8f} {radius*sin(-(float(r)/2)*(pi/180)):>8f} {0:>8f}
""")

        
        if idxr==0:
            
            scfprevious=None
            d = DDCASPT2(subdirpath,basis_set,name,12,9,6,scf_previous=scfprevious,casscf_previous=scfprevious,symmetry=1,spin=0,UHF=False,CIROOT="3 3 1",frozen=0,pt2maxiter=50,MSroots=3,charge=0,n_jobs=-1)(inputwrite=False,run=False,feat=True)
            
            
        else:            
            
            
            scfprevious=os.path.join(topdir,basis_set,f'{dirname}_{radius_range[idxr-1]:.2f}',f"{dirname}_{radius_range[idxr-1]:.2f}.ScfOrb")
            casscf_previous=os.path.join(topdir,basis_set,f'{dirname}_{radius_range[idxr-1]:.2f}',f"{dirname}_{radius_range[idxr-1]:.2f}.RasOrb")
            d = DDCASPT2(subdirpath,basis_set,name,12,9,6,scf_previous=scfprevious,casscf_previous=casscf_previous,symmetry=1,spin=0,UHF=False,CIROOT="3 3 1",frozen=0,pt2maxiter=50,MSroots=3,charge=0,n_jobs=-1)(inputwrite=False,run=False,feat=True)
            
    
    # for idxr, r in enumerate(radius_range):
    #     # Loop radius
    #     name=f"{dirname}_{r:.2f}"
        
    #     subdirpath = os.path.join(topdir,basis_set,f'{name}')
        
        
    #     # for j in glob(os.path.join(subdirpath,"*GMJ*.csv"))+glob(os.path.join(subdirpath,"*Orb*"))+glob(os.path.join(subdirpath,"*h5"))+glob(os.path.join(subdirpath,"xmldump")):
    #     #     os.remove(j)        
    


# In[ ]:


run('ozone','ANO-RCC-VTZP')


# In[ ]:




