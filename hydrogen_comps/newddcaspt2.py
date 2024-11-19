#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from DDCASPT2 import DDCASPT2
import pickle, os, shutil
from glob import glob
import numpy as np
from joblib import Parallel, delayed
import pandas as pd


# In[ ]:


radius_range=np.linspace(0.6,3,100)
chains=np.arange(2,14,2)

# train_ind,test_ind=train_test_split(radius_range, test_size=0.3, random_state=0)
# print(len(train_ind),len(test_ind))
# with open('train_ind.pickle', 'wb') as handle:
#     pickle.dump(train_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('test_ind.pickle', 'wb') as handle:
#     pickle.dump(test_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_ind.pickle', 'rb') as handle:
    test_ind = pickle.load(handle)

with open('train_ind.pickle', 'rb') as handle:
    train_ind = pickle.load(handle)
    
print(len(train_ind),len(test_ind))    


# In[ ]:


# for i in chains:
topdir = os.getcwd()
basis_set='ANO-RCC-VDZP'


# In[ ]:


def run(i):
    dirname=f'H{i}_chain'
    print(dirname)
    if os.path.exists(dirname)==False:
        os.mkdir(dirname)
        
    # os.chdir(os.path.join(topdir,dirname))    
    for idxr, r in enumerate(radius_range):
        
        # Loop radius
        name=f"H{i}_{r:.2f}"
        
        # Create files
        subdirpath = os.path.join(topdir,dirname,f'{name}')
        if os.path.exists(subdirpath)==False:
            os.mkdir(subdirpath)
        if os.path.exists(os.path.join(subdirpath,f'{name}.csv'))==False:
            shutil.rmtree(os.path.join('tmp'),name)
            
            # Write xyz
            with open(os.path.join(subdirpath,f'{name}.xyz'),'w') as f:
                f.write(f'{i}\n\n')
                for j in range(i):
                    f.write(f'H {0:>8f} {0:>8f} {j*r:>8f}\n')
            print(subdirpath)
            if idxr==0:
                d = DDCASPT2(subdirpath,basis_set,name,i,i,0,previous=None)(run=False)
            else:            
                previous=os.path.join(topdir,dirname,f'H{i}_{radius_range[idxr-1]:.2f}',f"H{i}_{radius_range[idxr-1]:.2f}.RasOrb")
                print(previous)
                d = DDCASPT2(subdirpath,basis_set,name,i,i,0,previous=previous)(run=False)
    
    for idxr, r in enumerate(radius_range):
        # Loop radius
        name=f"H{i}_{r:.2f}" 
        
        subdirpath = os.path.join(topdir,dirname,f'{name}')
        
        
        for j in glob(os.path.join(subdirpath,"*GMJ*.csv"))+glob(os.path.join(subdirpath,"*Orb*"))+glob(os.path.join(subdirpath,"*h5"))+glob(os.path.join(subdirpath,"xmldump")):
            os.remove(j)        
    


# In[ ]:


Parallel(n_jobs=-1)(delayed(run)(i) for i in chains)


# In[ ]:




