#!/usr/bin/env python
# coding: utf-8



from DDCASPT2 import DDCASPT2
import pickle, os, shutil
from glob import glob
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from math import sin, cos, pi
from tqdm.notebook import tqdm




radius_range=np.arange(106,182,0.5)



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




# for i in chains:
topdir = os.getcwd()





def run(dirname,basis_set):
    if os.path.exists(basis_set)==False:
        os.mkdir(basis_set)
        
    # os.chdir(os.path.join(topdir,dirname))    
    for idxr, r in tqdm(enumerate(radius_range)):
        
        # Loop radius
        name=f"{dirname}_{r:.2f}"
        
        # Create files
        subdirpath = os.path.join(topdir,basis_set,f'{name}')
        if os.path.exists(subdirpath)==False:
            os.mkdir(subdirpath)
        if os.path.exists(os.path.join(subdirpath,f'{name}.csv'))==False:
            shutil.rmtree(os.path.join('tmp'),name)
            
            # Write xyz
            with open(os.path.join(subdirpath,f'{name}.xyz'),'w') as f:
                f.write(f'{3}\n\n')
                
                radius=1.278
                f.write(f"""O {radius*cos(((float(r)/2)*(pi/180))):>8f} {radius*sin((float(r)/2)*(pi/180)):>8f} {0.0000:>8f}
O {0:>8f} {0:>8f} {0:>8f}
O {radius*cos(-(float(r)/2)*(pi/180)):>8f} {radius*sin(-(float(r)/2)*(pi/180)):>8f} {0:>8f}
""")

            
            if idxr==0:
                d = DDCASPT2(subdirpath,basis_set,name,4,3,10,previous=None)()
            else:            
                previous=os.path.join(topdir,basis_set,f'{dirname}_{radius_range[idxr-1]:.2f}',f"{dirname}_{radius_range[idxr-1]:.2f}.RasOrb")
                print(previous)
                d = DDCASPT2(subdirpath,basis_set,name,4,3,10,previous=previous)()
    
    for idxr, r in enumerate(radius_range):
        # Loop radius
        name=f"{dirname}_{r:.2f}"
        
        subdirpath = os.path.join(topdir,basis_set,f'{name}')
        
        
        for j in glob(os.path.join(subdirpath,"*GMJ*.csv"))+glob(os.path.join(subdirpath,"*Orb*"))+glob(os.path.join(subdirpath,"*h5"))+glob(os.path.join(subdirpath,"xmldump")):
            os.remove(j)        
    




# run('ozone','ANO-RCC-VDZP')




run('ozone','ANO-RCC-VTZP')






