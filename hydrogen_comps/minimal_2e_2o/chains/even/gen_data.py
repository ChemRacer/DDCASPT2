#!/usr/bin/env python
# coding: utf-8


from tqdm import tqdm
import sys
# !{sys.executable} -m pip install --upgrade  h5py
#######################################################
#Import packages
import numpy as np
import os
import re
from math import sin, cos, pi
from glob import glob
import subprocess
import pickle
from subprocess import call, check_output
import pandas as pd
# import psi4
from joblib import Parallel,effective_n_jobs,delayed
from time import time
import matplotlib.pyplot as plt
from plumbum.cmd import grep, awk

import shutil
import random
import sklearn
from shutil import copy
import csv
import h5py as h5
import seaborn as sns; sns.set(style="ticks", color_codes=True)

from sklearn.model_selection import train_test_split




#######################################################
# Obital labels
## Active t,u,v
## Virtual a,b

## Type 1: IA->AA
## Type 2: II->AA (P)
## Type 3: II->AA (M)
## Type 4: AA->VA
## Type 5: IA->VA/AV
## Type 6: II->AV (P)
## Type 7: II->AV (M)
## Type 8: AA->VV (P)
## Type 9: AA->VV (M)
## Type 10: IA->VV (P)
## Type 11: IA->VV (M)
## Type 12: II->VV (P)
## Type 13: II->VV (M)

## A: IA->AA
## B: II->AA
## C: AA->VA
## D: IA->VA/AV
## E: II->AV
## F: AA->VV
## G: IA->VV 
## H: II->VV
#######################################################









# Delete excessive extra files
def del_useless():
    '''
    Delete the extra files
    '''
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            for j in ['status','GssOrb','LprOrb','LoProp','guessorb','xmldump','RasOrb','SpdOrb']:
                if j in file:
    #                 print(root,dirs,file)
                    os.remove(os.path.join(root,file))




# When restarting a setr of calculations just clear everyting out
def clean_dir():
    for entry in os.scandir(path=os.getcwd()):
        if entry.is_dir():
            if entry.name=='Fock':
                shutil.rmtree(entry.name)
            if entry.name=='hdf5':
                shutil.rmtree(entry.name)
            if entry.name=='e2':
                shutil.rmtree(entry.name)                
            if entry.name=='Labels':
                shutil.rmtree(entry.name)
            if entry.name=='Coords':
                shutil.rmtree(entry.name)
            if 'dir' in entry.name:
                shutil.rmtree(entry.name)
                




# Run this before clean_dir, this pulls the xyz files out just to 
def pull_xyz():
    import re
    for i in struct_name:
        if os.path.exists(os.path.join(os.getcwd(),i))==False and os.path.exists(os.path.join(os.getcwd(),'Coords')):
            shutil.copy(os.path.join(os.getcwd(),'/'.join(('Coords',i))),os.path.join(os.getcwd(),i))









def gen_gateway(name,basis_set):
    string=f'''&GATEWAY 
coord={f'{name}.xyz'}
Basis = {basis_set}
Group = nosymm
Expert
'''
    return string

def gen_seward():
    string=f'''&SEWARD
'''
    return string

def gen_motra(name):
    string=f'''&MOTRA
Frozen=0
>>> COPY $WorkDir/GMJ_one_int_indx.csv $CurrDir/{name}.GMJ_one_int_indx.csv
>>> COPY $WorkDir/GMJ_one_int.csv $CurrDir/{name}.GMJ_one_int.csv
>>> COPY $WorkDir/GMJ_two_int_indx.csv $CurrDir/{name}.GMJ_two_int_indx.csv
>>> COPY $WorkDir/GMJ_two_int.csv $CurrDir/{name}.GMJ_two_int.csv
'''
    return string

def gen_scf(name):
    string=f"""&SCF &END
>>> COPY $WorkDir/{name}.scf.h5 $CurrDir/
"""
    return string    


def gen_rasscf(name,e,o,i,previous=None):
    start_string="""&RASSCF &END
Title= RASSCF
"""
    if previous!=None:
        fileorb=f"""FileOrb
{previous}
"""
    else:
        fileorb=""

    end_string=f"""
NACTEL
{e} 0 0
Inactive
{i}
RAS2
{o}
Symmetry
1
Spin
1
orblisting
all
ITERation
200 100
CIMX
200
SDAV
500
PRWF
0
PRSD
>>> COPY $WorkDir/{name}.rasscf.h5 $CurrDir/
>>> COPY $WorkDir/GMJ_Fock_MO.csv $CurrDir/{name}.GMJ_Fock_MO.csv
"""
    return start_string+fileorb+end_string 

def gen_caspt2():
    string="""&CASPT2 &END
Frozen 
0
Imaginary Shift
0.0

>>foreach i in (B,E,F,G,H)
>>foreach j in (P,M)
>>if ( -FILE GMJ_e2_${i}_${j}.csv )
>>> COPY $WorkDir/GMJ_RHS_${i}_${j}.csv $CurrDir/GMJ_RHS_${i}_${j}.csv
>>> COPY $WorkDir/GMJ_IVECW_${i}_${j}.csv $CurrDir/GMJ_IVECW_${i}_${j}.csv
>>> COPY $WorkDir/GMJ_e2_${i}_${j}.csv $CurrDir/GMJ_e2_${i}_${j}.csv
>>endif
>>enddo
>>enddo

>>foreach i in (A,C,D)
>>if ( -FILE GMJ_e2_$i.csv )
>>> COPY $WorkDir/GMJ_RHS_$i.csv $CurrDir/GMJ_RHS_$i.csv
>>> COPY $WorkDir/GMJ_IVECW_$i.csv $CurrDir/GMJ_IVECW_$i.csv
>>> COPY $WorkDir/GMJ_e2_$i.csv $CurrDir/GMJ_e2_$i.csv
>>endif
>>enddo

"""
    return string    












basis_set='ANO-RCC-VDZP'




top='/home/grierjones/DDCASPT2/hydrogen_comps/minimal_2e_2o/chains/even'




top




radius_range=np.flip(np.linspace(0.7,3,100))
# chains=np.arange(2,14,2)
# chains=np.arange(2,10,2)
chains=np.arange(6,20,2)



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




def gen_data(i):
    dirname=f'H{i}_chain'
    print(f"{i}e in {i}o with {0} inactive ({i} total MOs)")
    if os.path.exists(dirname)==False:
        os.mkdir(dirname)
        
        
    for idxr, r in tqdm(enumerate(radius_range)):
        
        # Loop radius
        name=f"H{i}_{r:.2f}"

        # Create files
        if os.path.exists(os.path.join(dirname,f'{name}'))==False:
            os.mkdir(os.path.join(dirname,f'{name}'))

        # Write xyz
        with open(os.path.join(dirname,f'{name}',f'{name}.xyz'),'w') as f:
            f.write(f'{i}\n\n')
            for j in range(i):
                f.write(f'H {0:>8f} {0:>8f} {j*r:>8f}\n')

        # Write input
        with open(os.path.join(dirname,f'{name}',f'{name}.input'),'wb') as g:
            g.write(gen_gateway(name,basis_set).encode())
            g.write(gen_seward().encode())
            g.write(gen_motra(name).encode())
            g.write(gen_scf(name).encode())   
            # Choose active space and inactive orbitals
            #g.write(gen_rasscf(name,2,2,int((i/2)-1)).encode())
            if idxr==0:
                g.write(gen_rasscf(name,i,i,0,previous=None).encode()) # int((i/2)-1)
            else:

                previous=os.path.join(top,dirname,f'H{i}_{radius_range[idxr-1]:.2f}',f"H{i}_{radius_range[idxr-1]:.2f}.RasOrb")
                g.write(gen_rasscf(name,i,i,0,previous=previous).encode()) # int((i/2)-1)
            g.write(gen_caspt2().encode())

        # Change dir
        if os.getcwd()!=os.path.join(dirname,f'{name}'):    
            os.chdir(os.path.join(dirname,f'{name}'))

        # Run
        call(['pymolcas','-new','-clean',f'{name}.input', '-oe', f'{name}.output'])


        # Back to top dir
        if os.getcwd()!=top:
            os.chdir(top)

    del_useless()








os.chdir('/home/grierjones/DDCASPT2/hydrogen_comps/minimal_2e_2o/chains/even')


    
# results = Parallel(n_jobs=9,verbose=10)(delayed(gen_data)(i) for i in chains)    
# for i in chains:
#     print(i)
#     gen_data(i)
gen_data(14)
del_useless()
