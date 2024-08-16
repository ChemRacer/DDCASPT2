#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
# !{sys.executable} -m pip install plumbum h5py seaborn 
#######################################################
#Import packages
import numpy as np
import os
os.environ['MOLCAS']='/home/grierjones/Test/build'


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
from time import perf_counter


# In[ ]:


#######################################################
# Obital labels
## Inactive i,j
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


# In[ ]:


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


# In[ ]:


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
                


# In[ ]:


# Run this before clean_dir, this pulls the xyz files out just to 
def pull_xyz():
    import re
    for i in struct_name:
        if os.path.exists(os.path.join(os.getcwd(),i))==False and os.path.exists(os.path.join(os.getcwd(),'Coords')):
            shutil.copy(os.path.join(os.getcwd(),'/'.join(('Coords',i))),os.path.join(os.getcwd(),i))


# In[ ]:





# In[ ]:


def gen_gateway(name,basis_set):
    string=f'''&GATEWAY 
coord={f'{name}.xyz'}
Basis = {basis_set}
Group = nosymm
Expert
End of Input
'''
    return string

def gen_seward():
    string=f'''&SEWARD
End of Input
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
        fileorb=''

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





# In[ ]:


os.getcwd()


# In[ ]:


basis_set='ANO-RCC-MB'


# In[ ]:


top=os.getcwd()
# top="/lustre/isaac/proj/UTK0022/GMJ/DDCASPT2/ozone"


# In[ ]:





# In[ ]:





# In[ ]:


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


# In[ ]:


def gen_data(verbose=True):
    
    dirname=f'O3'
    if os.path.exists(dirname)==False:
        if verbose:
            print(f'Making {dirname}')
        os.mkdir(dirname)
        
        
    for idxr, r in enumerate(radius_range):
        t0=perf_counter()
        # Loop radius
        name=f"O3_{float(r):.2f}"
        subdirname=os.path.join(dirname,f'{name}')
        
        # Create files
        if os.path.exists(subdirname)==False:
            if verbose:
                print(f'Making {subdirname}')
            os.mkdir(subdirname)
            

        # Write xyz
        with open(os.path.join(subdirname,f'{name}.xyz'),'w') as f:
            f.write(f'{3}\n\n')
            
            radius=1.278
            f.write(f"""O {radius*cos(((float(r)/2)*(pi/180))):>8f} {radius*sin((float(r)/2)*(pi/180)):>8f} {0.0000:>8f}
O {0:>8f} {0:>8f} {0:>8f}
O {radius*cos(-(float(r)/2)*(pi/180)):>8f} {radius*sin(-(float(r)/2)*(pi/180)):>8f} {0:>8f}
""")
        if verbose:
            print(f"xyz created")
        # Write input
        with open(os.path.join(subdirname,f'{name}.input'),'wb') as g:
            g.write(gen_gateway(name,basis_set).encode())
            g.write(gen_seward().encode())
            g.write(gen_motra(name).encode())
            g.write(gen_scf(name).encode())   
            # Choose active space and inactive orbitals
            #g.write(gen_rasscf(name,2,2,int((i/2)-1)).encode())
            if idxr==0:
                g.write(gen_rasscf(name,4,3,10,previous=None).encode()) # int((i/2)-1)
            else:

                previous=os.path.join(top,dirname,f'O3_{float(radius_range[idxr-1]):.2f}',f"O3_{float(radius_range[idxr-1]):.2f}.RasOrb")
                g.write(gen_rasscf(name,4,3,10,previous=previous).encode()) # int((i/2)-1)
            g.write(gen_caspt2().encode())

        if verbose:
            print(f"Input created")
        # Change dir
        if os.getcwd()!=subdirname:   
            if verbose:
                print(f"Changing directories to {subdirname}")
            os.chdir(subdirname)

        # Run
        call(['pymolcas','-new','-clean',f'{name}.input', '-oe', f'{name}.output'])
        if verbose:
            print(f"Calculation ran in {perf_counter()-t0:.2f} seconds")
        
        for c in glob("*csv"):
            if 'e2' in c or 'IVECW' in c:
                pd.read_csv(c).to_csv(c,compression='zip')
            else:
                pd.read_csv(c,header=None).to_csv(c,compression='zip') 
        if verbose:
            print("Compression done")
        
        # Back to top dir
        if os.getcwd()!=top:
            if verbose:
                print(f"Moving back to {top}")
            os.chdir(top)
        print()


# In[12]:



# In[ ]:


def gen_energy():
    energy=[]
    dirname=f'O3'
    drop=[]
    # Loop radius
    for idr,r in enumerate(radius_range):
        name=f"O3_{float(r):.2f}"
        try:
            output=os.path.join(dirname,f'{name}',f'{name}.output')
            energy.append([r,float((grep['-i', '::    CASPT2',output] | awk['{print $NF }'])())])
        except:
            energy.append([r,0])
            drop.append(idr)

    pd.DataFrame(energy,columns=['radius','energy']).to_csv(f'O3/CASPT2.csv')

    casscf_energy=[]
    dirname=f'O3'
    # Loop radius
    for idr,r in enumerate(radius_range):
        name=f"O3_{float(r):.2f}"
        try:
            output=os.path.join(dirname,f'{name}',f'{name}.output')
            casscf_energy.append([r,float((grep['-i', '::    RASSCF root number  1',output] | awk['{print $8 }'])())])
        except:
            casscf_energy.append([r,0])

    pd.DataFrame(casscf_energy,columns=['radius','energy']).to_csv(f'O3/CASSCF.csv')       





    E2_energy=[]
    dirname=f'O3'
    # Loop radius
    for idr,r in enumerate(radius_range):
        name=f"O3_{float(r):.2f}"
        try:
            output=os.path.join(dirname,f'{name}',f'{name}.output')
            E2_energy.append([r,float((grep['-i', 'E2 (Variational):',output] | awk['{print $NF }'])())])
        except:
            E2_energy.append([r,0])
    pd.DataFrame(E2_energy,columns=['radius','energy']).to_csv(f'O3/E2.csv')        



# In[ ]:





# In[ ]:


# gen_data()
# del_useless()


# In[ ]:


# gen_energy()


# In[ ]:


cmap=sns.color_palette('rocket',7)


# In[ ]:





# In[ ]:


# gen_energy()
# fig,ax=plt.subplots(2,2,figsize=(10,6),sharex=True)
CASSCF=pd.read_csv(f"O3/CASSCF.csv",index_col=0)
CASPT2=pd.read_csv(f"O3/CASPT2.csv",index_col=0)
plt.plot(CASSCF['radius'],CASSCF['energy']-CASSCF.iloc[-1]['energy'],label='CASSCF')
plt.plot(CASPT2['radius'],CASPT2['energy']-CASPT2.iloc[-1]['energy'],label='CASPT2')
plt.title('O$_{3}$')
plt.xlabel("Radius (Å)")
plt.ylabel("Energy (E$_{h}$)")
plt.legend()
plt.tight_layout()
plt.savefig('energies.png',dpi=300,bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:


cwd = os.getcwd()
#   Keep everything at float64
DTYPE = float
# DTYPE = np.float16

#   Create an array with the easy data
def createArrray(filename):
    files = sorted(glob(filename))
    arrayname = []
    for i in sorted(files):
        arrayname.append(
            np.stack(
                np.array(pd.read_csv(i, header=None),
                         dtype=DTYPE,
                         copy=False).flatten()))

    arrayname = np.asarray(arrayname, dtype=DTYPE)
    return arrayname


# In[ ]:


#   Start transforming the HDF5 files from the data directory
h5list = sorted(glob('O3/*/*rasscf.h5'))
f = h5.File(h5list[0], 'r')
datasetNames = [n for n in f.keys()]
b = []
labels = []

# Useful attributes from the hdf5 files
NBAS=[]
NACTEL=[]
for k, elem in enumerate(h5list):
    for count, ele in enumerate([i for i in f.attrs]):
        if ele =='NBAS':
            for i, elemt in enumerate(np.array(h5.File(h5list[k],'r').attrs[ele]).reshape(-1)):
                NBAS.append(elemt)
        if ele =='NACTEL':
            for i, elemt in enumerate(np.array(h5.File(h5list[k],'r').attrs[ele]).reshape(-1)):
                NACTEL.append(elemt)


MO_ENERGIES=[]
MO_OCCUPATIONS=[]
MO_TYPEINDICES=[]
MO_VECTORS=[]
t0=time()
#   Eliminate certain features that won't be good for regression
for k, elem in enumerate(h5list):
    for count, ele in enumerate([n for n in h5.File(elem, 'r').keys()]):
        if ele =='MO_TYPEINDICES':
            for i, elemt in enumerate(np.array(h5.File(elem,'r')[ele]).reshape(-1)):
                MO_TYPEINDICES.append(elemt)

        if ele =='MO_ENERGIES':
            for i, elemt in enumerate(np.array(h5.File(elem,'r')[ele]).reshape(-1)):
                MO_ENERGIES.append(elemt)

        if ele =='MO_OCCUPATIONS':
            for i, elemt in enumerate(np.array(h5.File(elem,'r')[ele]).reshape(-1)):
                MO_OCCUPATIONS.append(elemt)




print(f'time: {time()-t0} s')
shape=len(h5list),int(NBAS[0])
# AO_FOCKINT_MATRIX=np.array(AO_FOCKINT_MATRIX).reshape(len(dislist),int(NBAS[0]),int(NBAS[0]))
MO_ENERGIES= np.array(MO_ENERGIES).reshape(shape)
MO_OCCUPATIONS= np.array(MO_OCCUPATIONS).reshape(shape)
MO_TYPEINDICES=np.array(MO_TYPEINDICES).reshape(shape)


# In[ ]:


h5list_scf = sorted(glob('O3/*/*.scf.h5'))
f = h5.File(h5list_scf[0], 'r')
datasetNames = [n for n in f.keys()]
b = []
labels = []
# AO_FOCKINT_MATRIX=[]
# Useful attributes from the hdf5 files
NBAS=[]
NACTEL=[]
for k, elem in enumerate(h5list_scf):
    for count, ele in enumerate([i for i in f.attrs]):
        if ele =='NBAS':
            for i, elemt in enumerate(np.array(h5.File(h5list_scf[k],'r').attrs[ele]).reshape(-1)):
                NBAS.append(elemt)
MO_VECTORS=[]
scf_F=[]  
scf_OCC=[]
t0=time()
#   Eliminate certain features that won't be good for regression
for k, elem in enumerate(h5list_scf):
    for count, ele in enumerate([n for n in h5.File(h5list_scf[k], 'r').keys()]):
        if ele =='MO_VECTORS':
            for i, elemt in enumerate(np.array(h5.File(h5list_scf[k],'r')[ele]).reshape(-1)):
                MO_VECTORS.append(elemt)
        if ele =='MO_ENERGIES':
            for i, elemt in enumerate(np.array(h5.File(h5list_scf[k],'r')[ele]).reshape(-1)):
                scf_F.append(elemt)
        if ele =='MO_OCCUPATIONS':
            for i, elemt in enumerate(np.array(h5.File(h5list_scf[k],'r')[ele]).reshape(-1)):
                scf_OCC.append(elemt)







print(f'time: {time()-t0} s')
scf_F= np.array(MO_ENERGIES).reshape(shape)
scf_OCC= np.array(MO_OCCUPATIONS).reshape(shape)
MO_VECTORS=np.array(MO_VECTORS).reshape(len(h5list),int(NBAS[0]),int(NBAS[0]))





# In[ ]:


typ_exists=sorted(set([os.path.basename(i).replace('GMJ_e2_','').replace('.csv','') for i in glob(os.path.join(os.getcwd(),'O3/*/GMJ_e2_*.csv'))]))


# In[ ]:





# In[ ]:


# Generate the labels that match the IVECW and IVECC2 files
def gen_labels(path,typ):
    return [j.split()[0].replace('\n','').replace('00','').replace('S0','S').replace('I0','I').replace(',','') for j in pd.read_csv(f'{path}/GMJ_RHS_{typ}.csv',compression='zip',index_col=0)['0']]




# In[ ]:


def gen_pair_labels(path,typ):
    Labels=[]
    for i in pd.read_csv(f'{path}/GMJ_RHS_{typ}.csv',compression='zip',index_col=0)['0']:
        Labels.append('_'.join([''.join((j[0],j[1:].lstrip('0'))) for j in i.split('_')]))
    return sorted(Labels)



# In[ ]:


def gen_dim_dict(path,typ_exists):
    '''    
    Dimension check for DDCASPT2: check the ordering of the pair-energies,
    this notation follows a mix of the papers and code.
    
    A (IA->AA): \n TIUV \n E_{ti} E_{uv} \n pqrs=tiuv=0123 \n    
    B_P (II->AA) (P): \n IJTU \n E_{ti} E_{uj} \n pqrs=tiuj=2031 \n
    B_M (II->AA) (M): \n IJTU \n E_{ti} E_{uj} \n pqrs=tiuj=2031 \n
    C (AA->VA): \n UVAT \n E_{at} E_{uv} \n pqrs=atuv=2301 \n
    D (IA->VA/AV): \n IUAT/IUTA \n E_{ai} E_{tu}/E_{ti} E_{au} \n pqrs=(a/t)i(t/a)u=2031 \n
    E_P (II->AV) (P): \n IJAT \n E_{ti} E_{aj} \n pqrs=tiaj=3021 \n
    E_M (II->AV) (M): \n IJAT \n E_{ti} E_{aj} \n pqrs=tiaj=3021 \n
    F_P (AA->VV) (P): \n TUAB \n E_{at} E_{bu} \n pqrs=atbu=2031 \n
    F_M (AA->VV) (M): \n TUAB \n E_{at} E_{bu} \n pqrs=atbu=2031 \n
    G_P (IA->VV) (P): \n ITAB \n E_{ai} E_{bt} \n pqrs=aibt=2031 \n
    G_M (IA->VV) (M): \n ITAB \n E_{ai} E_{bt} \n pqrs=aibt=2031 \n
    H_P (II->VV) (P): \n IJAB \n E_{ai} E_{bj} \n pqrs=aibj=2031 \n
    H_M (II->VV) (M): \n IJAB \n E_{ai} E_{bj} \n pqrs=aibj=2031 \n
    '''    
    dims = {typ:np.array(pd.read_csv(f"{path}/GMJ_e2_{typ}.csv",compression='zip',index_col=0).columns[0].split('=')[-1].split('x')).astype(int) for typ in typ_exists}
    return dims


# In[ ]:


typ_exists


# In[ ]:


dims_dict=gen_dim_dict([i for i in glob('O3/*') if os.path.isdir(i)][0],typ_exists)


# In[ ]:


dims_dict


# In[ ]:


def strip(lst):   
    return '_'.join(i.replace('A00','A').replace('I00','I').replace('S00','S').replace('I0','I').replace('A0','A').replace('S0','S') for i in lst.split('_'))


# In[ ]:


def gen_ordered(path,typ):
    '''
    Return a dataframe for each type
    Index=proper indexing
    level_0=row
    level_1=column
    0=W value
    '''
    
    ordered=pd.read_csv(os.path.join(path,f'GMJ_IVECW_{typ}.csv'),sep='\s+',compression='zip',header=None, skiprows=[0],index_col=0)
    if len(ordered.columns)==0:
        ordered=pd.read_csv(os.path.join(path,f'GMJ_IVECW_{typ}.csv'),sep=',',compression='zip',header=None, skiprows=[0],index_col=0)
        
    
        
    ordered.index=list(range(len(ordered.index)))
    ordered.columns=list(range(len(ordered.columns)))
    ordered=ordered.stack()

    df=pd.read_csv(os.path.join(path,f'GMJ_RHS_{typ}.csv'),delimiter=',',index_col=0,compression='zip',usecols=[1,2])
    
    df.index=list(map(strip,df.index))
    
    merged=ordered.reset_index().sort_values(by=0).set_index(df.sort_values(by='1').index).sort_values(['level_0','level_1'])    

    return merged


# In[ ]:


## Generate IVECW
def gen_e2(paths,typ):
    e2=[]
    
    for i in paths:
        proper_labels=gen_labels(i,typ)
        
        # df=pd.read_csv(os.path.join(i,f'GMJ_e2_{typ}.csv'),sep='\s+',compression='zip',header=None, skiprows=[0],index_col=0).astype(np.float64).stack()
        df=pd.read_csv(os.path.join(i,f'GMJ_e2_{typ}.csv'),sep='\s+',compression='zip',header=None, skiprows=[0],index_col=0)
        if len(df.columns)==0:
            df=pd.read_csv(os.path.join(i,f'GMJ_e2_{typ}.csv'),sep=',',compression='zip',header=None, skiprows=[0],index_col=0)        
        
        df.index=list(range(len(df.index)))
        df.columns=list(range(len(df.columns)))
        df=df.astype(float).stack()
        
        df.index=gen_ordered(i,typ).index
        df=df.to_frame(name=str(i.split('/')[1].split('_')[1]))
        e2.append(df)
    df1=pd.concat(e2,axis=1).loc[proper_labels]
    df1.index=[i for idx,i in enumerate(proper_labels)]
    return df1


# In[ ]:


def gen_pair(paths,typ):
    Y=gen_e2(paths,typ).astype(float)
# Needs to be qs, we're summing over the occupied orbitals    
    Y_pair_set=list(set(['_'.join((i.split('_')[0],i.split('_')[1]))+'_' for i in Y.index.tolist()]))
    Y_pair_df=pd.concat([Y[Y.index.str.find(j)==0].sum() for j in Y_pair_set],axis=1)
    Y_pair_df.columns=list(set(['_'.join((i.split('_')[0],i.split('_')[1])) for i in Y.index.tolist()]))
    return Y_pair_df.T.sort_index().groupby(level=0).sum()



# In[ ]:


def stack_label(path,typ):
    if f'{typ}_M' in typ_exists and f'{typ}_P' in typ_exists:
        return gen_pair_labels(path,f'{typ}_P')+gen_pair_labels(path,f'{typ}_M')
    elif f'{typ}_P' in typ_exists:
        return gen_pair_labels(path,f'{typ}_P')
    elif f'{typ}_M' in typ_exists:
        return gen_pair_labels(path,f'{typ}_M')
    elif f'{typ}' in typ_exists:
        return gen_pair_labels(path,f'{typ}')


# In[ ]:


def stack_e2(path,typ):
    if f'{typ}_M' in typ_exists and f'{typ}_P' in typ_exists:
        df=pd.concat([gen_pair(path,f'{typ}_M'),gen_pair(path,f'{typ}_P')],axis=0).groupby(level=0).sum()
        return df
    elif f'{typ}_P' in typ_exists:
        return gen_pair(path,f'{typ}_P').groupby(level=0).sum()
    elif f'{typ}_M' in typ_exists:
        return gen_pair(path,f'{typ}_M').groupby(level=0).sum()
    elif f'{typ}' in typ_exists:
        return gen_pair(path,f'{typ}').groupby(level=0).sum()


# In[ ]:





# In[ ]:


paths=glob('O3/O3_*/')
path=paths[0]
# Generate the data
for typ in set([i.split('_')[0] for i in typ_exists ]):
    if typ=='A':
        typA_e2=stack_e2(paths,typ)
        typA_labels=stack_label(path,typ)
        
    if typ=='B':        
        typB_e2=stack_e2(paths,typ)
        typB_labels=stack_label(path,typ)
        
    if typ=='C':
        typC_e2=stack_e2(paths,typ)
        typC_labels=stack_label(path,typ)
        
    if typ=='D':        
        typD_e2=stack_e2(paths,typ)
        typD_labels=stack_label(path,typ)
                
    if typ=='E':
        typE_e2=stack_e2(paths,typ)
        typE_labels=stack_label(path,typ)
                
    if typ=='F':        
        typF_e2=stack_e2(paths,typ)
        typF_labels=stack_label(path,typ)
                
    if typ=='G':        
        typG_e2=stack_e2(paths,typ)
        typG_labels=stack_label(path,typ)
                
    if typ=='H':  
        typH_e2=stack_e2(paths,typ)
        typH_labels=stack_label(path,typ)


# In[ ]:


typ_exists


# In[ ]:


stacked_e2=pd.concat([gen_e2(paths,typ) for typ in typ_exists]).groupby(level=0).sum()
E2Dict=pd.read_csv(f"O3/E2.csv",index_col=0).to_numpy()
# stacked_e2.columns=[float(i.split('/')[1].split('_')[1]) for i in stacked_e2.columns]
stacked_e2=stacked_e2.sum(axis=0).sort_index().reset_index().to_numpy()


# In[ ]:


plt.scatter(E2Dict[:,1],stacked_e2[:,1])
plt.plot(E2Dict[:,1],E2Dict[:,1],'k--')


# In[ ]:


typ_exists


# In[ ]:


stacked_pairs=pd.concat([stack_e2(paths,typ) for typ in typ_exists]).groupby(level=0).sum()
pair_labels=stacked_pairs.index.tolist()


# In[ ]:


dummy_stack=pd.concat([gen_e2(paths,typ) for typ in typ_exists])


# In[ ]:





# In[ ]:


def gen_indx(list_of_dicts):
    indx=[]
    for i in list_of_dicts.keys():
        if len(list_of_dicts[i])>0:
            indx.append(list(list_of_dicts[i].keys()))
    return indx[0]


path_check=sorted(glob('O3/O3_*/O3_*.output'))[0]

# Sanity check...
# REMEVDZPER FROZEN CORE APPROXIMATION
# Number of frozen orbitals
fro=int(subprocess.Popen(f"grep -i 'Frozen orbitals' {path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
# Number of inactive orbitals
inact=int(subprocess.Popen(f"grep -i 'Inactive orbitals' {path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
# Number of active orbitals
act=int(subprocess.Popen(f"grep -i 'Active orbitals' {path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
# Number of seconary orbitals
virt=int(subprocess.Popen(f"grep -i 'Secondary orbitals' {path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
# Number of basis functions for sanity check
bas_check=int(subprocess.Popen(f"grep -i 'Number of basis functions' {path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])

Basis_Indices=[]
for i in range(fro):
    Basis_Indices.append(f'F{i+1}')
for i in range(inact):
    Basis_Indices.append(f'I{i+1}')
for i in range(act):
    Basis_Indices.append(f'A{i+1}')
for i in range(virt):
    Basis_Indices.append(f'S{i+1}')    
    
print(f'Basis sanity check passed={bas_check==len(Basis_Indices)}') 


# In[ ]:


# Grab molecular orbital occupations and make it into a dataframe labeled with xyz file name
# MO_OCC=dict(zip(Basis_Indices,MO_OCCUPATIONS))
# MO_OCC_Dict=dict(zip([str(k) for k in paths],MO_OCC))
# MO_OCC_DF=pd.DataFrame(MO_OCC_Dict)

# Dataframe of MO occupation, index=basis indices and columns=paths
MO_OCCUPATIONS_DF=pd.DataFrame(MO_OCCUPATIONS,index=paths,columns=Basis_Indices).transpose()

MO_OCC_DF=MO_OCCUPATIONS_DF


# In[ ]:


# 
# Keep in mind HDF5 zeroes out the actrive orbitals... we'll use the Fock matrix to recover these
# 
# Grab molecular orbital energy and make it into a dataframe labeled with xyz file name


# Dataframe of MO energies, index=basis indices and columns=paths
MO_ENERGIES_DF=pd.concat([pd.read_csv(os.path.join(j,j.split('/')[1]+'.GMJ_Fock_MO.csv'),sep=',',compression='zip',index_col=0).set_index(pd.Index(Basis_Indices)).set_axis([j], axis=1) for j in paths],axis=1).sort_index(axis=1)


# In[ ]:


def gen_one_int():
    one_int=[]
    Labels=[]
    Indexes=[]
    upd_1int_indx=[]
    def one_gener(i):
        return pd.DataFrame(np.array([i.split() for i in pd.read_csv(os.path.join(i,f"{i.split('/')[1]}.GMJ_one_int.csv"),compression='zip',index_col=None)['0']]).astype(float),index=Basis_Indices,columns=Basis_Indices)

#     Dict=dict(zip(Indexes,Labels))
    return dict((i,one_gener(i)) for i in paths)





# In[ ]:


t0=time()
int1=gen_one_int()
print(f'Integrals loaded in {time()-t0:0.4f} s')


# In[ ]:





# In[ ]:


# pd.set_option("precision", 2)
# np.set_printoptions(precision=2)
# pd.options.display.float_format = '{:,.2f}'.format



nmo=len(Basis_Indices)
indice=[]
ad_ind=[]
for ind,i in enumerate(range(nmo)):
    for indx,j in enumerate(range(nmo)):
        ad_ind.append(f'{i+1}_{j+1}')
        if j<=i:
            indice.append(f'{i+1}_{j+1}')


# In[ ]:


len(indice),len(ad_ind)


# In[ ]:


# raw_MO=pd.DataFrame(np.genfromtxt(os.path.join(path_to_2_ints,f'{paths[0]}.GMJ_two_int.csv'), delimiter='',dtype=float),index=indice,columns=indice)

# # This is wrong
# for ind,a in enumerate(indice):
#     for indx,b in enumerate(indice):
#         if a!=b:
#             print(a.split('_'),b.split('_'))
#             # print(f'{j+1}_{i+1}',raw_MO[f'{j+1}_{i+1}'])
#             # raw_MO.loc[f'{j+1}_{i+1}']=AO_DF.loc[f'{i+1}_{j+1}']


# In[ ]:


import itertools

def gen_MO(k):
    
    if os.path.exists(os.path.join(k,f"{k.split('/')[1]}.GMJ_two_int.csv")):

        raw_MO=pd.read_csv(os.path.join(k,f"{k.split('/')[1]}.GMJ_two_int.csv"),compression='zip',index_col=0).to_numpy().reshape(len(Basis_Indices),len(Basis_Indices),len(Basis_Indices),len(Basis_Indices))
    return raw_MO


# In[ ]:





# In[ ]:


# featurelist=list()
# def GenerateFeatures(wf_object, Miller=False, values=4):
#     b =   wf_object.triplecheck

#     #c = np.log10(np.absolute(wf_object.get_MO('oovv')*wf_object.t2start))
#     ##infcheck=(c == -np.inf)
#     #c[infcheck]=20

#     #d = wf_object.pairs

#     if Miller == False:
#         for i in range(b.shape[0]):
#             for j in range (b.shape[1]):
#                 print('Top',i,j)
#                 featurelist.clear()
# # sort MP2 two-electron excitations e_{ij}^{ab} for ij
# # (most negative, …, smallest negative, …, smallest positive, …, largest positive)
# # [:values]=first x values        
# # [-values:]=last x values
#                 ind=np.argsort(b[i,j].flatten(),axis=0)
# # εij{MP2}
#                 new=np.sum(b[i,j])#0
# #                 featurelist.append('Pair_Energy')
# # <ii||jj>        
#                 new=np.hstack((new,wf_object.MO[i,i,j,j]))
# # screen1 first 4 values (i) <ii||aa>
#                 new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[:values]))#1,2,3,4
# # screen2 first 4 values (j) <jj||bb>
#                 new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
# # <aa||bb> matrix first 4 values 
#                 new=np.hstack((new,np.take_along_axis(wf_object.screenvirt[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
# # e_{ij}^{ab}{MP2} first 4 values
#                 new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values]))#17,18,19,20
#                 if i==j:       
# # <ii||jj> is excluded since i==j                
#                     ind=np.argsort(b[i,j].flatten(),axis=0)
# # εij{MP2}                    
#                     new=np.sum(b[i,j])#0
# # screen1 first 4 values (i)
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[:values]))#1,2,3,4
# # screen2 first 4 values (j)       
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
# # screen1 last 4 values (i)
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[-values:]))#1,2,3,4
# # screen2 last 4 values (j)
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[-values:]))#9,10,11,12

# # e_{ij}^{ab}{MP2} first 4 values                    
#                     new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values]))#17,18,19,20
# # e_{ij}^{ab}{MP2} last 4 values    
#                     new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:]))#17,18,19,20
# # Missing εij{MP2} correlation in representation                    
#                     one=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:])
#                     two=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values])
#                     new=np.hstack((new, np.sum(b[i,j])-one-two))
#                     featurelist.append('triplecheck1')
#                 else:
# # i!=j                    
#                     ind=np.argsort(b[i,j].flatten(),axis=0)
# # εij{MP2}    
#                     new=np.sum(b[i,j])#0      
# # <ii||jj>                    
#                     new=np.hstack((new,wf_object.MO[i,i,j,j]))
# # screen1 first 4 values (i)
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[:values]))#1,2,3,4
# # screen2 first 4 values (j)                    
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[:values]))#9,10,11,12
# # screen1 last 4 values (i)
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen1[i,j].flatten(), ind, axis=0)[-values:]))#1,2,3,4
# # screen2 last 4 values (j)                    
#                     new=np.hstack((new,np.take_along_axis(wf_object.screen2[i,j].flatten(), ind, axis=0)[-values:]))#9,10,11,12
# # e_{ij}^{ab}{MP2} first 4 values                                        
#                     new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values]))#17,18,19,20
# # e_{ij}^{ab}{MP2} last 4 values    
#                     new=np.hstack((new,np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:]))#17,18,19,20                    
# # Missing εij{MP2} correlation in representation                    
#                     one=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[-values:])
#                     two=np.sum(np.take_along_axis(b[i,j].flatten(), ind, axis=0)[:values])
#                     new=np.hstack((new, np.sum(b[i,j])-one-two))


#                 if ((i==0) and (j==0)):
#                     a=new.copy()
#                     diag=wf_object.pairs[i,j]
#                 elif ((i==0) and (j==1)):
#                     g=new.copy()
#                     offdiag=wf_object.pairs[i,j]
#                 elif (i==j):        
#                     a=np.vstack((a,new))#41
#                     diag=np.vstack((diag,wf_object.pairs[i,j]))
#                 elif (j > i):
#                     g=np.vstack((g,new))#41
#                     offdiag=np.vstack((offdiag,wf_object.pairs[i,j]))
#         return a,diag,g,offdiag


# In[ ]:





# In[ ]:


froz=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('F')]
inact=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('I')]
act=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('A')]
virt=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('S')]


# In[ ]:


# gen_F=dict(zip(paths,scf_F))
# gen_occ=dict(zip(paths,scf_OCC))

gen_F=MO_ENERGIES_DF
gen_occ=MO_OCCUPATIONS_DF
gen_F_SCF=pd.DataFrame(scf_F,columns=Basis_Indices,index=paths).T
gen_occ_SCF=pd.DataFrame(scf_OCC,columns=Basis_Indices,index=paths).T


# In[ ]:


MO_OCCUPATIONS_DF


# In[ ]:


gen_occ


# In[ ]:





# In[ ]:


full_set=sorted(set(sum([gen_pair_labels(path,typ) for typ in typ_exists],[])))


# In[ ]:


# Rewrite the jacob style featurization step
occcc=len(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=0])
virttt=len(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=2])




# # START HERE TOMORROW
# - Data processing

# In[ ]:


argged=np.argsort(dummy_stack.abs().values,axis=0).T
top_excits={}
for idxc,c in enumerate(dummy_stack.columns):
    top_excits[c]=dummy_stack[c].iloc[argged[idxc]].iloc[-4:].index


# In[ ]:


train_ind


# In[ ]:


class gen_big_4(object):
    '''
    Generate a set of i,j,k,l indices corresponding to the largest 4 two-electron excitations 
    per pair-energy.

    !!!From here on out we need to use the same training set and test set.!!!
    Indices:
    i-internal
    j-internal
    k-external
    l-external

    '''    
    # Pick 4 largest contributers to the pair energy
    def big_4(self,dist):
        return [['_'.join(k.split('_')[2:4]) for k in dummy_stack.loc[[ j for j in dummy_stack.index.tolist() if j.split('_')[0]==i.split('_')[0] and j.split('_')[1]==i.split('_')[1]]][dist].abs().sort_values(ascending=False).index[0:4].tolist()] for i in stacked_pairs.index.tolist()]
    
    def gen_ijkl(self):
        # Dimensions: training index x pair energies x top 4 most frequent virtual orbitals for two electrion excitations
        
        self.train_4=np.array(Parallel(n_jobs=6,verbose=10)(delayed(self.big_4)(f'{float(i):.2f}') for i in train_ind))
        print(self.train_4.shape)
        # Most frequent: list of tuples containing (pair energy index, [top 4 most frequent virtual orbitals for two electrion excitations])
        # This can serve as a label too.
        self.train_freq=[(i,pd.DataFrame(self.train_4[:,idx,:]).describe().loc['top'].tolist()) for idx,i in enumerate(stacked_pairs.index.tolist())]
        # i,j, set of 4 largest [(k,l)]
        self.set_ijkl_indices=[(Basis_Indices.index(i.split('_')[0]),Basis_Indices.index(i.split('_')[1]),[(Basis_Indices.index(k.split('_')[0]),Basis_Indices.index(k.split('_')[1])) for k in pd.DataFrame(self.train_4[:,idx,:]).describe().loc['top'].tolist()]) for idx,i in enumerate(stacked_pairs.index.tolist())]
        internal_basis=[i for i in Basis_Indices if 'S' not in i]
        external_basis=[i for i in Basis_Indices if 'I' not in i]
        print(external_basis)
    # This is 1 indices per pair energy!        
        set_i_indices=[]
    # This is 1 indices per pair energy!            
        set_j_indices=[]
    # This is a set of 4 indices per pair energy!    
        set_k_indices=[]
    # This is a set of 4 indices per pair energy!    
        set_l_indices=[]
        for idx,i in enumerate(stacked_pairs.index.tolist()):
            set_i_indices.append(internal_basis.index(i.split('_')[0]))
            set_j_indices.append(internal_basis.index(i.split('_')[1]))
            set_k_indices.append([external_basis.index(k.split('_')[0]) for k in pd.DataFrame(self.train_4[:,idx,:]).describe().loc['top'].tolist()])
            set_l_indices.append([external_basis.index(k.split('_')[1]) for k in pd.DataFrame(self.train_4[:,idx,:]).describe().loc['top'].tolist()])
            
        return set_i_indices,set_j_indices,set_k_indices,set_l_indices,self.train_freq


# In[ ]:


ijkl_idx=gen_big_4().gen_ijkl()
set_i_indices=ijkl_idx[0]
set_j_indices=ijkl_idx[1]
set_k_indices=ijkl_idx[2]
set_l_indices=ijkl_idx[3]
set_indices=ijkl_idx[4]


# In[ ]:


occcc=len(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=0])
virttt=len(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=2])





# In[ ]:


# Create slices for AA->AA indices
# This will help zero out infinities
internal_A=[idx for idx,i in enumerate(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=0].index) if 'A' in i]
external_A=[idx for idx,i in enumerate(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=2].index) if 'A' in i]

inner_slice=slice(min(internal_A),max(internal_A)+1)
outer_slice=slice(min(external_A),max(external_A)+1)


# In[ ]:


class gen_two_ints(object):
# Some of the D_{ij}^{ab}=f_{ii}+f_{jj}-f_{aa}-f_{bb}=e_{ii}+e_{jj}-e_{aa}-e_{bb} elements 
# will be 0 since ij and ab overlap for CASPT2
# ij \in {I,A}
# ab \in {A,V}
# So ignore the warnings since they'll only be 0 when ijab \in {A} only
    import warnings
    warnings.simplefilter('ignore')
    
    def get_MO(self, string):
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]] 
    
    def compute_pairmatrix(self,selft2start,selfdoublecheck):
        test = 2*selft2start*selfdoublecheck
        test -= np.swapaxes(selft2start,2,3)*selft2start
        c=np.sum(test,axis=(2,3))
        return c    
    
    def build_tau(self,t2,t1):
        ttau = t2.copy()
        tmp = np.einsum('ia,jb->ijab', t1, t1,optimize=True)
        ttau += tmp
        return ttau    
    
#     def __init__(self,k):
    def gen_feat(self,k):
# Set variables
# nocc:=Occupied orbitals
# nvirt:=virtual orbitals
# nfzc:=frozen orbitals
# nmo:=number of orbitals
        nocc=occcc
        nvirt=virttt
        self.nfzc=len(froz)
        self.nocc=nocc
        self.nvirt=nvirt
        self.nmo=len(Basis_Indices)
        nmo=nocc+nvirt
# nmo=nocc+nvirt
# Set slices
# In CASPT2 occupied run I-A and virtual run A-S
# Unlike MP2 there will be overlap between internal and external indices
        self.slice_o = slice(0,nocc)
        self.slice_v = slice(len(Basis_Indices)-virttt,len(Basis_Indices))
        self.slice_a = slice(0,nmo)    

# Dictionary with the slices        

        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a
        }
        
        
        

        featurelist=list()    
# Virtual
        self.empty=np.zeros((nvirt,))
# Occupied    
        self.occupado=np.zeros((nocc,))
# MO integrals    
        self.MO=gen_MO(k)
# MO Fock matrix    
        self.F=gen_F[k].to_numpy()
# Zero out t1 matrix, dim(occ,virt)    
        self.t1=np.zeros((nocc,nvirt))
# MO Fock matrix internal
        Focc = self.F[self.slice_o]
# MO Fock matrix external    
        Fvir = self.F[self.slice_v]  
        self.orbocc=Focc
        self.orbvirt=Fvir

        self.Dia = Focc.reshape(-1, 1) - Fvir
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir 

        
# Clean up AA block to get rid of infinities, we do not need this anyway              
# t2=<ij||ab>/(e_i+e_j-e_a-e_b)
# But this is <ij|ab>... idk...
        self.t2start=self.MO[self.slice_o, self.slice_o, self.slice_v,self.slice_v] / self.Dijab
# Zero out AA->AA
        self.t2start[inner_slice,inner_slice,outer_slice,outer_slice]=0

# Related to the correlation energy
# np.einsum('ijab->',triplecheck)=MP2 correlation energy in jacob's code
#dim(triplecheck)=(occ,occ,virt,virt)
        self.triplecheck=2*self.t2start*self.get_MO('oovv')
        self.triplecheck -=  np.swapaxes(self.get_MO('oovv'),2,3)*self.t2start 
        # Zero out NaNs
        self.triplecheck[np.argwhere(np.isnan(self.triplecheck))]=0
# Zero out AA->AA
        self.triplecheck[inner_slice,inner_slice,outer_slice,outer_slice]=0

        
        
# Integral
# doublecheck =<ij|ab>
#dim(doublecheck)=(occ,occ,virt,virt)
        self.doublecheck = self.MO[self.slice_o, self.slice_o, self.slice_v, self.slice_v]   
# Zero out AA->AA
        self.doublecheck[inner_slice,inner_slice,outer_slice,outer_slice]=0


    
# Not related to the correlation energy?
# dim(pairenergy)=(occ,occ,virt,virt)
# dim(compute_pairmatrix(t2start,doublecheck))=(occ,occ)->(occ,occ,virt,virt)
        self.pairenergy=(np.zeros(self.doublecheck.shape)+self.compute_pairmatrix(self.t2start,self.doublecheck)[:,:,np.newaxis,np.newaxis])
# Zero out AA->AA
        self.pairenergy[inner_slice,inner_slice,outer_slice,outer_slice]=0


# Related to the correlation energy
#dim(pairs)=(virt,virt)
# np.einsum('ab->',pairs)=np.einsum('ijab->',triplecheck)=MP2 correlation energy in jacob's code
        tmp_tau = self.build_tau(self.t2start,self.t1)
        self.pairs=2*tmp_tau*self.get_MO('oovv')
        self.pairs-= np.swapaxes(self.get_MO('oovv'),2,3)*tmp_tau
        self.pairs[np.argwhere(np.isnan(self.pairs))]=0
        self.pairs = np.sum(self.pairs,axis=(2,3))
# Zero out AA->AA
        self.pairs[inner_slice,outer_slice]=0        
        


        test=np.zeros(self.t2start.shape)
        self.diag=test
        for i in range (0,self.nocc):
            for j in range (0,self.nocc):
                np.fill_diagonal(self.diag[i,j,:,:],1)

# Dim(temp)=(occ,virt)
# Basically dim(<ii|jj>)->dim(<i|j>)
        temp=np.zeros((self.nocc,self.nvirt))
        for i in range (0,self.nocc):
            for j in range (0,self.nvirt):
                temp[i,j]=self.doublecheck[i,i,j,j]
# Dim(test1)=(occ,occ,virt,virt)                
        test1=np.zeros((self.t2start.shape))

# dim(temp)=(occ,virt) -> dim(temp[:,np.newaxis,:,np.newaxis])=(occ,1,virt,1)
# np.newaxis makes this a dummy index, : will be unique
# i.e. (occ,copy,virt,copy)
# (occ,occ,virt,virt) + (occ,1,virt,1)
# <ii||aa>    
        self.screen1=test1+temp[:,np.newaxis,:,np.newaxis]
# Zero out AA->AA
        self.screen1[inner_slice,inner_slice,outer_slice,outer_slice]=0            

# dim(temp)=(occ,virt) -> dim(temp[np.newaxis,:,np.newaxis,:])=(1,occ,1,virt)
# (occ,occ,virt,virt) + (1,occ,1,virt)
# np.newaxis makes this a dummy index, : will be unique
# i.e. (copy,occ,copy,virt)
# <jj||bb>    
        self.screen2=test1+temp[np.newaxis,:,np.newaxis,:]
# Zero out AA->AA
        self.screen2[inner_slice,inner_slice,outer_slice,outer_slice]=0                
# screen1[i,j,k,l]==screen2[j,i,l,k].T

# nfzc=# frozen core
        val=self.nmo-self.nfzc
# Dim(temp)=(nmo,nmo)
# Basically dim(<ii|jj>)->dim(<i|j>)    
        temp=np.zeros((val,val))        
        for i in range (0,val):
            for j in range (0,val):
                temp[i,j]=self.MO[i,i,j,j]
# Get virtual indices: <aa|bb> basically
# <aa||bb>
# (occ,occ,virt,virt) + (virt,virt)
# (occ,occ,virt,virt) + (1,1,virt,virt)                
        temp =temp[self.slice_v,self.slice_v]
        self.screenvirt=test1+temp[np.newaxis,np.newaxis,:,:]
# Zero out AA->AA
        self.screenvirt[inner_slice,inner_slice,outer_slice,outer_slice]=0
        # b=(i,j,a,b)=(13, 13, 5, 5)
        b=self.triplecheck
        
# Zero out AA->AA
        diag_indx=[]
        off_diag_indx=[]     
        featurelist=[]
        featurelist.clear()
        feature=[]
        feature.clear()    
        index=['pair_energy','coulomb','screen1_1','screen1_2','screen1_3','screen1_4','screen2_1','screen2_2','screen2_3','screen2_4','eijab_1','eijab_2','eijab_3','eijab_4','screenvirt_1','screenvirt_2','screenvirt_3','screenvirt_4']        
        for idx,i in enumerate([j for j,v in set_indices]):
# εij{MP2}              
            new=np.sum(b[set_i_indices[idx],set_j_indices[idx]])#
# <ii||jj>    
            new=np.hstack((new,self.MO[set_i_indices[idx],set_i_indices[idx],set_j_indices[idx],set_j_indices[idx]]))
# <ii||aa>    
            new=np.hstack((new,np.array([self.screen1[set_i_indices[idx],set_j_indices[idx],k,l] for k,l in zip(set_k_indices[idx],set_l_indices[idx])]).flatten()))
# <jj||bb>    
            new=np.hstack((new,np.array([self.screen2[set_i_indices[idx],set_j_indices[idx],k,l] for k,l in zip(set_k_indices[idx],set_l_indices[idx])]).flatten()))
# e_{ij}^{ab} MP2
            new=np.hstack((new,np.array([b[set_i_indices[idx],set_j_indices[idx],k,l] for k,l in zip(set_k_indices[idx],set_l_indices[idx])]).flatten()))
# # <aa||bb>    
            new=np.hstack((new,np.array([self.screenvirt[set_i_indices[idx],set_j_indices[idx],k,l] for k,l in zip(set_k_indices[idx],set_l_indices[idx])]).flatten()))
            featurelist.append((i,new))
        return pd.DataFrame(dict(featurelist),index=index).T
#         return pd.concat([pd.DataFrame(self.a,index=diag_indx),pd.DataFrame(self.g,index=off_diag_indx)]).loc[full_set]

# gen_feat
#     self.dict=dict([(k,gen_feat(k))for k in paths])





# In[ ]:


def elements(x):
    '''
    Takes an integer, x, and returns the number of off-diagonal elements of an upper triangular matrix
    f(x)=(x*(x-1))/2
    '''
    return (x*(x-1))/2


# In[ ]:


def gen_1_feats(k):
    # t0=time()
    # For an e_pq, there should be 10 F, 10 occs, and then 10 SCF F and 10 SCF occs
    Fp=[]
    Fq=[]

    occp=[]
    occq=[]

    SCFFp=[]
    SCFFq=[]


    SCFOCCp=[]
    SCFOCCq=[]

    hpp=[]
    hqq=[]
    hrr=[]
    hss=[]

    rs_df=[]
    h=int1[k]
    for idx,i in enumerate(set_indices):

        p,q=i[0].split('_')
        # print(p,q)
        Fp.append(gen_F.loc[p,k])
        Fq.append(gen_F.loc[q,k])
        occp.append(gen_occ.loc[p,k])
        occq.append(gen_occ.loc[q,k])

        SCFFp.append(gen_F_SCF.loc[p,k])
        SCFFq.append(gen_F_SCF.loc[q,k])
        SCFOCCp.append(gen_occ_SCF.loc[p,k])
        SCFOCCq.append(gen_occ_SCF.loc[q,k])
        hpp.append(h.loc[p,p])
        hqq.append(h.loc[p,p])



        Fr=[]
        Fs=[]
        occr=[]
        occs=[]
        SCFFr=[]
        SCFFs=[]
        SCFOCCr=[]
        SCFOCCs=[]
        for idxx,j in enumerate(i[1]):
            r,s=j.split('_')

            Fr.append((f'Fr{idxx+1}',gen_F.loc[r,k]))
            Fs.append((f'Fs{idxx+1}',gen_F.loc[s,k]))

            occr.append((f'occr{idxx+1}',gen_occ.loc[r,k]))
            occs.append((f'occs{idxx+1}',gen_occ.loc[s,k]))

            SCFFr.append((f'SCFFr{idxx+1}',gen_F_SCF.loc[r,k]))
            SCFFs.append((f'SCFFs{idxx+1}',gen_F_SCF.loc[s,k]))


            SCFOCCr.append((f'SCFOCCr{idxx+1}',gen_occ_SCF.loc[r,k]))
            SCFOCCs.append((f'SCFOCCs{idxx+1}',gen_occ_SCF.loc[s,k]))

            hrr.append((f'hrr{idxx+1}',h.loc[r,r]))
            hss.append((f'hss{idxx+1}',h.loc[s,s]))


        rs_df.append(pd.DataFrame.from_dict({**dict(Fr),**dict(Fs),**dict(occr),**dict(occs),**dict(SCFFr),**dict(SCFFs),**dict(SCFOCCr),**dict(SCFOCCs),**dict(hrr),**dict(hss)},orient='index',columns=[i[0]]))
    rs=pd.concat(rs_df,axis=1).T
    dummy_df=pd.DataFrame({'hpp':hpp,'hqq':hqq,'Fp':Fp,'Fq':Fq,'occp':occp,'occq':occq,'SCFFp':SCFFp,'SCFFq':SCFFq,'SCFOCCp':SCFOCCp,'SCFOCCq':SCFOCCq},index=pair_labels)
    # print(f'{k} {time()-t0}')
    return pd.concat([rs,dummy_df],axis=1)


# In[ ]:


def gen_two_el():
    return dict([(k.split('/')[1].lstrip('O3_'),gen_two_ints().gen_feat(k))for k in paths])


def gen_one_diag():
    return dict([(k.split('/')[1].lstrip('O3_'),gen_1_feats(k)) for k in paths])


def gen_bin():
    # FSO=From same orbital
    FSO=[]
    # Indexing
    featind=[]
    # Keys=paths in string format
    keys=[]
    for i in paths:
        k=str(i)
        keys.append(k)
        for ind,g in enumerate(pair_labels):
# Epq Ers
# # e_q+e_s-e_p-e_r
# TIUV
#'A'+g[0]+'_I'+g[1]+''+g[2]+''+g[3]
# Eti Euv (E01 E23)
# e_i+e_v-e_u-e_t
# e[0] + e[3] - e[1] - e[2]
            
            idx=g.split('_')
            featind.append(g)
            q=idx[0]
            s=idx[1]
# Since I!=A just append 0 since they'll never both come from the same orbital                
            if q==s:
                FSO.append(1)
            else:
                FSO.append(0)
    return dict([(z.split('/')[1].lstrip('O3_'),pd.DataFrame({'From_Same_Orbital':np.array(FSO).reshape(len(paths),-1)[idx]},index=np.array(featind).reshape(len(paths),-1)[idx])) for idx,z in enumerate(keys)])               




# In[ ]:





# In[ ]:


def Big_Data_GS():
    t0=time()
    with open('O3/fixed_feats.pickle', 'wb') as handle:
        pickle.dump(pd.concat([pd.concat({k: v for k,v in gen_bin().items()},axis=0),
                               pd.concat({k: v for k,v in gen_two_el().items()},axis=0),
                               pd.concat({k: v for k,v in gen_one_diag().items()},axis=0)],axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('O3/fixed_targets.pickle', 'wb') as handle:
        pickle.dump(stacked_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(time()-t0)


# In[ ]:





# In[ ]:


Big_Data_GS()


# In[ ]:


# pd.read_pickle(f'fixed_feats.pickle'),
# pd.read_pickle(f'typH_targets.pickle').plot()


# In[ ]:


for i in paths:
    shutil.rmtree(i)


# In[ ]:




