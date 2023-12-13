#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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





# In[3]:


radius_range=np.linspace(0.7,3,100)
chains=np.arange(2,14,2)

# train_ind,test_ind=train_test_split(radius_range, test_size=0.3, random_state=0)
train_ind,test_ind=radius_range[0::2],radius_range[1::2]
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


# In[4]:


topdir=os.getcwd()
data_dirs=sorted([i for i in os.listdir() if 'chain' in i])


# In[5]:


def gen_energy():
    chain_E={}
    drop=[]
    for i in chains:
        energy=[]
        dirname=f'H{i}_chain'
        # Loop radius
        for idr,r in enumerate(radius_range):
            name=f"H{i}_{r:.2f}"
            try:
                output=os.path.join(dirname,f'{name}',f'{name}.output')
                energy.append([r,float((grep['-i', '::    CASPT2',output] | awk['{print $NF }'])())])
            except:
                energy.append([r,0])
                drop.append(idr)
        chain_E[i]=np.array(energy)

    for i in chains:
        chain_E[i]=chain_E[i][~np.in1d(range(len(chain_E[i])),drop)]    
        pd.DataFrame(chain_E[i],columns=['radius','energy']).to_csv(f'H{i}_chain/CASPT2.csv')
        

    casscf_chain_E={}
    for i in chains:
        casscf_energy=[]
        dirname=f'H{i}_chain'
        # Loop radius
        for idr,r in enumerate(radius_range):
            name=f"H{i}_{r:.2f}"
            try:
                output=os.path.join(dirname,f'{name}',f'{name}.output')
                casscf_energy.append([r,float((grep['-i', '::    RASSCF root number  1',output] | awk['{print $8 }'])())])
            except:
                casscf_energy.append([r,0])
        casscf_chain_E[i]=np.array(casscf_energy)

    for i in chains:
        casscf_chain_E[i]=casscf_chain_E[i][~np.in1d(range(len(casscf_chain_E[i])),drop)]    
        pd.DataFrame(casscf_chain_E[i],columns=['radius','energy']).to_csv(f'H{i}_chain/CASSCF.csv')        
        

    E2_chain_E={}
    for i in chains:
        E2_energy=[]
        dirname=f'H{i}_chain'
        # Loop radius
        for idr,r in enumerate(radius_range):
            name=f"H{i}_{r:.2f}"
            try:
                output=os.path.join(dirname,f'{name}',f'{name}.output')
                E2_energy.append([r,float((grep['-i', 'E2 (Variational):',output] | awk['{print $NF }'])())])
            except:
                E2_energy.append([r,0])
        E2_chain_E[i]=np.array(E2_energy)

    for i in chains:
        E2_chain_E[i]=E2_chain_E[i][~np.in1d(range(len(E2_chain_E[i])),drop)]    
        pd.DataFrame(E2_chain_E[i],columns=['radius','energy']).to_csv(f'H{i}_chain/E2.csv')        


# In[6]:


len(chains)


# In[7]:


cmap=sns.color_palette('rocket',len(chains))


# In[8]:


chains


# In[9]:


gen_energy()
fig,ax=plt.subplots(2,3,figsize=(10,6),sharex=True)
for idx,i in enumerate(chains):
    CASSCF=pd.read_csv(f"H{i}_chain/CASSCF.csv",index_col=0).to_numpy()
    CASPT2=pd.read_csv(f"H{i}_chain/CASPT2.csv",index_col=0).to_numpy()
    if 3<=idx:
        print(i)
        ax[0,idx%3].plot(CASSCF[:,0],CASSCF[:,1],color=cmap[idx],label='CASSCF')
        ax[0,idx%3].plot(CASPT2[:,0],CASPT2[:,1],'--',color=cmap[idx],label='CASPT2')
        ax[0,idx%3].legend()
        ax[0,idx%3].set_title(f"H$_{i}$")
        ax[0,idx%3].set_xlabel("Radius (Å)")
        ax[0,idx%3].set_ylabel("Energy (E$_{h}$)")
    else:
        ax[1,idx%3].plot(CASSCF[:,0],CASSCF[:,1],color=cmap[idx],label='CASSCF')
        ax[1,idx%3].plot(CASPT2[:,0],CASPT2[:,1],'--',color=cmap[idx],label='CASPT2')
        ax[1,idx%3].legend()
        ax[1,idx%3].set_title("H$_{"+str(i)+"}$")
        ax[1,idx%3].set_xlim(0.5,3.1)
        ax[1,idx%3].set_xticks(np.round(np.linspace(min(radius_range),max(radius_range),4),2))
        ax[1,idx%3].set_xlabel("Radius (Å)")
        ax[1,idx%3].set_ylabel("Energy (E$_{h}$)")        
    
plt.tight_layout()
plt.savefig('energies.png',dpi=300,bbox_inches='tight')
plt.show()


# In[10]:


for idx,i in enumerate(chains):
    CASSCF=pd.read_csv(f"H{i}_chain/CASSCF.csv",index_col=0).to_numpy()
    CASPT2=pd.read_csv(f"H{i}_chain/CASPT2.csv",index_col=0).to_numpy()
    plt.plot(CASSCF[:,0],CASSCF[:,1],color=cmap[idx],label=f'CASSCF H$_{i}$')
    plt.plot(CASPT2[:,0],CASPT2[:,1],'--',color=cmap[idx],label=f'CASPT2 H$_{i}$')    

# plt.legend()


# In[ ]:





# In[11]:


cwd = os.getcwd()
#   Keep everything at float64
DTYPE = np.float_
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





# In[12]:


# Generate the labels that match the IVECW and IVECC2 files
def gen_labels(path,typ):
    return [j.split()[0].replace('\n','').replace('00','').replace('S0','S').replace('I0','I').replace('A0','A').replace(',','') for j in pd.read_csv(f'{path}/GMJ_RHS_{typ}.csv',header=None)[0]]


def gen_pair_labels(path,typ):
    Labels=[]
    Indexes=[]
    return sorted(set(['_'.join(j.split()[0].replace('\n','').replace('00','').replace('S0','S').replace('I0','I').replace('A0','A').replace(',','').split('_')[0:2]) for j in open(f'{path}/GMJ_RHS_{typ}.csv','r').readlines()]))

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
    dims=[]
    for typ in typ_exists:
        dims.append((typ,np.array([i.split('=')[-1].split('x') for i in open(os.path.join(f'{path}',f'GMJ_e2_{typ}.csv'),'r').readlines() if 'mat. size =' in i ]).flatten().astype(int)))
    return dict(dims)


def strip(lst):   
    return '_'.join(i.replace('A00','A').replace('I00','I').replace('S00','S').replace('I0','I').replace('A0','A').replace('S0','S') for i in lst.split('_'))

def gen_ordered(path,typ):
    '''
    Return a dataframe for each type
    Index=proper indexing
    level_0=row
    level_1=column
    0=W value
    '''
    ordered=pd.read_csv(os.path.join(path,f'GMJ_IVECW_{typ}.csv'),delim_whitespace=True, skiprows=[0],header=None).astype(np.float64).dropna(axis=1)
    ordered.columns=list(range(len(ordered.columns)))
    ordered=ordered.stack()
    df=pd.read_csv(os.path.join(path,f'GMJ_RHS_{typ}.csv'),header=None,delimiter=',',index_col=0)
    df.index=list(map(strip,df.index))
    merged=ordered.reset_index().sort_values(by=0).set_index(df.sort_values(by=1).index).sort_values(['level_0','level_1'])    

    return merged

## Generate IVECW
def gen_e2(paths,typ):
    e2=[]
    
    for i in paths:
        proper_labels=gen_labels(i,typ)
        df=pd.read_csv(os.path.join(i,f'GMJ_e2_{typ}.csv'),delim_whitespace=True, skiprows=[0],header=None).astype(np.float64).dropna(axis=1).stack()
        df.index=gen_ordered(i,typ).index
        df=df.to_frame(name=str(i.split('/')[1].split('_')[1]))
        e2.append(df)
    df1=pd.concat(e2,axis=1).loc[proper_labels]
    df1.index=[i for idx,i in enumerate(proper_labels)]
    return df1

def gen_pair(paths,typ):
    Y=gen_e2(paths,typ).astype(float)
# Needs to be qs, we're summing over the occupied orbitals    
    Y_pair_set=list(set(['_'.join((i.split('_')[0],i.split('_')[1]))+'_' for i in Y.index.tolist()]))
    Y_pair_df=pd.concat([Y[Y.index.str.find(j)==0].sum() for j in Y_pair_set],axis=1)
    Y_pair_df.columns=list(set(['_'.join((i.split('_')[0],i.split('_')[1])) for i in Y.index.tolist()]))
    return Y_pair_df.T.sort_index().groupby(level=0).sum()


def stack_label(path,typ):
    if f'{typ}_M' in typ_exists and f'{typ}_P' in typ_exists:
        return gen_pair_labels(path,f'{typ}_P')+gen_pair_labels(path,f'{typ}_M')
    elif f'{typ}_P' in typ_exists:
        return gen_pair_labels(path,f'{typ}_P')
    elif f'{typ}_M' in typ_exists:
        return gen_pair_labels(path,f'{typ}_M')
    elif f'{typ}' in typ_exists:
        return gen_pair_labels(path,f'{typ}')
    
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
    
    
def gen_indx(list_of_dicts):
    indx=[]
    for i in list_of_dicts.keys():
        if len(list_of_dicts[i])>0:
            indx.append(list(list_of_dicts[i].keys()))
    return indx[0]    


# In[13]:


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
        
        self.train_4=np.array(Parallel(n_jobs=6,verbose=10)(delayed(self.big_4)(f'{i:.2f}') for i in train_ind))
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


# In[14]:


class gen_two_ints(object):
    """Some of the D_{ij}^{ab}=f_{ii}+f_{jj}-f_{aa}-f_{bb}=e_{ii}+e_{jj}-e_{aa}-e_{bb} elements 
    will be 0 since ij and ab overlap for CASPT2
    ij \in {I,A}
    ab \in {A,V}
    So ignore the warnings since they'll only be 0 when ijab \in {A} only"""
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
            new=np.sum(b[set_i_indices[idx],set_j_indices[idx]])#0
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





# In[15]:


def elements(x):
    '''
    Takes an integer, x, and returns the number of off-diagonal elements of an upper triangular matrix
    f(x)=(x*(x-1))/2
    '''
    return (x*(x-1))/2


# In[16]:


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


# In[17]:


def gen_two_el():
    return dict([(k,gen_two_ints().gen_feat(k))for k in paths])


def gen_one_diag():
    return dict([(k,gen_1_feats(k)) for k in paths])


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
        for ind,g in enumerate(full_set):
            # Epq Ers
            # # e_q+e_s-e_p-e_r
            # TIUV
            #'A'+g[0]+'_I'+g[1]+''+g[2]+''+g[3]
            # Eti Euv (E01 E23)
            # e_i+e_v-e_u-e_t
            # e[0] + e[3] - e[1] - e[2]
            featind.append(g)
            idx=g.split('_')
            q=idx[0]
            s=idx[1]
            # Since I!=A just append 0 since they'll never both come from the same orbital                
            if q==s:
                FSO.append(1)
            else:
                FSO.append(0)
    return dict([(z,pd.DataFrame({'From_Same_Orbital':np.array(FSO).reshape(len(paths),-1)[idx]},index=np.array(featind).reshape(len(paths),-1)[idx])) for idx,z in enumerate(keys)])               




# In[18]:


def Big_Data_GS(path):
    t0=time()
    with open(f'{path}/fixed_feats.pickle', 'wb') as handle:
        pickle.dump(pd.concat([pd.concat({k: v for k,v in gen_bin().items()},axis=0),
                               pd.concat({k: v for k,v in gen_two_el().items()},axis=0),
                               pd.concat({k: v for k,v in gen_one_diag().items()},axis=0)],axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}/fixed_targets.pickle', 'wb') as handle:
        pickle.dump(stacked_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{time()-t0:.4f}\n")


# In[19]:


chain_dirs=[i for i in sorted(os.listdir()) if os.path.isdir(i) and '.ipynb' not in i]


# In[20]:


chain_dirs


# In[21]:


# for chaindir in sorted(chain_dirs,key=lambda x: int(x.split('_')[0].replace('H',''))):
for chaindir in ['H10_chain']:
    print(chaindir)

    paths=sorted(glob(os.path.join(chaindir,chaindir.replace('chain','*'))))
    path=paths[0]
    print(path)
    path_check=glob(os.path.join(path,'*.output'))[0]

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



    #   Start transforming the HDF5 files from the data directory
    h5list = sorted(glob(f'{chaindir}/*/*rasscf.h5'))
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


    # SCF Features
    h5list_scf = sorted(glob(f'{chaindir}/*/*.scf.h5'))
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





    typ_exists=sorted(sum(list([j.replace('GMJ_e2_','') for j in i.split('/')[-1].split('.') if 'GMJ' in j] for i in glob(os.path.join(path,'GMJ_e2_*.csv'))),[]))


    dims_dict=gen_dim_dict(path,typ_exists)


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


    stacked_e2=pd.concat([gen_e2(paths,typ) for typ in typ_exists]).groupby(level=0).sum()
    E2Dict=pd.read_csv(f"{chaindir}/E2.csv",index_col=0).to_numpy()
    # stacked_e2.columns=[float(i.split('/')[1].split('_')[1]) for i in stacked_e2.columns]
    stacked_e2=stacked_e2.sum(axis=0).sort_index().reset_index().to_numpy()  


    stacked_pairs=pd.concat([stack_e2(paths,typ) for typ in typ_exists]).groupby(level=0).sum()
    pair_labels=stacked_pairs.index.tolist()

    dummy_stack=pd.concat([gen_e2(paths,typ) for typ in typ_exists])




    ijkl_idx=gen_big_4().gen_ijkl()
    set_i_indices=ijkl_idx[0]
    set_j_indices=ijkl_idx[1]
    set_k_indices=ijkl_idx[2]
    set_l_indices=ijkl_idx[3]
    set_indices=ijkl_idx[4]





    # Grab molecular orbital occupations and make it into a dataframe labeled with xyz file name
    MO_OCC=[]
    for j in range(len(chains)):
        MO_OCC.append(dict(zip(Basis_Indices,[i for i in list(MO_OCCUPATIONS[j])])))
    MO_OCC_Dict=dict(zip([str(k) for k in paths],MO_OCC))
    MO_OCC_DF=pd.DataFrame(MO_OCC_Dict)

    # Dataframe of MO occupation, index=basis indices and columns=paths
    MO_OCCUPATIONS_DF=pd.DataFrame(MO_OCCUPATIONS,index=paths,columns=Basis_Indices).transpose()



    # 
    # Keep in mind HDF5 zeroes out the actrive orbitals... we'll use the Fock matrix to recover these
    # 
    # Grab molecular orbital energy and make it into a dataframe labeled with xyz file name
    MO_ENERGIES=[]
    for j in paths:
        MO_ENERGIES.append(np.genfromtxt(os.path.join(j,f"{j.split('/')[1]}.GMJ_Fock_MO.csv"), delimiter=''))


    # Dataframe of MO energies, index=basis indices and columns=paths
    MO_ENERGIES_DF=pd.DataFrame(MO_ENERGIES,index=paths,columns=Basis_Indices).transpose()


    # Create slices for AA->AA indices
    # This will help zero out infinities
    internal_A=[idx for idx,i in enumerate(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=0].index) if 'A' in i]
    external_A=[idx for idx,i in enumerate(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=2].index) if 'A' in i]

    inner_slice=slice(min(internal_A),max(internal_A)+1)
    outer_slice=slice(min(external_A),max(external_A)+1)



    def gen_one_int():
        one_int=[]
        Labels=[]
        Indexes=[]
        upd_1int_indx=[]
        def one_gener(i):
            return pd.DataFrame(np.genfromtxt(os.path.join(i,f"{i.split('/')[1]}.GMJ_one_int.csv"), delimiter='',dtype=float),index=Basis_Indices,columns=Basis_Indices)


    #     Dict=dict(zip(Indexes,Labels))
        return dict((i,one_gener(i)) for i in paths)



    t0=time()
    int1=gen_one_int()
    print(f'Integrals loaded in {time()-t0:0.4f} s')




    nmo=len(Basis_Indices)
    indice=[]
    ad_ind=[]
    for ind,i in enumerate(range(nmo)):
        for indx,j in enumerate(range(nmo)):
            ad_ind.append(f'{i+1}_{j+1}')
            if j<=i:
                indice.append(f'{i+1}_{j+1}')


    import itertools

    def gen_MO(k):

        if os.path.exists(os.path.join(k,f"{k.split('/')[1]}.GMJ_two_int.csv"))==True:

            raw_MO=np.genfromtxt(os.path.join(k,f"{k.split('/')[1]}.GMJ_two_int.csv")).reshape(len(Basis_Indices),len(Basis_Indices),len(Basis_Indices),len(Basis_Indices))
        return raw_MO     



    froz=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('F')]
    inact=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('I')]
    act=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('A')]
    virt=[indx for indx,i in enumerate(Basis_Indices) if i.startswith('S')]



    # gen_F=dict(zip(paths,scf_F))
    # gen_occ=dict(zip(paths,scf_OCC))

    gen_F=MO_ENERGIES_DF
    gen_occ=MO_OCCUPATIONS_DF
    gen_F_SCF=pd.DataFrame(scf_F,columns=Basis_Indices,index=paths).T
    gen_occ_SCF=pd.DataFrame(scf_OCC,columns=Basis_Indices,index=paths).T


    full_set=sorted(set(sum([gen_pair_labels(path,typ) for typ in typ_exists],[])))

    # Rewrite the jacob style featurization step
    occcc=len(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=0])
    virttt=len(MO_OCC_DF.T.describe().loc['mean'][MO_OCC_DF.T.describe().loc['mean']!=2])


    argged=np.argsort(dummy_stack.abs().values,axis=0).T
    top_excits={}
    for idxc,c in enumerate(dummy_stack.columns):
        top_excits[c]=dummy_stack[c].iloc[argged[idxc]].iloc[-4:].index



    Big_Data_GS(f'{chaindir}')    


# In[22]:


# pd.read_pickle(f'fixed_feats.pickle'),
# pd.read_pickle(f'typH_targets.pickle').plot()


# In[ ]:




