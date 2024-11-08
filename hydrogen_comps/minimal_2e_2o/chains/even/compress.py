#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import pandas as pd
from joblib import Parallel,effective_n_jobs,delayed
from tqdm import tqdm


# In[2]:


def compress(c):
    if 'e2' in c or 'IVECW' in c:
        try:
            pd.read_csv(c).to_csv(c,compression='zip')
        except:
            print(f"{c} is done")
    else:
        try:
            pd.read_csv(c,header=None).to_csv(c,compression='zip')
        except:
            print(f"{c} is done")


# In[3]:


def decompress(c):
    if 'e2' in c or 'IVECW' in c:
        try:
            pd.read_csv(c,compression='zip').to_csv(c)
        except:
            print(f"{c} is done")
    else:
        try:
            pd.read_csv(c,header=None,compression='zip').to_csv(c)
        except:
            print(f"{c} is done")


# In[4]:


Parallel(n_jobs=16)(delayed(compress)(c) for c in tqdm(glob('*/*/*csv')))


# In[5]:


# Parallel(n_jobs=16)(delayed(decompress)(c) for c in tqdm(glob('*/*/*csv')))

