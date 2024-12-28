#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
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




# In[ ]:


for path in tqdm(sorted(glob('cluster/*'))):
    rad = os.path.basename(path)
    if os.path.exists(os.path.join(path,rad+'.csv'))==False:
        print(path)
        DDCASPT2(path,'ANO-RCC-VDZP',rad,10,14,None,previous=None,symmetry=1,spin=4,UHF=True,charge=2,clean=False,n_jobs=12)(run=False)


# In[ ]:




