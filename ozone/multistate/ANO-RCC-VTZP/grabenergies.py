#Import packages
from collections import Counter
from glob import glob
from math import sin, cos, pi
from plumbum.cmd import grep, awk
from shutil import copy
from subprocess import call, check_output
from time import time
from tqdm import tqdm
import csv
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import shutil
import sklearn
import subprocess
import sys

def write_energies(path,MSroots=3,CIROOT="3 3 1"):
    radius = float(os.path.dirname(i).split("_")[1])
    path_check = path 
    HF = float((grep['-i', '::    Total SCF energy',path_check] | awk['{print $NF }'])())
    if CIROOT is None and MSroots is None:
    
        E2 = float((grep['-i', 'E2 (Variational):',path_check] | awk['{print $NF }'])())
        CASSCF_E = float((grep['-i', '::    RASSCF root number  1',path_check] | awk['{print $8 }'])())
        CASPT2_E = float((grep['-i', '::    CASPT2',path_check] | awk['{print $NF }'])())
        pd.DataFrame.from_dict({"E2":E2,"CASSCF_E":CASSCF_E,"CASPT2_E":CASPT2_E},orient='index').rename(columns={0:name}).to_excel(os.path.join(path,f"{name}_energies.xlsx"))
    else:
        
        hf=float((grep['-i', '::    Total SCF energy',path_check] | awk['{print $NF }'])())
        corr=(grep['-i', 'E2 (Variational):',path_check] | awk['{print $NF }'])().strip().split('\n')
        rasscf=(grep['-i', '::    RASSCF',path_check] | awk['{print $NF }'])().strip().split('\n')
        caspt2=(grep['-i', '::    CASPT2',path_check] | awk['{print $NF }'])().strip().split('\n')
        
        df = pd.DataFrame(np.vstack([corr,rasscf,caspt2,MSroots*[hf]]).T.astype(float),index=[f"{i+1}" for i in range(MSroots)],columns=['E2','CASSCF_E','CASPT2_E','SCF_E']) #.to_excel(os.path.join(path,f"{name}_energies.xlsx"))

        df['radius'] = len(df) * [radius]
        df.reset_index(inplace=True)
        dfmelt = df.melt(id_vars=['radius','index'], value_vars=['CASSCF_E','CASPT2_E','SCF_E']).rename(columns={'index':'root'})
        return dfmelt             
dfenerges = []
for i in sorted(glob("*/*.output")):
    try:
        dfenerges.append(write_energies(i))
    except:
        dfenerges.append(None)

dfenergies = pd.concat(dfenerges)
dfenergies.to_csv("energies.csv")
print(dfenergies)
sns.lineplot(data=dfenergies,x='radius',y='value',hue='root',style='variable')
plt.savefig("energies.png",dpi=300,bbox_inches='tight')
