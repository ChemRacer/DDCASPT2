#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
# !{sys.executable} -m pip install --upgrade  xeus-python notebook
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
from tqdm import tqdm
import shutil
import random
import sklearn
from shutil import copy
import csv
import h5py as h5
import seaborn as sns; sns.set(style="ticks", color_codes=True)

from sklearn.model_selection import train_test_split


# In[ ]:





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


# 
# Dimension check for DDCASPT2: check the ordering of the pair-energies,
# this notation follows a mix of the papers and code.
# 
# A (IA->AA): \ TIUV \ E$_{ti}$ E$_{uv}$ \ pqrs=tiuv=0123 \    
# B_P (II->AA) (P): \ IJTU \ E$_{ti}$ E$_{uj}$ \ pqrs=tiuj=2031 \
# B_M (II->AA) (M): \ IJTU \ E$_{ti}$ E$_{uj}$ \ pqrs=tiuj=2031 \
# C (AA->VA): \ UVAT \ E$_{at}$ E$_{uv}$ \ pqrs=atuv=2301 \
# D (IA->VA/AV): \ IUAT/IUTA \ E$_{ai}$ E$_{tu}$/E$_{ti}$ E$_{au}$ \ pqrs=(a/t)i(t/a)u=2031 \
# E_P (II->AV) (P): \ IJAT \ E$_{ti}$ E$_{aj}$ \ pqrs=tiaj=3021 \
# E_M (II->AV) (M): \ IJAT \ E$_{ti}$ E$_{aj}$ \ pqrs=tiaj=3021 \
# F_P (AA->VV) (P): \ TUAB \ E$_{at}$ E$_{bu}$ \ pqrs=atbu=2031 \
# F_M (AA->VV) (M): \ TUAB \ E$_{at}$ E$_{bu}$ \ pqrs=atbu=2031 \
# G_P (IA->VV) (P): \ ITAB \ E$_{ai}$ E$_{bt}$ \ pqrs=aibt=2031 \
# G_M (IA->VV) (M): \ ITAB \ E$_{ai}$ E$_{bt}$ \ pqrs=aibt=2031 \
# H_P (II->VV) (P): \ IJAB \ E$_{ai}$ E$_{bj}$ \ pqrs=aibj=2031 \
# H_M (II->VV) (M): \ IJAB \ E$_{ai}$ E$_{bj}$ \ pqrs=aibj=2031 \
# 


# In[3]:


class DDCASPT2:
    def __init__(self,path,basis_set,name,electrons,occupied,inactive,scf_previous=None,casscf_previous=None,symmetry=1,spin=0,UHF=False,CIROOT=None,frozen=0,pt2maxiter=50,MSroots=None,charge=0,clean=False,n_jobs=None):
        '''
        Initialize
        '''
        self.path=path
        self.basis_set=basis_set
        self.name=name
        self.electrons=electrons
        self.occupied=occupied
        self.inactive=inactive
        self.scf_previous=scf_previous
        self.casscf_previous=casscf_previous
        self.symmetry=symmetry
        self.spin=spin      
        self.UHF=UHF
        self.CIROOT = CIROOT
        self.frozen=frozen
        self.pt2maxiter=pt2maxiter     
        self.MSroots=MSroots
        self.charge=charge
        self.clean=clean
        self.n_jobs=n_jobs
        


    def del_useless(self):
        '''
        Delete the extra files
        '''
        for root, dirs, files in os.walk(self.path):
            for file in files:
                for j in ['status','GssOrb','LprOrb','LoProp','guessorb','xmldump','RasOrb','SpdOrb','ScfOrb']:
                    if j in file:
                        os.remove(os.path.join(root,file))   
                        
        for i in glob("GMJ*csv")+glob("*GMJ*int*csv")+glob('*h5'):
            os.remove(i)


    def _gen_gateway(self):
        string=f'''&GATEWAY 
coord={f'{self.name}.xyz'}
Basis = {self.basis_set}
Group = nosymm
End of Input

'''
        return string
    
    def _gen_seward(self):
        string=f'''&SEWARD
End of Input

'''
        return string
    
    def _gen_motra(self):
        if self.frozen is not None:
            string=f'''&MOTRA
Frozen={self.frozen}
LUMORB
>>> COPY $WorkDir/GMJ_one_int_indx.csv $CurrDir/{self.name}.GMJ_one_int_indx.csv
>>> COPY $WorkDir/GMJ_one_int.csv $CurrDir/{self.name}.GMJ_one_int.csv
>>> COPY $WorkDir/GMJ_two_int_indx.csv $CurrDir/{self.name}.GMJ_two_int_indx.csv
>>> COPY $WorkDir/GMJ_two_int.csv $CurrDir/{self.name}.GMJ_two_int.csv

'''
        else:
            string=f'''&MOTRA
LUMORB
>>> COPY $WorkDir/GMJ_one_int_indx.csv $CurrDir/{self.name}.GMJ_one_int_indx.csv
>>> COPY $WorkDir/GMJ_one_int.csv $CurrDir/{self.name}.GMJ_one_int.csv
>>> COPY $WorkDir/GMJ_two_int_indx.csv $CurrDir/{self.name}.GMJ_two_int_indx.csv
>>> COPY $WorkDir/GMJ_two_int.csv $CurrDir/{self.name}.GMJ_two_int.csv

'''
        return string
    
    def _gen_scf(self):
        
        if self.UHF:
            string=f"""&SCF &END
UHF
charge
{self.charge}
spin
{self.spin + 1}            
"""            
        else:
            string=f"""&SCF &END
"""
        if self.scf_previous is not None:
            fileorb=f"""FileOrb
{self.scf_previous}
"""
        else:
            fileorb=''
            
        endstring=f""">>> COPY $WorkDir/{self.name}.scf.h5 $CurrDir/
        
"""
        return string+fileorb+endstring  
    
    
    def _gen_rasscf(self):
        start_string="""&RASSCF &END
Title= RASSCF
"""
        if self.casscf_previous is not None:
            fileorb=f"""FileOrb
{self.casscf_previous}
"""
        else:
            fileorb=''

        if self.CIROOT is None:
            ciroot=''
        else:
            ciroot=f"""CIROOT
{self.CIROOT}            
"""

        if self.inactive is None:
            end_string=f"""NACTEL
{self.electrons} 0 0
RAS2
{self.occupied}
Symmetry
{self.symmetry}
Spin
{self.spin + 1}
charge
{self.charge}
orblisting
all
ITERation
200 100
CIMX
200
SDAV
500
>>> COPY $WorkDir/{self.name}.rasscf.h5 $CurrDir/
>>> COPY $WorkDir/GMJ_Fock_MO.csv $CurrDir/{self.name}.GMJ_Fock_MO.csv

"""
        else:
            end_string=f"""NACTEL
{self.electrons} 0 0
Inactive
{self.inactive}
RAS2
{self.occupied}
Symmetry
{self.symmetry}
Spin
{self.spin + 1}
orblisting
all

>>> COPY $WorkDir/{self.name}.rasscf.h5 $CurrDir/
>>> COPY $WorkDir/GMJ_Fock_MO.csv $CurrDir/{self.name}.GMJ_Fock_MO.csv

"""
        return start_string+fileorb+ciroot+end_string 
    
    def _gen_caspt2(self):
        startstring="""&CASPT2 &END

Imaginary Shift
0.2
IPEA
0.25       
"""
        if self.frozen is None:
            frozstr=''
        else:
            frozstr=f"""
Frozen 
{self.frozen}         
"""
        if self.pt2maxiter is None:
            pt2maxiter=''
        else:
            pt2maxiter=f"""
MAXITER
{self.pt2maxiter}
"""
        if self.MSroots is None:
            caspt2data="""
>>foreach i in (B,E,F,G,H)
>>foreach j in (P,M)
>>if ( -FILE GMJ_e2_${i}_${j}_1_.csv )
>>> COPY $WorkDir/GMJ_RHS_${i}_${j}_1_.csv $CurrDir/GMJ_RHS_${i}_${j}_1_.csv
>>> COPY $WorkDir/GMJ_IVECW_${i}_${j}_1_.csv $CurrDir/GMJ_IVECW_${i}_${j}_1_.csv
>>> COPY $WorkDir/GMJ_IVECX_${i}_${j}_1_.csv $CurrDir/GMJ_IVECX_${i}_${j}_1_.csv
>>> COPY $WorkDir/GMJ_IVECC2_${i}_${j}_1_.csv $CurrDir/GMJ_IVECC2_${i}_${j}_1_.csv
>>> COPY $WorkDir/GMJ_e2_${i}_${j}_1_.csv $CurrDir/GMJ_e2_${i}_${j}_1_.csv
>>endif
>>enddo
>>enddo

>>foreach i in (A,C,D)
>>if ( -FILE GMJ_e2_$i_1_.csv )
>>> COPY $WorkDir/GMJ_RHS_$i_1_.csv $CurrDir/GMJ_RHS_$i_1_.csv
>>> COPY $WorkDir/GMJ_IVECW_$i_1_.csv $CurrDir/GMJ_IVECW_$i_1_.csv
>>> COPY $WorkDir/GMJ_IVECX_$i_1_.csv $CurrDir/GMJ_IVECX_$i_1_.csv
>>> COPY $WorkDir/GMJ_IVECC2_$i_1_.csv $CurrDir/GMJ_IVECC2_$i_1_.csv
>>> COPY $WorkDir/GMJ_e2_$i_1_.csv $CurrDir/GMJ_e2_$i_1_.csv
>>endif
>>enddo
"""
        else:

            looproots="("+','.join(str(i) for i in range(1,self.MSroots+1))+")"
            caspt2data=f"""
>>foreach k in {looproots}
>>foreach i in (B,E,F,G,H)
>>foreach j in (P,M)
"""+"""
>>if ( -FILE GMJ_e2_${i}_${j}_${k}_.csv )
>>> COPY $WorkDir/GMJ_RHS_${i}_${j}_${k}_.csv $CurrDir/GMJ_RHS_${i}_${j}_${k}_.csv
>>> COPY $WorkDir/GMJ_IVECW_${i}_${j}_${k}_.csv $CurrDir/GMJ_IVECW_${i}_${j}_${k}_.csv
>>> COPY $WorkDir/GMJ_IVECX_${i}_${j}_${k}_.csv $CurrDir/GMJ_IVECX_${i}_${j}_${k}_.csv
>>> COPY $WorkDir/GMJ_IVECC2_${i}_${j}_${k}_.csv $CurrDir/GMJ_IVECC2_${i}_${j}_${k}_.csv
>>> COPY $WorkDir/GMJ_e2_${i}_${j}_${k}_.csv $CurrDir/GMJ_e2_${i}_${j}_${k}_.csv
>>endif
>>enddo
>>enddo
>>enddo
"""+f"""
>>foreach j in {looproots}
>>foreach i in (A,C,D)
"""+"""
>>if ( -FILE GMJ_e2_${i}_${j}_.csv )
>>> COPY $WorkDir/GMJ_RHS_${i}_${j}_.csv $CurrDir/GMJ_RHS_${i}_${j}_.csv
>>> COPY $WorkDir/GMJ_IVECW_${i}_${j}_.csv $CurrDir/GMJ_IVECW_${i}_${j}_.csv
>>> COPY $WorkDir/GMJ_IVECX_${i}_${j}_.csv $CurrDir/GMJ_IVECX_${i}_${j}_.csv
>>> COPY $WorkDir/GMJ_IVECC2_${i}_${j}_.csv $CurrDir/GMJ_IVECC2_${i}_${j}_.csv
>>> COPY $WorkDir/GMJ_e2_${i}_${j}_.csv $CurrDir/GMJ_e2_${i}_${j}_.csv
>>endif
>>enddo
>>enddo
"""

        return startstring+frozstr+pt2maxiter+caspt2data    
        
    def write_input(self):
       # Write input
        with open(os.path.join(self.path,f'{self.name}.input'),'wb') as g:
            g.write(self._gen_gateway().encode())
            g.write(self._gen_seward().encode())
            g.write(self._gen_scf().encode())   
            g.write(self._gen_rasscf().encode())
            g.write(self._gen_motra().encode())
            g.write(self._gen_caspt2().encode())    

    def write_energies(self):
        # Grab energies
        self.path_check = os.path.join(self.path,f'{self.name}.output')
        self.HF = float((grep['-i', '::    Total SCF energy',self.path_check] | awk['{print $NF }'])())
        if self.CIROOT is None and self.MSroots is None:
        
            self.E2 = float((grep['-i', 'E2 (Variational):',self.path_check] | awk['{print $NF }'])())
            self.CASSCF_E = float((grep['-i', '::    RASSCF root number  1',self.path_check] | awk['{print $8 }'])())
            self.CASPT2_E = float((grep['-i', '::    CASPT2',self.path_check] | awk['{print $NF }'])())
            pd.DataFrame.from_dict({"E2":self.E2,"CASSCF_E":self.CASSCF_E,"CASPT2_E":self.CASPT2_E},orient='index').rename(columns={0:self.name}).to_excel(os.path.join(self.path,f"{self.name}_energies.xlsx"))
        else:
            
            hf=float((grep['-i', '::    Total SCF energy',self.path_check] | awk['{print $NF }'])())
            corr=(grep['-i', 'E2 (Variational):',self.path_check] | awk['{print $NF }'])().strip().split('\n')
            rasscf=(grep['-i', '::    RASSCF',self.path_check] | awk['{print $NF }'])().strip().split('\n')
            caspt2=(grep['-i', '::    CASPT2',self.path_check] | awk['{print $NF }'])().strip().split('\n')
            
            pd.DataFrame(np.vstack([corr,rasscf,caspt2,self.MSroots*[hf]]).T.astype(float),index=[f"root_{i+1}" for i in range(self.MSroots)],columns=['E2','CASSCF_E','CASPT2_E','SCF_E']).to_excel(os.path.join(self.path,f"{self.name}_energies.xlsx"))               

    def orbitals(self):
        #Grab basis information
        self.fro=int(subprocess.Popen(f"grep -i 'Frozen orbitals' {self.path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
        # Number of inactive orbitals
        self.inact=int(subprocess.Popen(f"grep -i 'Inactive orbitals' {self.path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
        # Number of active orbitals
        self.act=int(subprocess.Popen(f"grep -i 'Active orbitals' {self.path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
        # Number of seconary orbitals
        self.virt=int(subprocess.Popen(f"grep -i 'Secondary orbitals' {self.path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
        # Number of basis functions for sanity check
        self.bas_check=int(subprocess.Popen(f"grep -i 'Number of basis functions' {self.path_check} | tail -n 1",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].split()[-1])
        
        Basis_Indices=[]
        for i in range(self.fro):
            Basis_Indices.append(f'F{i+1}')
        for i in range(self.inact):
            Basis_Indices.append(f'I{i+1}')
        for i in range(self.act):
            Basis_Indices.append(f'A{i+1}')
        for i in range(self.virt):
            Basis_Indices.append(f'S{i+1}')   
        
        self.Basis_Indices = Basis_Indices
        self.basis_dict = {v:k for k,v in dict(enumerate(Basis_Indices)).items()}        

    def strip(self,lst):   
        '''
        Strips preceeding 0s in indexing files
        '''
        return '_'.join(re.sub(r'(?<!\d)0+(\d+)', r'\1', i) for i in lst.split('_'))



        
    def eightfold(self,i,j,k,l,integrals):
        '''
        Generate the permutational symmetries of the two-electron integrals 
        Parameters
        ----------
        idx: Int
            Index of the sorted two-electron excitations
        i: Int
            Index of orbital i
        j: Int
            Index of orbital j
        k: Int
            Index of orbital k
        l: Int
            Index of orbital l
        integrals: np.ndarray
            Integral array with format [i,j,k,l,value]
    
        Returns
        -------
        outint: np.ndarray
            Integral array with format [i,j,k,l,value]
        
        '''
        permutations = [[i,j,k,l], [j,i,l,k], [k,l,i,j], [l,k,j,i], [k,j,i,l], [l,i,j,k], [i,l,k,j], [j,k,l,i]]
        r_integrals=[]
        for p,q,r,s in permutations:
            ints = integrals[(integrals[:,0]==p)&(integrals[:,1]==q)&(integrals[:,2]==r)&(integrals[:,3]==s)]
            if len(ints)>0:
                r_integrals.append(ints)
        outint = np.unique(np.array(r_integrals).reshape(-1,5),axis=0).flatten()
        if len(outint)>0:
            return outint[-1]
        else:
            return 0
    
    
    def Coulomb(self,i,j,integrals):
        '''
        Generate the permutational symmetries of the Coulomb integrals 
    
        Parameters
        ----------
        i: Int
            Index of orbital i
        q: Int
            Index of orbital q
        integrals: np.ndarray
            Integral array with format [p,q,r,s,value]
    
        Returns
        -------
        eightfold: function
            help(eightfold)    
        '''    
        return self.eightfold(i,j,i,j,integrals)
    
    def Exchange(self,i,j,integrals):
        '''
        Generate the permutational symmetries of the Coulomb integrals 
    
        Parameters
        ----------
        i: Int
            Index of orbital i
        q: Int
            Index of orbital q
        integrals: np.ndarray
            Integral array with format [p,q,r,s,value]
    
        Returns
        -------
        eightfold: function
            help(eightfold)
        '''        
        return self.eightfold(i,j,j,i,integrals)
    
    
    def find_integrals(self,idx,p,q,r,s,integrals):
        '''
        Simplify returning the integrals
        
        cases:
        ijkl
        ijlk
        Jii
        Jjj
        Jkk
        Jll
        Jij
        Jkl
        Kij
        Kkl
    
        Parameters
        ----------
        idx: Int
            Index of the sorted two-electron excitations
        p: Int
            Index of orbital p
        q: Int
            Index of orbital q
        r: Int
            Index of orbital r
        s: Int
            Index of orbital s
        integrals: np.ndarray
            Integral array with format [p,q,r,s,value]
    
        Returns
        -------
        intdict: dict
            Dictionary with the integrals. Keys are the LaTeX formatted features names and the values should be the integrals
        '''
        intdict = {}
        # Find <pq|rs>
        intdict[r"$(\langle pq \vert rs \rangle)_{"+f"{idx}"+"}$"] = self.eightfold(p,q,r,s,integrals)
        # Find <pq|sr>
        intdict[r"$(\langle pq \vert sr \rangle)_{"+f"{idx}"+"}$"] = self.eightfold(p,q,r,s,integrals)
        
        # Find Jpp <pp|pp>
        intdict[r"$(\langle pp \vert pp \rangle)_{"+f"{idx}"+"}$"] = self.Coulomb(p,p,integrals)
        # Find Jqq <qq|qq>
        intdict[r"$(\langle qq \vert qq \rangle)_{"+f"{idx}"+"}$"] = self.Coulomb(q,q,integrals)
        # Find Jrr <rr|rr>
        intdict[r"$(\langle rr \vert rr \rangle)_{"+f"{idx}"+"}$"] = self.Coulomb(r,r,integrals)
        # Find Jss <ss|ss>
        intdict[r"$(\langle ss \vert ss \rangle)_{"+f"{idx}"+"}$"] = self.Coulomb(s,s,integrals)
        
        # Find Jpq <pq|pq>
        intdict[r"$(\langle pq \vert pq \rangle)_{"+f"{idx}"+"}$"] = self.Coulomb(p,q,integrals)
        # Find Kpq <pq|qp>
        intdict[r"$(\langle pq \vert qp \rangle)_{"+f"{idx}"+"}$"] = self.Exchange(p,q,integrals)
    
        # Find Jrs <rs|rs>
        intdict[r"$(\langle rs\vert rs \rangle)_{"+f"{idx}"+"}$"] = self.Coulomb(r,s,integrals)
        # Find Krs <rs|sr>
        intdict[r"$(\langle rs \vert sr \rangle)_{"+f"{idx}"+"}$"] = self.Exchange(r,s,integrals)
        return intdict    
        
    def parallel_feat(self,uniquepair):
        '''
        This is a helper function to create the features

        parameters
        ----------
        uniquepair: str
            Unique pair-energy label XX_YY
        '''
        q,s = uniquepair.split('_')
        qidx = self.basis_dict[q]
        sidx = self.basis_dict[s]
    
        # From same orbital = 1, else 0
        if q==s:
            self.binary_feat.append((uniquepair,1))
        else:
            self.binary_feat.append((uniquepair,0))
    
    
        # Get the pair-energies that share the same qs
        subpairs = self.pairs[self.pairs[:,3]==uniquepair]
    
        # Grab the largest 4 two-electron contributers
        best4 = subpairs[np.argsort(abs(subpairs[:,-1].astype(float)))][-4:]
    
        # Loop over best four 
        for b4idx, (typ, pq, rs) in enumerate(best4[:,0:3]):
            self.b4_type.append((uniquepair,f"typ_{b4idx}",self.typedict[typ]))
            p,q = pq.split('_')
            r,s = rs.split('_')
            pidx = self.basis_dict[p]
            qidx = self.basis_dict[q]
            ridx = self.basis_dict[r]
            sidx = self.basis_dict[s]
            self.two_el_feats.append((uniquepair,self.find_integrals(b4idx,pidx,qidx,ridx,sidx,self.twostacked)))
                
            
            # MO features for each index
            pMOdf,qMOdf,rMOdf,sMOdf =  self.MO_df.loc[p].to_frame(), self.MO_df.loc[q].to_frame(), self.MO_df.loc[r].to_frame(), self.MO_df.loc[s].to_frame()
            
            pMOdf.rename(index={'MO_ENERGIES_SCF':r'$(F_{p}^{\text{SCF}})_{'+f"{b4idx}"+"}$", 'MO_OCCUPATIONS_SCF':r"$(\omega_{p})_{"+f"{b4idx}"+"}$",'MO_ENERGIES':r"$(F_{p})_{"+f"{b4idx}"+"}$",'MO_OCCUPATIONS':r"$(\eta_{p})_{"+f"{b4idx}"+"}$"},inplace=True)
            qMOdf.rename(index={'MO_ENERGIES_SCF':r'$(F_{q}^{\text{SCF}})_{'+f"{b4idx}"+"}$", 'MO_OCCUPATIONS_SCF':r"$(\omega_{q})_{"+f"{b4idx}"+"}$",'MO_ENERGIES':r'$(F_{q})_{'+f"{b4idx}"+"}$",'MO_OCCUPATIONS':r"$(\eta_{q})_{"+f"{b4idx}"+"}$"},inplace=True)       
            rMOdf.rename(index={'MO_ENERGIES_SCF':r'$(F_{r}^{\text{SCF}})_{'+f"{b4idx}"+"}$", 'MO_OCCUPATIONS_SCF':r"$(\omega_{r})_{"+f"{b4idx}"+"}$",'MO_ENERGIES':r'$(F_{r})_{'+f"{b4idx}"+"}$",'MO_OCCUPATIONS':r"$(\eta_{r})_{"+f"{b4idx}"+"}$"},inplace=True) 
            sMOdf.rename(index={'MO_ENERGIES_SCF':r'$(F_{s}^{\text{SCF}})_{'+f"{b4idx}"+"}$", 'MO_OCCUPATIONS_SCF':r"$(\omega_{s})_{"+f"{b4idx}"+"}$",'MO_ENERGIES':r'$(F_{s})_{'+f"{b4idx}"+"}$",'MO_OCCUPATIONS':r"$(\eta_{s})_{"+f"{b4idx}"+"}$"},inplace=True)
    
            self.MO_feat.append((uniquepair,pMOdf,qMOdf,rMOdf,sMOdf))
            
            # Set of label index pairs
            pqrsindex_dict = {"p":[p,pidx],"q":[q,qidx],"r":[r,ridx],"s":[s,sidx]}
    
            # All possible two-index pairs
            twoidxpairs = [['p','q'],['r','s'],['p','r'],['q','s'],['p','p'],['q','q'],['r','r'],['s','s']]
            # h_{ij} features
            for u,v in twoidxpairs:
                u_item, u_idx = pqrsindex_dict[u]
                v_item, v_idx = pqrsindex_dict[v]
                if u_idx>=v_idx:
                    self.h_features.append((uniquepair,"h$_{"+f"{u}{v}"+"}^{"+f"{b4idx}"+"}$",self.h_stacked[(self.oneelint_idx[:,0]==u_idx)&(self.oneelint_idx[:,1]==v_idx)].flatten()[-1]))     
                else:
                    self.h_features.append((uniquepair,"h$_{"+f"{u}{v}"+"}^{"+f"{b4idx}"+"}$",self.h_stacked[(self.oneelint_idx[:,0]==v_idx)&(self.oneelint_idx[:,1]==u_idx)].flatten()[-1]))
                
    
        
        # Pair-energies
        pairenergy = np.sum(subpairs[:,-1].astype(float))
        self.pairenergylist.append((uniquepair,pairenergy))
        self.checkE2 += pairenergy


        return self.checkE2, self.h_features, self.b4_type, self.binary_feat, self.MO_feat, self.two_el_feats, self.pairenergylist  
    

    def gen_df(self,root):
        '''
        Generate feature dataframe
        '''        
        # binary feature df
        bindf = pd.DataFrame(self.binary_feat).set_index(0).rename(columns={0:'binary'})
        
        # one-electron dataframe
        h_df = pd.DataFrame(self.h_features).pivot(index=0, columns=1)
        h_df.columns=h_df.columns.droplevel()
        h_df.drop(columns=["h$_{qs}^{3}$","h$_{qs}^{1}$","h$_{qs}^{2}$"],inplace=True)
        h_df.rename(columns={"h$_{qs}^{0}$":"h$_{qs}$"},inplace=True)
        
        # Important 4 types
        important2e = pd.DataFrame(self.b4_type).pivot(index=0, columns=1)
        important2e.columns=important2e.columns.droplevel()
        
        
        # two-electron data frame 
        two_el_df = pd.concat([pd.concat([pd.DataFrame.from_dict(i[1],orient='index').rename(columns={0:i[0]}) for i in self.two_el_feats if i[0]==j]) for j in self.uniquepairs],axis=1).T
        
        
        listconcatmo = []
        
        for i in self.MO_feat:
        
            concatMO = pd.concat([j.rename(columns={j.columns[0]:i[0]}) for idx,j in enumerate(i) if idx>0])
            # print(concatMO)
            listconcatmo.append(concatMO)

        allMO_feats = pd.concat([pd.concat([i for i in listconcatmo if i.columns[0]==j]) for j in self.uniquepairs],axis=1).T

        pairenergy_df = pd.DataFrame(self.pairenergylist,columns=['index','Pair_Energies']).set_index('index').astype({'Pair_Energies':float})
        # Everything together so far
        concatdf = pd.concat([h_df,important2e,bindf,allMO_feats,two_el_df,pairenergy_df],axis=1)
        concatdf.to_csv(os.path.join(self.path,f"{self.name}_{root}.csv"),compression='zip') 

    def gen_pairs(self,i,root):
        '''
        Generate pairs in a parallel manner
        '''
        pairs = []
        typ = os.path.basename(i).split('.')[0].replace('GMJ_e2_','').replace(f'_{root}_','')

        IVEC = pd.read_csv(os.path.join(self.path,f'GMJ_IVECW_{typ}_{root}_.csv'),sep='\s+',header=None,skiprows=[0])
        RHS = pd.read_csv(os.path.join(self.path,f'GMJ_RHS_{typ}_{root}_.csv'),sep=',',header=None,index_col=0)
        RHS.index = list(map(self.strip,RHS.index))
        RHS = np.array(RHS.index).reshape(IVEC.shape)
        e2 = np.genfromtxt(os.path.join(self.path,f'GMJ_e2_{typ}_{root}_.csv'),skip_header=True).reshape(RHS.shape)
        IVECX = pd.read_csv(os.path.join(self.path,f'GMJ_IVECX_{typ}_{root}_.csv'),sep='\s+',header=None,skiprows=[0])

        IVECC2 = pd.read_csv(os.path.join(self.path,f'GMJ_IVECC2_{typ}_{root}_.csv'),sep='\s+',header=None,skiprows=[0])    
        for idxi,i in enumerate(RHS):
            for idxj,j in enumerate(i):
                # Split the index and enforce a standardization of p,q,r,s 
                split_index = j.split('_')
                type_idx = self.index_dict[typ]
                p,q,r,s = split_index[type_idx['p']],split_index[type_idx['q']],split_index[type_idx['r']],split_index[type_idx['s']]
                # typ, pq,rs,qs,e2
                pairs.append((typ,'_'.join((p,q)),'_'.join((r,s)),'_'.join((q,s)),e2[idxi,idxj])) 
        self.pairs = np.array(pairs)
        return self.pairs
        
    def gen_feats(self):
        '''
        Generate features
        '''
        self.orbitals()
        
        
        # Read CASSCF Fock from file
        CASSCF_fock = np.fromfile(os.path.join(self.path,f"{self.name}.GMJ_Fock_MO.csv"))
        
        # Load one-electron integrals
        oneelint = np.fromfile(os.path.join(self.path,f"{self.name}.GMJ_one_int.csv")).reshape(-1,1)
        self.oneelint_idx = np.fromfile(os.path.join(self.path,f"{self.name}.GMJ_one_int_indx.csv"),dtype=int).reshape(-1,4)[:,0:2]-1
        self.h_stacked = np.hstack([self.oneelint_idx,oneelint])
        
        # Load two-electron integrals (they're in physicist notation by default!) ijkl are indeed <ik|jl>
        twoelint = np.fromfile(os.path.join(self.path,f"{self.name}.GMJ_two_int.csv")).reshape(-1,1)
        twoelint_idx_chemist = np.fromfile(os.path.join(self.path,f"{self.name}.GMJ_two_int_indx.csv"),dtype=int).reshape(-1,4)-1
        
        twoelint_idx_physicist = twoelint_idx_chemist.copy()
        # <ij|kl>
        twoelint_idx_physicist[:,1] = twoelint_idx_chemist[:,2]
        twoelint_idx_physicist[:,2] = twoelint_idx_chemist[:,1]
        
        self.twostacked = np.hstack([twoelint_idx_physicist,twoelint])
        
        # Grab rasscf and scf hdf5 data
        rasscf_h5 = h5.File(os.path.join(self.path,f"{self.name}.rasscf.h5"), 'r')
        scf_h5 = h5.File(os.path.join(self.path,f"{self.name}.scf.h5"), 'r')
        
        datasetNames = [n for n in rasscf_h5.keys()]
        NBAS = rasscf_h5.attrs['NBAS']
        NACTEL = rasscf_h5.attrs['NACTEL']
        
        #Keys: MO_VECTORS, MO_ENERGIES, MO_OCCUPATIONS
        casMO_dict = {k:np.array(rasscf_h5[k]) for k in datasetNames if "MO_E" in k or "MO_O" in k}
        scfMO_dict = {k:np.array(scf_h5[k]) for k in datasetNames if "MO_E" in k or "MO_O" in k}
        
        
        # MO features made easy!
        MO_df = pd.DataFrame.from_dict(scfMO_dict).rename(columns={"MO_ENERGIES":"MO_ENERGIES_SCF","MO_OCCUPATIONS":"MO_OCCUPATIONS_SCF"})
        MO_df['MO_ENERGIES']=CASSCF_fock
        MO_df['MO_OCCUPATIONS']=casMO_dict['MO_OCCUPATIONS']
        # MO_df = MO_df.reset_index()
        MO_df.index = self.basis_dict.keys()
        self.MO_df = MO_df
        
        # Get two-electron indices
        
        two_el_ex_labels = {i.split('.')[0].replace("GMJ_RHS_",""):[re.sub(r'(?<!\d)0+(\d+)', r'\1', j) for j in pd.read_csv(i,header=None)[0].values] for i in glob(os.path.join(self.path,"GMJ_RHS_*.csv"))}
        
        pair_labels = {i.split('.')[0].replace("GMJ_RHS_",""):['_'.join(re.sub(r'(?<!\d)0+(\d+)', r'\1', j).split('_')[0:2]) for j in pd.read_csv(i,header=None)[0].values] for i in glob(os.path.join(self.path,"GMJ_RHS_*.csv"))}
        
        # CASPT2 E_pq E_rs ordering
        self.index_dict = {"A":{"p":0,"q":1,"r":2,"s":3},
        "B_P":{"p":2,"q":0,"r":3,"s":1},
        "B_M":{"p":2,"q":0,"r":3,"s":1},
        "C":{"p":2,"q":3,"r":0,"s":1},
        "D":{"p":2,"q":0,"r":3,"s":1},
        "E_P":{"p":3,"q":0,"r":2,"s":1},
        "E_M":{"p":3,"q":0,"r":2,"s":1},
        "F_P":{"p":2,"q":0,"r":3,"s":1},
        "F_M":{"p":2,"q":0,"r":3,"s":1},
        "G_P":{"p":2,"q":0,"r":3,"s":1},
        "G_M":{"p":2,"q":0,"r":3,"s":1},
        "H_P":{"p":2,"q":0,"r":3,"s":1},
        "H_M":{"p":2,"q":0,"r":3,"s":1}}
        
        self.typedict = {v:k for k,v in dict(enumerate(["A", "B_P", "B_M", "C", "D", "E_P", "E_M", "F_P", "F_M", "G_P", "G_M", "H_P", "H_M"])).items()}
        
        
        # Add MS-CASPT2 support below
        # IVECW and IRHS should have same indices
        # Same as IVECC2, it should all be element wise
        if self.MSroots  is None:
            msroots_range = [1]
        else:
            msroots_range = range(1, self.MSroots + 1)
        
        for msr in tqdm(msroots_range,desc="Root"):
            if self.n_jobs is None:
                self.pairs = np.vstack([self.gen_pairs(i,msr) for i in tqdm(glob(os.path.join(self.path,f"GMJ_e2_*_{msr}_.csv")),desc="Pairs")])    
            else:
                self.pairs = np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(self.gen_pairs)(i,msr) for i in tqdm(glob(os.path.join(self.path,f"GMJ_e2_*_{msr}_.csv")),desc="Pairs")))  
            
            # qs pairs!
            uniquepairs = np.unique(self.pairs[:,3])
            self.uniquepairs = uniquepairs
            self.checkE2=0
    
            
            self.h_features = []
            self.b4_type = []
            self.binary_feat = []
            self.MO_feat = []
            self.two_el_feats = []
            self.pairenergylist = []
            
            if self.n_jobs is None:
                out = []
                for i in tqdm(self.uniquepairs,desc="Features"):
                    outpar = self.parallel_feat(i)
                
            else:
                outpar=Parallel(n_jobs=self.n_jobs)(delayed(self.parallel_feat)(i) for i in tqdm(self.uniquepairs,desc="Features"))
                for i in outpar:
                    self.checkE2 += i[0]
                    self.h_features.append(i[1])
                    self.b4_type.append(i[2])
                    self.binary_feat.append(i[3])
                    self.MO_feat.append(i[4])
                    self.two_el_feats.append(i[5])
                    self.pairenergylist.append(i[6])
    
      
                self.h_features = sum(self.h_features,[])
                self.b4_type = sum(self.b4_type,[])
                self.binary_feat = sum(self.binary_feat,[])
                self.MO_feat = sum(self.MO_feat,[])
                self.two_el_feats = sum(self.two_el_feats,[])
                self.pairenergylist = sum(self.pairenergylist,[])
            
    
            
            self.gen_df(msr)
        
    
    def __call__(self,inputwrite=True,run=True,feat=True):
        '''
        Create input, run file, write energies to file, generate feature data, and clean up
        '''
        if inputwrite==True:
            self.write_input()
        
        top = os.getcwd()
        os.chdir(self.path)
        
        if run==True:
            print(f"Running on {self.n_jobs} cores")
            print(f"Found a valid MOLCAS installation at {os.environ['MOLCAS']}")
            print(f"MOLCAS_WORKDIR is set to {os.environ['MOLCAS_WORKDIR']}")
            
            call(['pymolcas','-new','-clean',os.path.join(self.path,f'{self.name}.input'), '-oe', os.path.join(self.path,f'{self.name}.output')])

        if feat==True:
            self.write_energies()
            self.gen_feats()
        
        if self.clean:
            self.del_useless()
        os.chdir(top)



# In[4]:


# for i in glob("GMJ*csv")+glob("*GMJ*int*csv")+glob('*h5'):
#     os.remove(i)

# path='./'
# basis_set='ANO-RCC-VDZP'
# name="H2"
# electrons=2
# occupied=2
# inactive=None


# DDCASPT2(path,basis_set,name,electrons,occupied,inactive,scf_previous=None,casscf_previous=None,symmetry=1,spin=0,UHF=False,CIROOT="3 3 1",frozen=0,pt2maxiter=50,MSroots=3,charge=0,clean=False,n_jobs=-1)(run=True)


# In[ ]:




