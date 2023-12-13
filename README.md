DDCASPT2
=============
Project Repo for DDCASPT2 development. All src code for my OpenMolcas build is held in https://github.com/ChemRacer/Test

-------------
**Note**
Note to self/Kostas: The molecular orbital energies associated with the rasscf.h5 zero out the active space elements. The diagonal elements of the MCSCF Fock matrix I print from sxctl.f are twice as large as the MO energies and zero out the secondary orbitals. To obtain the orbital energies from this matrix, multiply the diagonal by 0.5. These seems like double counting the MO energies/Fock matrix elements, but this gives a more reliable description of the electronic structure to the machine learning model.


**Features**


*CHANGE*
For each pair-energy, e_pqrs, the feature set includes elements from the molecular orbital vector, Fock matrix, and atomic orbital overlap matrix. Pair-energies are <img src="https://render.githubusercontent.com/render/math?math=V^{\dagger} \cdot C">, where C is in the solution array in covariant representation and V is in contravariant representative.


For each pair-energy, the features below are collected:
1. Binary features to denote if the excitations came from the same molecular orbital and if they are going to the same molecular orbital (an argument against this feature would be that this is a posteriori knowledge since this is related to the excitation energies indexing. My argument for using this is that given a training data set, we know which excitation types will be important for the total correlation. Also by definition, we’re putting the equilibrium geometry into the training set. This may not be ideal in the future, if we want to do geometry optimizations but for a proof of concept I think it helps with our performance elucidating the total PES. We aren’t trying to predict equilibrium geometries at this point but that may be a good next step)
2. Pseudo-PT2 style denominator <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{\left( F_{aa}-E_{0}  \right)} ">
3. MP2 style denominator <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{\left( \epsilon_{p} + \epsilon_{r} - \epsilon_{s} - \epsilon_{q} \right)} ">
4. Molecular orbital energies of orbitals p, q, r, s (rasscf.h5)
5. Diagonal elements of the molecular orbital Fock matrix associated with p,q,r,s (from sxctl.f->gmj.csv)
6. Molecular orbitals associated with the molecular orbital Fock matrix associated with p,q,r,s (from sxctl.f->gmj.csv)
7. Diagonal elements of the atomic orbital overlap matrix associated with p,q,r,s (rasscf.h5)
8. Coefficients of the diagonal elements of the molecular orbital vectors associated with p,q,r,s (rasscf.h5)
9. Off-diagonal elements of the molecular orbital Fock matrix associated with excitation pairs pr and qs (from sxctl.f->gmj.csv)
10. Molecular orbitals associated with the off-diagonal elements of the molecular orbital Fock matrix associated with excitation pairs pr and qs (from sxctl.f->gmj.csv)
11. Off-diagonal elements of the atomic orbital overlap matrix associated with excitation pairs pr and qs (rasscf.h5)
12. Coefficients of the off-diagonal elements of the molecular orbital vectors associated with with excitation pairs pr and qs (rasscf.h5)

