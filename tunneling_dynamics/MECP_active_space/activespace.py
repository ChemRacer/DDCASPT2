# Suppress PySCF warning...
import pyscf
pyscf.__config__.B3LYP_WITH_VWN5 = False
from pathlib import Path

# The Mole class is used to define molecular information in PySCF.
from pyscf.gto import Mole

# logger contains definitions of verbosity levels for PySCF.
from pyscf.lib import logger

# Functionality for (state-averaged) CASSCF.
from pyscf.mcscf import CASSCF, state_average_mix
from pyscf.fci.direct_spin1 import FCISolver
from pyscf.fci.addons import fix_spin

# Wrapper functions to perform selection for variable and fixed active space sizes
from asf.wrapper import find_from_mol, find_from_scf, sized_space_from_mol, sized_space_from_scf

# Various utility functions...
from asf.utility import compare_active_spaces, show_mos_grid, pictures_Jmol


mol = Mole()
mol.atom = "anie202206314-sup-0001-cycl_mecp.xyz"
mol.basis = "anorcc"
mol.charge = 0
mol.spin = 2
# Set mol.verbose = logger.INFO to enable printing of SCF iterations and further output.
mol.verbose = logger.NOTE
mol.build()

active_space = find_from_mol(mol)

print(active_space)

