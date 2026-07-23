#!/bin/bash
grep -i '::    CASPT2' */*.output > energies.txt
python3 plot.py
gnuplot plot_energies.gp
