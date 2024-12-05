# Script: plot_energies.gp

set terminal pngcairo size 800,600
set output "energies_plot.png"

set title "CASPT2 Total Energy vs X-Value"
set xlabel "X-Value"
set ylabel "Total Energy (a.u.)"

plot "plot_data.txt" using 1:2 with linespoints title "CASPT2 Energy"


