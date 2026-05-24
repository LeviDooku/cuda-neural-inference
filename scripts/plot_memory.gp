set terminal pngcairo size 1000,700
set output "graph/memory.png"

set title "Consumo de memoria"
set xlabel "Número de muestras de inferencia"
set ylabel "Memoria (MB)"

set grid
set key left top

set logscale x
set format x "10^{%L}"

plot "data/results.dat" using 1:7 with linespoints title "Memoria"