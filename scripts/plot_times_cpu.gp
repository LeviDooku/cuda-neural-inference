set terminal pngcairo size 1000,700
set output "graph/times_cpu.png"

set title "Tiempo de inferencia en CPU"
set xlabel "Número de muestras de inferencia"
set ylabel "Tiempo CPU (ms)"

set grid
set key left top

set logscale x
set format x "10^{%L}"

plot "data/results.dat" using 1:2 with linespoints title "CPU"