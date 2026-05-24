set terminal pngcairo size 1000,700
set output "graph/times_comparison.png"

set title "Comparación de tiempos: CPU vs GPU total vs GPU kernel"
set xlabel "Número de muestras de inferencia"
set ylabel "Tiempo (ms)"

set grid
set key left top

set logscale x
set logscale y
set format x "10^{%L}"
set format y "10^{%L}"

plot "data/results.dat" using 1:2 with linespoints title "CPU", \
     "data/results.dat" using 1:3 with linespoints title "GPU total", \
     "data/results.dat" using 1:4 with linespoints title "GPU kernel"