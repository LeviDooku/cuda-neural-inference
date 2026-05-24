set terminal pngcairo size 1000,700
set output "graph/times_gpu.png"

set title "Tiempo de inferencia en GPU"
set xlabel "Número de muestras de inferencia"
set ylabel "Tiempo GPU (ms)"

set grid
set key left top

set logscale x
set format x "10^{%L}"

plot "data/results.dat" using 1:3 with linespoints title "GPU total", \
     "data/results.dat" using 1:4 with linespoints title "GPU kernel"