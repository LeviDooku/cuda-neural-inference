set terminal pngcairo size 1000,700
set output "graph/speedup.png"

set title "Speedup CPU/GPU"
set xlabel "Número de muestras de inferencia"
set ylabel "Speedup"

set grid
set key left top

set logscale x
set format x "10^{%L}"

plot "data/results.dat" using 1:($2/$3) with linespoints title "Speedup total = CPU / GPU total", \
     "data/results.dat" using 1:($2/$4) with linespoints title "Speedup kernel = CPU / GPU kernel"