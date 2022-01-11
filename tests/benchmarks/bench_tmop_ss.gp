#gnuplot tmop_ss.gp > tmop_ss.pdf
reset
set encoding iso_8859_15
set datafile separator '|'

set terminal png transparent size 3072,2048 truecolor enhanced fontscale 4
#set terminal pdf enhanced linewidth 1.0 color
set size 1,1

set logscale y

set style line 1 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#8E56A1' # violet
set style line 2 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#644B9E' # dark violet
set style line 3 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#3D81C2' # blue
set style line 4 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#46BEA2' # turquoise
set style line 5 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#68C03C' # green
set style line 6 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#FAE13E' # yellow
set style line 7 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#F58023' # orange
set style line 8 pt 7 ps 0.5 lt 1 lw 3 lc rgb '#E01B3F' # red

set style line 9 pt 7 ps 0.5 lt 1 lw 2 lc rgb '#A0A0A0' # gray

set logscale x 2
set xrange [3:192]
set xlabel 'Number of GPUs'
set xtics ("4" 4, "16" 16, "64" 64, "128" 128)

set ylabel 'Solve Time [seconds]'
#set ytics ("1 s." 1, "15 s." 15)
set yrange [0.4:20]

plot \
"tmop_ss.org" skip 1 using 2:(($3==0&&$4==0)?$5:NaN) ls 9 w l  title "Ideal strong scaling", \
"tmop_ss.org" skip 1 using 2:(($3==1&&$4==4)?$5:NaN) ls 1 w lp title "p=1, n=3M", \
"tmop_ss.org" skip 1 using 2:(($3==2&&$4==4)?$5:NaN) ls 3 w lp title "p=2, n=22M", \
"tmop_ss.org" skip 1 using 2:(($3==3&&$4==4)?$5:NaN) ls 5 w lp title "p=3, n=72M", \
"tmop_ss.org" skip 1 using 2:(($3==4&&$4==4)?$5:NaN) ls 7 w lp title "p=4, n=171M"