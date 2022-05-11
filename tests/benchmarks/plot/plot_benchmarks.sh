#!/usr/bin/env bash

solver_types="Full_CG Local_CG_Lobatto Local_CG_Legendre Direct Direct_cuSolver Direct_cuBLAS"
op_type="Setup Solve Setup_and_Solve"

for stype in $solver_types
do
   for op in $op_type
   do
      bench_name="${stype}_${op//_/}"
      csv_name="${bench_name,,}.csv"
      tex_name="${bench_name,,}.tex"

      title_name="${stype} ${op}"
      title_name=${title_name//_/ }

      cat << EOF > ${tex_name}
\def\RUNTITLE{${title_name}}
\def\DATA{${csv_name}}
\input{template.tex}
EOF

      latexmk -pdflatex="pdflatex -halt-on-error" -synctex=1 -outdir=build ${tex_name}
      cp build/${bench_name,,}.pdf ./
   done
done
