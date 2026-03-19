#!/bin/sh

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

run_step()
{
   dir=$1
   cmd=$2
   expect=$3

   printf '\nDirectory: %s\n' "$dir"
   printf 'Command: %s\n' "$cmd"
   printf 'Check/: %s\n' "$expect"
   printf 'Press Enter to start (Ctrl-C to stop)... '
   IFS= read -r _

   (
      cd "$ROOT/$dir"
      sh -c "$cmd"
   )
}

run_step \
   examples \
   "mpirun -np 4 ./ex1p -glvis ../../glvis/glvis" \
   "Expect one parallel GLVis window. Close it when done."

run_step \
   examples \
   "mpirun -np 4 ./ex5p -m ../data/square-disc.mesh -glvis ../../glvis/glvis" \
   "Expect velocity and pressure GLVis windows. Close both when done."

run_step \
   examples \
   "mpirun -np 4 ./ex6p -m ../data/amr-quad.mesh -glvis ../../glvis/glvis" \
   "Expect one persistent AMR GLVis window. Close it when done."

run_step \
   examples \
   "./ex9 -glvis ../../glvis/glvis" \
   "Expect one GLVis window that starts paused. Press space to start, then close it when done."

run_step \
   examples \
   "mpirun -np 4 ./ex9p -glvis ../../glvis/glvis" \
   "Expect one parallel GLVis window that starts paused. Press space to start, then close it when done."

run_step \
   examples \
   "mpirun -np 4 ./ex10p -glvis ../../glvis/glvis" \
   "Expect velocity and elastic-energy windows that start paused. Press space in each as needed, then close them."

run_step \
   examples \
   "./ex16 -glvis ../../glvis/glvis" \
   "Expect one GLVis window that starts paused. Press space to start, then close it when done."

run_step \
   examples \
   "mpirun -np 4 ./ex35p -glvis ../../glvis/glvis" \
   "Expect multiple GLVis windows, including a paused harmonic animation window. Press space there to start, then close all windows."

run_step \
   miniapps/electromagnetics \
   "mpirun -np 8 ./joule -m cylinder-hex.mesh -p rod -glvis ../../../glvis/glvis" \
   "Expect several GLVis field windows. Close them when done."

run_step \
   miniapps/tools \
   "./display-basis -glvis ../../../glvis/glvis" \
   "Expect several basis-function windows plus a terminal prompt. Use 'c' in the terminal menu to close windows and quit."

run_step \
   miniapps/tools \
   "mpirun -np 4 ./plor-transfer -glvis ../../../glvis/glvis" \
   "Expect multiple GLVis windows showing the transfer fields. Close them when done."
