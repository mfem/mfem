#!/usr/bin/env bash

nodes=( 16 )
rs_levels=( 0 1 2 3 4 5 6 )
mesh_poly_deg=( 1 2 3 4 )  
quad_order=( 0 )
solver_type=( 0 )
solver_art_type=( 0 )
lin_solver=( 2 )
max_lin_iter=( 20 )
pa=( pa )
tasks_per_node=( 4 )
newton_iter=1
init_scale=1.e-7
meshes=( hex6.mesh hex1266.mesh hex12126.mesh )
partition=( 111 )

rm out_ss_06*.tmop
rm err_ss_06*.tmop
count=6000
for i9 in "${meshes[@]}"; do
  for i8 in "${pa[@]}"; do
    for i0 in "${tasks_per_node[@]}"; do
      for i1 in "${nodes[@]}"; do
        for i6 in "${lin_solver[@]}"; do
          for i2 in "${rs_levels[@]}"; do
            for i3 in "${mesh_poly_deg[@]}"; do
              for i4 in "${quad_order[@]}"; do
                for i5 in "${solver_type[@]}"; do
                  for i7 in "${max_lin_iter[@]}"; do
                    count=$(($count+1))
                    printf -v count2 "%05d" $count
                    echo $i1 $i2 $i3 $i4 $i5 $i6 $i7 $count2
                    jobfile="run_ss_tmop_"$count2".run"
                    np=$((i0*i1))

rm out_ss_$count2.tmop
rm err_ss_$count2.tmop

cat > $jobfile << EOF
#!/bin/bash
#BSUB -nnodes $i1
#BSUB -W 00:10
#BSUB -C 0
#BSUB -G asccasc
#BSUB -q pbatch
#BSUB -e err_ss_$count2.tmop
#BSUB -o out_ss_$count2.tmop
lrun -N$i1 -T$i0 -g1 pmesh-optimizer -m $i9 -mid 303 -tid 1 -vl 2 -bm -bnd -ni $newton_iter -art $solver_art_type -ls $i6 -qt 1 -$i8 -qo $i4 -li $i7 -o $i3 -st $i5 -rs $i2 -scale $init_scale -bm_id 1 -pt $partition -d cuda

EOF

bsub < $jobfile

done
done
done
done
done
done
done
done
done
done
