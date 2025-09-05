for a in 1 2; do
    for nel in 8 11 16 23 32 45 64 92 128; do
        for o in 2 3 4; do
            for m in 1; do
                for pc in 0 1 2; do
                    for ir in 0 1; do
                        [[ $pc -eq 0 && $ir -ne 0 ]] && continue
                        filename=ho_meshes/d3_n1_a${a}_nel${nel}_o${o}_m${m}.mesh
                        ./nurbs_lor_solve -patcha -pa -m $filename -pc $pc -int $ir
                    done
                done
            done
        done
    done
done
