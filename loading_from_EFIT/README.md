# Loading EFIT data into MFEM

## Output EFIT to .gf file

```
cd loading_from_EFIT/EFIT_loading
python3 output_to_mfem.py
```

This will output the EFIT data to a .gf file. The .gf file can be loaded into MFEM using the following command:

```
cd ..
make mesh_maker && ./mesh_maker
make mesh_loader && ./mesh_loader
```