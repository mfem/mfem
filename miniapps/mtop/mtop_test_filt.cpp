// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Sample runs:
//    mpirun -np 4 mtop_test_iso_elasticity -tri -o 2
//
//    mpirun -np 4 mtop_test_iso_elasticity -quad -o 2
//
// Device sample runs:
//    mpirun -np 4 mtop_test_iso_elasticity -d gpu -quad -o 2

#include "mtop_solvers.hpp"

using namespace std;
using namespace mfem;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_tri.mesh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_quad.mesh";

class DensCoeff : public mfem::Coefficient
{
private:
    real_t l;
public:
    DensCoeff(real_t d=1.0) : l(d) {}

    virtual real_t Eval(mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip)
    {
        Vector x;
        T.Transform(ip, x);
        real_t r = x.Norml2();
        r=sin(M_PI*r/l);
        if(r>0.5)
            r=1.0;
        else
            r=0.0;
        return r;
    }
};



int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

    // Parse command-line options.
    const char *mesh_file = MESH_QUAD;
    const char *device_config = "cpu";
    int order = 2;
    bool mesh_tri = false;
    bool mesh_quad = false;
    int par_ref_levels = 1;
    bool paraview = false;
    bool visualization = true;
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
    args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
    args.AddOption(&mesh_tri, "-tri", "--triangular", "-no-tri",
                  "--no-triangular", "Enable or not triangular mesh.");
    args.AddOption(&mesh_quad, "-quad", "--quadrilateral", "-no-quad",
                  "--no-quadrilateral", "Enable or not quadrilateral mesh.");
    args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or not Paraview visualization");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
    args.ParseCheck();

    // Enable hardware devices such as GPUs, and programming models such as
    // CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    if (Mpi::Root()) { device.Print(); }

    // Read the (serial) mesh from the given mesh file on all processors.  We
    // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    // and volume meshes with the same code.
    Mesh mesh(mesh_tri ? MESH_TRI : mesh_quad ? MESH_QUAD : mesh_file, 1, 1);
    const int dim = mesh.Dimension();

    // Refine the serial mesh on all processors to increase the resolution. In
    // this example we do 'ref_levels' of uniform refinement. We choose
    // 'ref_levels' to be the largest number that gives a final mesh with no
    // more than 1000 elements.
    {
       const int ref_levels =
          (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
       for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
    }
    if (Mpi::Root())
    {
       std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
    }

    // Define a parallel mesh by a partitioning of the serial mesh. Refine
    // this mesh further in parallel to increase the resolution. Once the
    // parallel mesh is defined, the serial mesh can be deleted.
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    for (int l = 0; l < par_ref_levels; l++) { pmesh.UniformRefinement(); }


    PDEFilter* filt=new PDEFilter(&pmesh, 0.1, order);

    filt->Assemble();

    // define two grid functions: one for the input and one for the output
    ParGridFunction filt_gf(&filt->GetFilteredFESpace());
    ParGridFunction orig_gf(&filt->GetInputFESpace());
    QuadratureFunction orig_qf;
                    

    filt_gf.GetTrueVector()=0.0;

    DensCoeff dens_coeff(1.0);
    orig_gf.ProjectCoefficient(dens_coeff);

    filt->Mult(orig_gf.GetTrueVector(), filt_gf.GetTrueVector());
    filt_gf.SetFromTrueVector();



    if (paraview)
    {
        ParaViewDataCollection paraview_dc("isoel", &pmesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.RegisterField("filt", &filt_gf);
        paraview_dc.RegisterField("orig", &orig_gf);
        paraview_dc.Save();
    }


    delete filt;
    return EXIT_SUCCESS;
}

