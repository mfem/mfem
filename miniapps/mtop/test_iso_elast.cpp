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

#include "mtop_solvers.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file =
      MFEM_SOURCE_DIR "/miniapps/mtop/data/canti_2D_6_quad.msh";
   const char *device_config = "cpu";
   int order = 2;
   bool pa = false;
   bool dfem = false;
   bool paraview = true;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&dfem, "-dfem", "--dFEM", "-no-dfem", "--no-dFEM",
                  "Enable or not dFEM.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or not Paraview visualization");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();
   MFEM_VERIFY(!(pa && dfem), "pa and dfem cannot be both set");

   //    Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   //    Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   //    Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      const int ref_levels =
         (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   }

   //    Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      const int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++) { pmesh.UniformRefinement(); }
   }

   // Create the solver
   IsoLinElasticSolver elsolver(&pmesh, order, pa, dfem);

   // set BC
   elsolver.AddDispBC(3, 4, 0.0);
   elsolver.AddDispBC(4, 4, 0.0);
   elsolver.AddDispBC(5, 4, 0.0);
   elsolver.AddDispBC(6, 4, 0.0);
   elsolver.AddDispBC(7, 0, -0.3);
   elsolver.AddDispBC(7, 1, 0.0);

   // delete all BC
   elsolver.DelDispBC();
   // set some of them again
   elsolver.AddDispBC(2, 4, 0.0);
   elsolver.AddDispBC(5, 4, 0.0);

   // set material properties
   ConstantCoefficient E(1.0), nu(0.2);
   elsolver.SetMaterial(E, nu);

   // set surface load
   elsolver.AddSurfLoad(1, 0.0, 1.0);

   // set convergence tolerances and max iterations
   elsolver.SetLinearSolver(1e-6,1e-8,100);

   // assemble the discrete system
   elsolver.Assemble();

   // solve the system
   elsolver.FSolve();

   // extract the solution
   ParGridFunction &sol = elsolver.GetDisplacements();

   if (paraview)
   {
      ParaViewDataCollection paraview_dc("isoel", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("disp", &sol);
      paraview_dc.Save();
   }

   if (socketstream glvis; visualization &&
       (glvis.open("localhost", 19916), glvis.is_open()))
   {
      glvis << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      glvis << "solution\n" << pmesh << sol << std::flush;
   }

   return EXIT_SUCCESS;
}
