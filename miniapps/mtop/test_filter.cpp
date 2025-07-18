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

#include <iostream>

#include "mfem.hpp"
#include "mtop_solvers.hpp"

using namespace std;
using namespace mfem;

class DensCoeff:public Coefficient
{
public:
   DensCoeff(real_t ll=1):len(ll)
   {}

   virtual
   ~DensCoeff() {};

   virtual
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      real_t x[3];
      Vector transip(x, 3); transip=0.0;
      T.Transform(ip, transip);

      real_t l=transip.Norml2();
      return (sin(len*l)>real_t(0.0));
   }
private:
   real_t len;
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // 2. Parse command-line options.
   const char *mesh_file = "canti_2D_6.msh";
   int order = 2;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }


   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   //allocate the fiter
   double r=0.1;
   FilterOperator* filt=new FilterOperator(r,&pmesh);
   //set the boundary conditions
   filt->AddBC(1,1.0);
   filt->AddBC(2,1.0);
   filt->AddBC(3,1.0);
   filt->AddBC(4,1.0);
   filt->AddBC(5,1.0);
   filt->AddBC(6,1.0);
   filt->AddBC(7,1.0);
   filt->AddBC(8,0.0);
   //allocate the slover after setting the BC and before applying the filter
   filt->Assemble();

   ParGridFunction odens(filt->GetDesignFES());
   DensCoeff dc(M_PI);
   odens.ProjectCoefficient(dc);
   odens.SetTrueVector();
   const Vector& tdv=odens.GetTrueVector();

   ParGridFunction fdens(filt->GetFilterFES());
   Vector fdv(filt->GetFilterFES()->TrueVSize()); fdv=0.0;

   ParGridFunction cdens;
   filt->FFilter(&dc,cdens);

   filt->Mult(tdv,fdv);
   fdens.SetFromTrueDofs(fdv);

   {
      ParaViewDataCollection paraview_dc("filt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("odens",&odens);
      paraview_dc.RegisterField("fdens",&fdens);
      paraview_dc.RegisterField("cdens",&cdens);
      paraview_dc.Save();
   }

   delete filt;

   Mpi::Finalize();
   return 0;
}
