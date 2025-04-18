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

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "navier_solver_gcn.hpp"

namespace mfem {

class RampVectorCoefficient:public VectorCoefficient{
public:
    RampVectorCoefficient(VectorCoefficient& vc_, real_t t0_=0.0, real_t tc_=1.0):
        VectorCoefficient(vc_.GetVDim()),vc(&vc_),t0(t0_),tc(tc_)
    {}

    virtual
    void Eval(Vector &V, ElementTransformation &T,
                  const IntegrationPoint &ip)
    {
        V.SetSize(vc->GetVDim());
        real_t t=GetTime();

        if(t<t0){
            V=0.0;
            return;
        }

        real_t sc=1.0;
        if(t<tc){
            sc=t/(tc-t0);
        }

        vc->Eval(V,T,ip);
        V*=sc;
    }

private:
    VectorCoefficient* vc;
    real_t t0;
    real_t tc;
};

}



using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   mfem::Mpi::Init(argc, argv);
   int myrank = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   real_t newton_rel_tol = 1e-7;
   real_t newton_abs_tol = 1e-12;
   int newton_iter = 10;
   int print_level = 1;
   bool visualization = false;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   std::cout<<"My rank="<<pmesh.GetMyRank()<<std::endl;

   mfem::ConstantCoefficient* visc=new mfem::ConstantCoefficient(0.01);
   mfem::NavierSolverGCN* solver=new mfem::NavierSolverGCN(&pmesh,2,std::shared_ptr<mfem::Coefficient>(visc));


   mfem::Vector bcz(dim); bcz=0.0;
   mfem::Vector bci(dim); bci(0)=1.0;
   mfem::VectorConstantCoefficient vcz(bcz);
   mfem::VectorConstantCoefficient vci(bci);


   // set the BCs
   solver->AddVelocityBC(1,std::shared_ptr<mfem::VectorCoefficient>(new mfem::RampVectorCoefficient(vcz,0.0,1.0)));
   solver->AddVelocityBC(2,std::shared_ptr<mfem::VectorCoefficient>(new mfem::RampVectorCoefficient(vci,0.0,1.0)));
   solver->AddVelocityBC(4,std::shared_ptr<mfem::VectorCoefficient>(new mfem::RampVectorCoefficient(vci,0.0,1.0)));

   real_t dt=0.01;
   real_t time=0.0;
   for(int i=0;i<10;i++){
       solver->Step(time,dt,i);
   }






   delete solver;

   return 0;
}
