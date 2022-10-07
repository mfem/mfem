// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// 3D flow over a cylinder benchmark example

//#include "navier_solver.hpp"
#include "navier_3d_brink_workflow.hpp"
#include "MeshOptSolver.hpp"
#include <fstream>
#include <ctime>
#include <cstdlib> 
#include <unistd.h>

using namespace mfem;
using namespace navier;

s_NavierContext ctx;
 //ctx.order = 2;
 //ctx.kin_vis = 0.1;
// ctx.t_final = 1.0;
// ctx.dt = 1e-4;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   u(0) = 0.2;
   u(1) = 0.0;
   if(zi<=1e-8){
       if(t<1.0){ u(2) = t;}
       else{ u(2)=1.0; }
   }else{
       u(2)=0.0;
   }
   u(2)=0.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   const char *mesh_file = "bar3d.msh";
   int run_id = 0;

   int serial_refinements = 0;

   double tForce_Magnitude = 0.0;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&run_id, "-r", "--runID",
                  "Run ID.");

   args.AddOption(&tForce_Magnitude, "-fo", "--forceMag",
                  "Force Magnitude.");

   args.AddOption(&serial_refinements, "-ref", "--refinements",
                  "refinements.");

   args.Parse();



   //Mesh *mesh = new Mesh("bar3d.msh");
   //Mesh *mesh = new Mesh("./cube.mesh");
  //Mesh *mesh = new Mesh("./inline-hex.mesh");

  
   //mesh->EnsureNCMesh(true);

    double Lx = 4.0;
    double Ly = 4.0;

   int NX = 4;
   int NY = 4;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(NX, NY, mfem::Element::QUADRILATERAL, true, Lx, Ly);

    std::cout<<" mech build"<<std::endl;

// Mesh *mesh = &tmesh;
   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   //delete mesh;
   if (mpi.Root())
   {
      std::cout << "Mesh of elements: " << pmesh->GetNE() << std::endl;
   }

            const int dim = pmesh->Dimension();
         mfem::H1_FECollection FEColl(1, dim);
         mfem::ParFiniteElementSpace FEScalarSpace(pmesh, &FEColl, 1);
         mfem::ParFiniteElementSpace FEVectorSpace(pmesh, &FEColl, dim);
         pmesh->SetNodalFESpace(&FEVectorSpace);

         // Set the nodal grid function
         mfem::ParGridFunction x(&FEVectorSpace);
         pmesh->SetNodalGridFunction(&x);

   if(true)
   {
         // Mesh morphing based on level set field
         const int quadOrder = 8; // integration order
         const int targetId = 1;
         const int metricId =  2;
         const int maxSolvItr = 50;
         const bool moveBnd = false;
         const int verboLev = 2;
         const double surfaceFit = 1e1;

         mfem::MeshMorphingSolver  tmopMeshMorphing_( pmesh, quadOrder, targetId, metricId, maxSolvItr, moveBnd, verboLev, surfaceFit);

      std::vector<double> PlaneVar
    {
        //+0.00, +0.00, 0.75
        +2.0, +0.5, 1.0
    };

    // Define set of design variables
    int numOptVars = FEScalarSpace.GetTrueVSize();

    std::vector<double> iniOptVals(numOptVars);

    mfem::ParGridFunction gridfuncLSVal(&FEScalarSpace);

    mfem::Vector locationVector(dim);

    for ( int Ik = 0; Ik<numOptVars; Ik++)
    {
        pmesh->GetNode(Ik, &locationVector[0]);

        const double * pCoords(static_cast<const double*>(locationVector));

        iniOptVals[Ik] = pCoords[0] - PlaneVar[0];
        gridfuncLSVal[Ik]= iniOptVals[Ik];

        std::cout<<"iniOptVals["<<Ik<<"]"<< iniOptVals[Ik]<<std::endl;
    };


         tmopMeshMorphing_.Solve(gridfuncLSVal);

      if( true )
      {


         mfem::ParaViewDataCollection mPvdc("TMOP_test", pmesh);
         mPvdc.SetCycle(0);
         mPvdc.SetTime(0.0);
         //mPvdc.RegisterField("LevelSet", &gridfuncLSVal);
         mPvdc.Save();
      }
   }


   std::cout<<"------ done ---------------"<<std::endl;

     


   delete pmesh;

   return 0;
}
