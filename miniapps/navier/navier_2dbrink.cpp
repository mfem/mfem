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

   u(0) = 0.2;
   u(1) = 0.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   //const char *mesh_file = "bar3d.msh";
   int run_id = 0;

   int serial_refinements = 1;

   double tForce_Magnitude = 1.0;


   OptionsParser args(argc, argv);
   // args.AddOption(&mesh_file, "-m", "--mesh",
   //                "Mesh file to use.");
   args.AddOption(&run_id, "-r", "--runID",
                  "Run ID.");

   args.AddOption(&tForce_Magnitude, "-fo", "--forceMag",
                  "Force Magnitude.");

   args.AddOption(&serial_refinements, "-ref", "--refinements",
                  "refinements.");

   args.Parse();

   bool LoadSolVecFromFile = false;
   enum DensityCoeff::PatternType tGeometry = DensityCoeff::PatternType::Ball;
   enum DensityCoeff::ProjectionType tProjectionType = DensityCoeff::ProjectionType::zero_one;
  
   double tLengthScale = 1.0e-1;
   double tThreshold = 0.25;
   double tDensity = 1.0e3;
   double tRefVelocity = 1.0e-3; 
   double tKinViscosity = 1.0e-6; 
   double ReynoldsNumber = tLengthScale * tRefVelocity / tKinViscosity;

   double tPoriosity = 0.1; 
   double tPermeability = 1.0e-3; 
   double tDa = tPoriosity / tPermeability; 
   double tBrinkmann = 10000; 

   s_NavierContext ctx;
   ctx.order = 2;
   ctx.kin_vis = 1.0 / ReynoldsNumber;
   ctx.t_final = 1.5;
   ctx.dt = 1e-4;

   //mesh->EnsureNCMesh(true);

   double MultInX = 1.0;
   double MultInY = 1.0;

   double Lx = 1.0 * MultInX;
   double Ly = 1.0 * MultInY;

   int NX = 64 * MultInX;
   int NY = 64 * MultInY;


   Mesh mesh = Mesh::MakeCartesian2D(NX, NY, Element::QUADRILATERAL, true,Lx, Ly);

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   // Create translation vectors defining the periodicity
   Vector x_translation({1.0*MultInX, 0.0});
   Vector y_translation({0.0, 1.0*MultInY});

   std::vector<Vector> translations = {x_translation, y_translation};

   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,
                                           mesh.CreatePeriodicVertexMapping(translations));

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
   //delete mesh;
   if (mpi.Root())
   {
      std::cout << "Mesh of elements: " << pmesh->GetNE() << std::endl;
   }

   if( true )
   {

      mfem::ParaViewDataCollection mPvdc("2D_Mesh", pmesh);
      mPvdc.SetCycle(0);
      mPvdc.SetTime(0.0);
      mPvdc.Save();
   }

   //----------------------------------------------------------


   double preasureGrad = 0.01 + 0.001 *(tForce_Magnitude - 1.0);

   tForce_Magnitude = preasureGrad * tLengthScale /(tDensity * std::pow(tRefVelocity ,2));
 
   // get random vals
   std::vector< double > tRand(5,0.0);
   if (mpi.Root())
   {
      srand(run_id+1);  

      //   tRand[0] = (rand() / double(RAND_MAX)) * 2.0 - 1.0; //nx
      //   tRand[1] = (rand() / double(RAND_MAX)) * 2.0 - 1.0; //ny
      //   tRand[2] = (rand() / double(RAND_MAX)) * 2.0 - 1.0; //nz
      //   tRand[3] = rand() / double(RAND_MAX) * 10;          //a
      //   tRand[4] = rand() / double(RAND_MAX);   //eta

        tRand[0] = 1.0; //nx
        tRand[1] = 0.0;// / sqrt( 3.0 ); //ny
        tRand[2] = 0.0;// / sqrt( 3.0 ); //nz
        tRand[3] = tForce_Magnitude;//150.7*5/10*1.5;//150.7*1.5;        //a
        tRand[4] = tThreshold;//0.65;  //0.4365
   }

   if (mpi.WorldSize() > 1 )
   {
      MPI_Bcast( tRand.data(), tRand.size(), MPI_DOUBLE , 0, MPI_COMM_WORLD );
   }

   std::cout<< tRand[0]<<" "<<tRand[1]<<" "<<tRand[2]<<" "<<tRand[4]<<" "<<tRand[4]<<" "<<std::endl;

   //----------------------------------------------------------
   {
      Navier3dBrinkWorkflow tWorkflow( mpi, pmesh, ctx );

      tWorkflow.SetParams( tRand[0],tRand[1],tRand[2],tRand[3],tRand[4] );


      tWorkflow.SetDensityCoeff( tGeometry, tProjectionType );

      std::cout<<"---density set---"<<std::endl;

      tWorkflow.SetupFlowSolver(  );
            std::cout<<"---solver set---"<<std::endl;

      tWorkflow.SetInitialConditions(  vel, LoadSolVecFromFile );
            std::cout<<"---ini cond set---"<<std::endl;

      tWorkflow.SetupOutput(  );
            std::cout<<"---output set---"<<std::endl;

      tWorkflow.Perform(  );
            std::cout<<"---perform---"<<std::endl;

      tWorkflow.Postprocess(  run_id );


   std::cout<<  "perform executed" << std::endl;
   }

   delete pmesh;

   return 0;
}
