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

using namespace mfem;
using namespace navier;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   u(0) = 0.0;
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

   bool tPerturbed = false;
   double PerturbationSize = 0.001;
   bool LoadSolVecFromFile = false;
   enum DensityCoeff::PatternType tGeometry = DensityCoeff::PatternType::Gyroids;
   enum DensityCoeff::ProjectionType tProjectionType = DensityCoeff::ProjectionType::zero_one;

   double tLengthScale = 1.0;
   double tThreshold = 0.2;
   double tDensity = 1.0e3;
   double tRefVelocity = 0.01; 
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

   double MultInX = 0.02;
   double MultInY = 0.02;
   double MultInZ = 0.02;

   double Lx = 1.0 * MultInX;
   double Ly = 1.0 * MultInY;
   double Lz = 1.0 * MultInX;

   int NX = 64;
   int NY = 64;
   int NZ = 64;


   Mesh mesh = Mesh::MakeCartesian3D(NX, NY, NZ, Element::HEXAHEDRON, Lx, Ly, Lz);

   //Mesh *mesh = new Mesh("bar3d.msh");
   //Mesh *mesh = new Mesh("./cube.mesh");
   //mesh->EnsureNCMesh(true);

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   if (mpi.Root())
   {
      std::cout << "Mesh of elements: " << pmesh->GetNE() << std::endl;
   }
   
   //----------------------------------------------------------


   double preasureGrad = 0.01 + 0.001 *(tForce_Magnitude - 1.0);

   tForce_Magnitude = preasureGrad * tLengthScale /(tDensity * std::pow(tRefVelocity ,2));
 
   // get random vals
   std::vector< double > tRand(5,0.0);
   if (mpi.Root())
   {
      srand(run_id+1);  

      //double rForceMag = rand() / double(RAND_MAX) * 10.0;
      double rForceMag = 1.0;

      tRand[0] = ((rand() / double(RAND_MAX)) * 2.0 - 1.0)*rForceMag; //nx
      tRand[1] = ((rand() / double(RAND_MAX)) * 2.0 - 1.0)*rForceMag; //ny
      tRand[2] = ((rand() / double(RAND_MAX)) * 2.0 - 1.0)*rForceMag; //nz
      tRand[3] = 1.0;          //a
      tRand[4] = 0.008; //rand() / double(RAND_MAX)  * 0.007 + 0.002;   //eta  //0.006;
      //      tRand[4] = 0.5;   //eta 0.004;

      if( tPerturbed )
      {
         double tRand4 = ((rand() / double(RAND_MAX)) * 2.0 - 1.0);
         double tRand5 = ((rand() / double(RAND_MAX)) * 2.0 - 1.0);
         double tRand6 = ((rand() / double(RAND_MAX)) * 2.0 - 1.0);
         double norm = std::sqrt( tRand4* tRand4 + tRand5*tRand5+ tRand6*tRand6);
         tRand4 = tRand4 * PerturbationSize / norm;
         tRand5 = tRand5 * PerturbationSize / norm;
         tRand6 = tRand6 * PerturbationSize / norm;

         tRand[0] = tRand[0] + tRand4;
         tRand[1] = tRand[1] + tRand5;
         tRand[2] = tRand[2] + tRand6;
      }
   }

   if (mpi.WorldSize() > 1 )
   {
      MPI_Bcast( tRand.data(), tRand.size(), MPI_DOUBLE , 0, MPI_COMM_WORLD );
   }

   std::cout<< tRand[0]<<" "<<tRand[1]<<" "<<tRand[2]<<" "<<tRand[3]<<" "<<tRand[4]<<" "<<std::endl;

   //----------------------------------------------------------
   {
      Navier3dBrinkWorkflow tWorkflow( mpi, pmesh, ctx );

      tWorkflow.SetParams( tRand[0],tRand[1],tRand[2],tRand[3],tRand[4] );

      tWorkflow.SetDensityCoeff( tGeometry, tProjectionType );

      tWorkflow.SetupFlowSolver(  );

       tWorkflow.SetInitialConditions(  vel, LoadSolVecFromFile, tBrinkmann );

      tWorkflow.SetupOutput(  );

      tWorkflow.Perform(  );

      tWorkflow.Postprocess(  run_id );



   std::cout<<  "perform executed" << std::endl;
   }


   delete pmesh;

   return 0;
}

