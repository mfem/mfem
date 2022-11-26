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
#include "Stokes.hpp"
#include <fstream>
#include <ctime>
#include <cstdlib> 

using namespace mfem;
using namespace stokes;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   u(0) = 1.0;
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

   bool LoadSolVecFromFile = true;
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

      // Create translation vectors defining the periodicity
   Vector x_translation({1.0*MultInX, 0.0});
   Vector y_translation({0.0, 1.0*MultInY});

   std::vector<Vector> translations = {x_translation, y_translation};

   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,
                                           mesh.CreatePeriodicVertexMapping(translations));

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);

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

   std::cout<< tRand[0]<<" "<<tRand[1]<<" "<<tRand[2]<<" "<<tRand[3]<<" "<<tRand[4]<<" "<<std::endl;
   // Allocate the nonlinear diffusion solver
   mfem::stokes::Stokes* solver=new mfem::stokes::Stokes(pmesh,2);


   solver-> SetParams(1.0, 0.0, 0.0, 1.0, 0.25 );
   solver->SetDensityCoeff(tGeometry,tProjectionType);

   //solve
   solver->FSolve();

   // average of vel field, remove and apply to advection

   return 0;
}
