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
#include "advection_diffusion_solver.hpp"
#include "ascii.hpp"
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

   u(0) = 0.0;
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

   bool IsSolveAdvDiffProblem = true;
   bool tPerturbed = false;
   double PerturbationSize = 0.001;
   bool LoadSolVecFromFile = false;
   enum DensityCoeff::PatternType tGeometry = DensityCoeff::PatternType::Ball;   //Triangle
   enum DensityCoeff::ProjectionType tProjectionType = DensityCoeff::ProjectionType::zero_one;
  
   double tLengthScale = 0.02;
   double tThreshold = 0.25;
   double tDensity = 1.0e3;
   double tRefVelocity = 1.0; 
   double tKinViscosity = 1.0e-6; 
   double ReynoldsNumber = tLengthScale * tRefVelocity / tKinViscosity;

   double tPoriosity = 0.1; 
   double tPermeability = 1.0e-3; 
   double tDa = tPoriosity / tPermeability; 
   double tBrinkmann = 10000; 

   s_NavierContext ctx;
   ctx.order = 2;
   ctx.kin_vis = 1.0 / ReynoldsNumber;
   ctx.t_final = 1e-0;
   ctx.dt = 1e-4;

   //mesh->EnsureNCMesh(true);

   double MultInX = 0.02;
   double MultInY = 0.02;

   double Lx = 1.0 * MultInX;
   double Ly = 1.0 * MultInY;

   int NX = 64;
   int NY = 64;


   Mesh mesh = Mesh::MakeCartesian2D(NX, NY, Element::QUADRILATERAL, true,Lx, Ly);

   int dim = mesh.Dimension();

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

      //double rForceMag = rand() / double(RAND_MAX) * 10.0;
      double rForceMag = 1.0;//100.0;

      tRand[0] = ((rand() / double(RAND_MAX)) * 2.0 - 1.0)*rForceMag; //nx
      tRand[1] = ((rand() / double(RAND_MAX)) * 2.0 - 1.0)*rForceMag; //ny
      tRand[2] = 0.0;//M_PI/2.0; //(rand() / double(RAND_MAX))*M_PI*2.0/3.0; //nz
      tRand[3] = 1.0;          //a
      tRand[4] = rand() / double(RAND_MAX)  * 0.007 + 0.002;   //eta  //0.008;
      //      tRand[4] = 0.5;   //eta 0.004;

      if( tPerturbed )
      {
         double tRand4 = ((rand() / double(RAND_MAX)) * 2.0 - 1.0)*rForceMag;
         double tRand5 = ((rand() / double(RAND_MAX)) * 2.0 - 1.0)*rForceMag;
         double norm = std::sqrt( tRand4* tRand4 + tRand5*tRand5);
         tRand4 = tRand4 * PerturbationSize / norm;
         tRand5 = tRand5 * PerturbationSize / norm;

         tRand[0] = tRand[0] + tRand4;
         tRand[1] = tRand[1] + tRand5;
      }

      //   tRand[0] = 1.0; //nx
      //   tRand[1] = 0.0;// / sqrt( 3.0 ); //ny
      //   tRand[2] = 0.0;// / sqrt( 3.0 ); //nz
      //   tRand[3] = tForce_Magnitude;//150.7*5/10*1.5;//150.7*1.5;        //a
      //   tRand[4] = tThreshold;//0.65;  //0.4365
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

      tWorkflow.SetInitialConditions(  vel, LoadSolVecFromFile );

      tWorkflow.SetupOutput(  );

      tWorkflow.Perform(  );
      std::cout<<"---perform---"<<std::endl;

      tWorkflow.Postprocess(  run_id );


      mfem::Vector Flux_x;
      mfem::Vector Flux_y;

      if( IsSolveAdvDiffProblem)
      {
         const mfem::ParGridFunction * tVelocity = tWorkflow.GetVel();
         mfem::VectorGridFunctionCoefficient VelCoeff(tVelocity);
         const mfem::Vector & tAvgVel = tWorkflow.GetAvgGVel();

         mfem::VectorConstantCoefficient AvgVelCoeff(tAvgVel);
         mfem::VectorSumCoefficient OzillVelCoeff(VelCoeff, AvgVelCoeff, 1.0, -1.0);

         navier::DensityCoeff * mDensCoeff = new navier::DensityCoeff;

         mDensCoeff->SetThreshold(tRand[4]);
         mDensCoeff->SetPatternType(tGeometry);
         mDensCoeff->SetProjectionType(tProjectionType);

         mfem::ParGridFunction tOszillVelGF(*tVelocity);
         tOszillVelGF = 0.0;
         tOszillVelGF.ProjectCoefficient(OzillVelCoeff);

         mfem::MatrixCoefficient * tIdentityMatrix = new mfem::IdentityMatrixCoefficient(dim);
         mfem::DiffusionCoeff * tDiffCoeff = new mfem::DiffusionCoeff();
         tDiffCoeff->SetDensity(mDensCoeff);

         mfem::MatrixCoefficient * tMaterialCoeff = new ScalarMatrixProductCoefficient (*tDiffCoeff, *tIdentityMatrix);

         if( true )
         {
            ParaViewDataCollection mPvdc("velmesh", pmesh);
            mPvdc.SetDataFormat(VTKFormat::BINARY32);
            mPvdc.SetHighOrderOutput(true);
            //mPvdc->SetLevelsOfDetail(mCtk.order);
            mPvdc.SetCycle(0);
            mPvdc.SetTime(0.0);
            mPvdc.RegisterField("vel", &tOszillVelGF);
            mPvdc.Save();

         }

         mfem::Advection_Diffusion_Solver * solverAdvDiff = new mfem::Advection_Diffusion_Solver(pmesh,2);
         mfem::Advection_Diffusion_Solver * solverAdvDiff2 = new mfem::Advection_Diffusion_Solver(pmesh,2);

         {

            solverAdvDiff->SetVelocity(&OzillVelCoeff);

            mfem::Vector ConstVector(pmesh->Dimension());   ConstVector = 0.0;   ConstVector(0) = 1.0;
            mfem::VectorConstantCoefficient avgTemp(ConstVector);
            solverAdvDiff->SetGradTempMean( &avgTemp );

            //add material
            solverAdvDiff->AddMaterial(tMaterialCoeff);

            solverAdvDiff->SetDensityCoeff(tGeometry,tProjectionType,tRand[4]);

            solverAdvDiff->FSolve();

            Flux_x = solverAdvDiff->Postprocess();
         }
         {
            solverAdvDiff2->SetVelocity(&OzillVelCoeff);

            mfem::Vector ConstVector(pmesh->Dimension());   ConstVector = 0.0;   ConstVector(1) = 1.0;
            mfem::VectorConstantCoefficient avgTemp(ConstVector);
            solverAdvDiff2->SetGradTempMean( &avgTemp );

            //add material
            solverAdvDiff2->AddMaterial(tMaterialCoeff);

            solverAdvDiff2->SetDensityCoeff(tGeometry,tProjectionType,tRand[4]);

            solverAdvDiff2->FSolve();

            Flux_y = solverAdvDiff2->Postprocess();
         }

         delete solverAdvDiff2;
         delete solverAdvDiff;

         if (mpi.Root())
         {
            std::string tString = "./OutputFileFlux_" + std::to_string(run_id);

            Ascii tAsciiWriter( tString, FileMode::NEW );

            tAsciiWriter.print(stringify( tRand[0] ));
            tAsciiWriter.print(stringify( tRand[1] ));
            tAsciiWriter.print(stringify( tRand[4] ));

            for( int Ik = 0; Ik < dim; Ik ++)
            {
               tAsciiWriter.print(stringify( Flux_x(Ik) ));
               tAsciiWriter.print(stringify( Flux_y(Ik) ));
            }
         
            tAsciiWriter.save();
         }
      }




   std::cout<<  "perform executed" << std::endl;
   }

   delete pmesh;

   return 0;
}
