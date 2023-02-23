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
#include "hpc4solvers.hpp"
#include "hpc4mat.hpp"
#include <fstream>
#include <ctime>
#include <cstdlib> 
#include <unistd.h>
#include "ascii.hpp"

using namespace mfem;

static int BoxIterator = 0;

double BoxFunction(const Vector &x)
{
   
   std::vector<double> rightval_x = {0.2, 0.4, 0.6, 1.2, 0.2, 0.4, 0.6, 1.2 , 0.2, 0.4, 0.6, 1.2 };
   std::vector<double> leftVal_x = {0.1, 0.3, 0.5, 1.1, 0.1, 0.3, 0.5, 1.1 , 0.1, 0.3, 0.5, 1.1 };

   std::vector<double> lowerVal_y  = {0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 0.8 };
   std::vector<double> upperVal_y = {0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.9, 0.9 };

   double val = 0.0;
   if( x[0] > leftVal_x[BoxIterator] && x[0] < rightval_x[BoxIterator] &&
       x[1] > lowerVal_y[BoxIterator] && x[1] < upperVal_y[BoxIterator] )
   {
      val = 1.0;
   }
   return val;
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

  
   double tLengthScale = 1.0e-2;
   double tThreshold = 0.6;
   double tDensity = 1.0e3;
   double tRefVelocity = 1.0e-3; 
   double tKinViscosity = 1.0e-6; 
   double ReynoldsNumber = tLengthScale * tRefVelocity / tKinViscosity;

   double tPoriosity = 0.1; 
   double tPermeability = 1.0e-3; 
   double tDa = tPoriosity / tPermeability; 

   Mesh mesh("Flow2D_full.msh");
   int dim = mesh.Dimension();

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


   // Allocate the nonlinear diffusion solver
   mfem::NLDiffusion* solver=new mfem::NLDiffusion(pmesh,2);

   //add boundary conditions
   //solver->AddDirichletBC(2,0.0);
   solver->AddDirichletBC(3,0.0);
   solver->AddDirichletBC(4,0.0);

   // build h1 desing space
   int orderDesing = 1;
   int orderAnalysis = 2;
   ::mfem::H1_FECollection desFECol_H1(orderDesing, dim);
   ::mfem::ParFiniteElementSpace desFESpace_scalar_H1(pmesh, &desFECol_H1 );

   ::mfem::H1_FECollection anaFECol_H1(orderAnalysis, dim);
   ::mfem::ParFiniteElementSpace anaVelFESpace_scalar_H1(pmesh, &desFECol_H1,2 );

   //----------------------------------------------------------
   // initialize desing field
   mfem::ParGridFunction desingVarVec(&desFESpace_scalar_H1); desingVarVec=tThreshold;
 
   //----------------------------------------------------------
   // add material

   mfem::SurrogateNLDiffusionCoefficient* tMatCoeff = new mfem::SurrogateNLDiffusionCoefficient();
   solver->AddMaterial(tMatCoeff);
   mfem::Coefficient * DesingCoeff = new mfem::GridFunctionCoefficient(&desingVarVec);
   solver->AddDesignCoeff(DesingCoeff);
   solver->AddDesignGF(&desingVarVec);

//-----------------------------

   // Vector NNInput(dim+1); NNInput=0.0;
   // Vector rr(dim+1);
   // NNInput[0] =  1.0;
   // NNInput[2] = 0.3;

   //     tMatCoeff->Grad(NNInput,rr);
   //     rr.Print();

   //----------------------------------------------------------
   //solve
   solver->FSolve();

   mfem::ParGridFunction tPreassureGF;
   solver->GetSol(tPreassureGF);

   //----------------------------------------------------------
   //postprocess
   mfem::ParaViewDataCollection paraview_dc("2DValidationHomoginization", pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(orderAnalysis);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(1);
   paraview_dc.SetTime(1.0);
   paraview_dc.RegisterField("design",&desingVarVec);
   paraview_dc.RegisterField("pressure",&tPreassureGF);

   ::mfem::VectorCoefficient* VelVoeff = new mfem::VelCoefficient(
            tMatCoeff,
            &tPreassureGF,
            &desingVarVec  ); 

   mfem::ParGridFunction velGF(&anaVelFESpace_scalar_H1);
   velGF.ProjectCoefficient(*VelVoeff);

   paraview_dc.RegisterField("velocity",&velGF);     
   paraview_dc.Save();

      double BoxVal = 0.0;

      int NumBox = 12;

      for( int Ik = 0; Ik < NumBox; Ik++)
      {
         mfem::Coefficient * tPreasusreCoeff = new mfem::GridFunctionCoefficient( &tPreassureGF);
         mfem::Coefficient * tIndicatorCoeff = new mfem::FunctionCoefficient( BoxFunction);
         mfem::Coefficient * tFinalCoeff = new mfem::ProductCoefficient( *tPreasusreCoeff, *tIndicatorCoeff);
   
         mfem::ParLinearForm BoxQILinearForm(&desFESpace_scalar_H1);
         BoxQILinearForm.AddDomainIntegrator( new mfem::DomainLFIntegrator( *tFinalCoeff ) );
         BoxQILinearForm.Assemble();
         BoxQILinearForm.ParallelAssemble();

         mfem::ParGridFunction OneGridGunction(&desFESpace_scalar_H1); OneGridGunction =1.0;
         BoxVal = BoxQILinearForm * OneGridGunction;
         double TotalBoxVal = 0.0;

         MPI_Allreduce(
            &BoxVal,
            &TotalBoxVal, 
            1, 
            MPI_DOUBLE, 
            MPI_SUM,
            MPI_COMM_WORLD);

         if (mpi.Root())
         {
            std::cout<<"--------------------------------------------------"<<std::endl;
            std::cout<<"BoxVal Preassures: "<< Ik<<" | "<<TotalBoxVal<<std::endl;
            std::cout<<"--------------------------------------------------"<<std::endl;
         }

         BoxIterator = Ik + 1;

         delete tFinalCoeff;
         delete tIndicatorCoeff;
         delete tPreasusreCoeff;
      }




   
   delete solver;
   delete pmesh;

   return 0;
}
