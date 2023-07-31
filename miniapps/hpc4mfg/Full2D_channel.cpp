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
   
   std::vector<double> rightval_x = {0.06, 0.16, 0.26, 0.36, 0.46 };
   std::vector<double> leftVal_x = {0.04, 0.14, 0.24, 0.34, 0.44 };

   std::vector<double> lowerVal_y  = {0.02, 0.02, 0.02, 0.02, 0.02  };
   std::vector<double> upperVal_y = {0.06, 0.06, 0.06, 0.06, 0.06 };

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

  
   double tLengthScale = 2.0e-2;
   double tThreshold = 0.3;
   double tDensity = 1.0e3;
   double tRefVelocity = 1.0e-3; 
   double tKinViscosity = 1.0e-6; 
   double ReynoldsNumber = tLengthScale * tRefVelocity / tKinViscosity;

   double tPoriosity = 0.1; 
   double tPermeability = 1.0e-3; 
   double tDa = tPoriosity / tPermeability; 

   double MultInX = 0.5;
   double MultInY = 0.12;
   double Lx = 1.0 * MultInX;
   double Ly = 1.0 * MultInY;
   int NX = 64;
   int NY = 16;

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
   Vector x_translation({10.0, 0.0});
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

   int dim = pmesh->Dimension();

   // build h1 desing space
   int orderDesing = 1;
   int orderAnalysis = 1;
   ::mfem::H1_FECollection desFECol_H1(orderDesing, dim);
   ::mfem::ParFiniteElementSpace desFESpace_scalar_H1(pmesh, &desFECol_H1 );

   ::mfem::H1_FECollection anaFECol_H1(orderAnalysis, dim);
   ::mfem::ParFiniteElementSpace anaVelFESpace_scalar_H1(pmesh, &desFECol_H1,2 );

   //----------------------------------------------------------
   // initialize desing field
   mfem::ParGridFunction desingVarVec(&desFESpace_scalar_H1); desingVarVec=tThreshold;

   // Allocate the nonlinear diffusion solver
   mfem::NLDiffusion* solver=new mfem::NLDiffusion(pmesh,orderAnalysis);


   //add boundary conditions
   //solver->AddDirichletBC(2,0.0);
   solver->AddDirichletBC(2,0.0);
   solver->AddDirichletBC(4,0.317072);

 
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
   // NNInput[0] =  4.5835;
   // NNInput[1] = 7.7465;
   // NNInput[2] = 0.3;

   // tMatCoeff->Grad(NNInput,rr);
   // rr.Print();

   // DenseMatrix hh(3);

   // tMatCoeff->Hessian(NNInput,hh);

   // hh.Print();

   //    //  4.5835, 7.7465, 0.3000],
   //    //   [1.3385, 3.7950, 0.3000

   //     mfem_error("kill run");

//----------------------------------------------------------
   if(false)
   {
      mfem::Vector locationVector(2);

      for(int Ij = 0; Ij< desingVarVec.Size(); Ij++)
      {
         pmesh->GetNode(Ij, &locationVector[0]);

         //const double * pCoords(static_cast<const double*>(locationVector));

         desingVarVec[Ij] = std::sin( locationVector[0]/0.5*M_PI)*0.25 + 0.12;
      }
   }

   //----------------------------------------------------------
   //solve
   int NumTimeSteps = 1;
   solver->SetNewtonSolver(1e-6, 1e-8,15, 1, 1.0);
   solver->SetLinearSolver(1e-10, 1e-12, 1000, 0);
   double tFinalLoad = 0.0;
   for(int Ik = 1; Ik <= NumTimeSteps; Ik++)
   {
      double tLoad = tFinalLoad / NumTimeSteps * Ik;
      solver->AddNeumannLoadVal( tLoad );
      solver->FSolve();
   }


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

      double BoxPVal = 0.0;
      double BoxVVal = 0.0;

      int NumBox = 5;

      for( int Ik = 0; Ik < NumBox; Ik++)
      {
         mfem::Coefficient * tPreasusreCoeff = new mfem::GridFunctionCoefficient( &tPreassureGF);
         mfem::Coefficient * tVelInnerProdCoeff = new mfem::InnerProductCoefficient (*VelVoeff, *VelVoeff);
         mfem::Coefficient * tIndicatorCoeff = new mfem::FunctionCoefficient( BoxFunction);
         mfem::Coefficient * tFinalPCoeff = new mfem::ProductCoefficient( *tPreasusreCoeff, *tIndicatorCoeff);
         mfem::Coefficient * tFinalVCoeff = new mfem::ProductCoefficient( *tVelInnerProdCoeff, *tIndicatorCoeff);
   
         mfem::ParLinearForm BoxPQILinearForm(&desFESpace_scalar_H1);
         BoxPQILinearForm.AddDomainIntegrator( new mfem::DomainLFIntegrator( *tFinalPCoeff ) );
         BoxPQILinearForm.Assemble();
         BoxPQILinearForm.ParallelAssemble();

         mfem::ParLinearForm BoxVQILinearForm(&desFESpace_scalar_H1);
         BoxVQILinearForm.AddDomainIntegrator( new mfem::DomainLFIntegrator( *tFinalVCoeff ) );
         BoxVQILinearForm.Assemble();
         BoxVQILinearForm.ParallelAssemble();

         mfem::ParGridFunction OneGridGunction(&desFESpace_scalar_H1); OneGridGunction =1.0;
         BoxPVal = BoxPQILinearForm * OneGridGunction;
         BoxVVal = BoxVQILinearForm * OneGridGunction;
         double TotalPBoxVal = 0.0;
         double TotalVBoxVal = 0.0;

         MPI_Allreduce(
            &BoxPVal,
            &TotalPBoxVal, 
            1, 
            MPI_DOUBLE, 
            MPI_SUM,
            MPI_COMM_WORLD);

            
         MPI_Allreduce(
            &BoxVVal,
            &TotalVBoxVal, 
            1, 
            MPI_DOUBLE, 
            MPI_SUM,
            MPI_COMM_WORLD);

         if (mpi.Root())
         {
            std::cout<<"--------------------------------------------------"<<std::endl;
            std::cout<<"BoxVal Preassures: "<< Ik<<" |P: "<<TotalPBoxVal<<" |V: "<<TotalVBoxVal <<std::endl;
            std::cout<<"--------------------------------------------------"<<std::endl;
         }

         BoxIterator = Ik + 1;

         delete tFinalPCoeff;
         delete tFinalVCoeff;
         delete tVelInnerProdCoeff;
         delete tIndicatorCoeff;
         delete tPreasusreCoeff;
      }




   
   delete solver;
   delete pmesh;

   return 0;
}
