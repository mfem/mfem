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
   
   std::vector<double> rightval_x = {0.88, 1.08, 1.28, 1.48, 1.68 };
   std::vector<double> leftVal_x = {0.82, 1.02, 1.22, 1.42, 1.62 };

   std::vector<double> lowerVal_y  = {0.22, 0.32, 0.22, 0.32, 0.22  };
   std::vector<double> upperVal_y = {0.28, 0.38, 0.28, 0.38, 0.28 };

   //    std::vector<double> rightval_x = { 0.92, 1.12, 1.32, 1.52, 1.72 };
   // std::vector<double> leftVal_x = { 0.80, 1.00, 1.20, 1.40, 1.60 };

   // std::vector<double> lowerVal_y  = {0.20, 0.30, 0.20, 0.30, 0.20  };
   // std::vector<double> upperVal_y = {0.32, 0.42, 0.32, 0.42, 0.32 };

   //    std::vector<double> rightval_x = { 0.98, 1.18, 1.38, 1.58, 1.78 };
   // std::vector<double> leftVal_x = { 0.80, 1.00, 1.20, 1.40, 1.60 };

   // std::vector<double> lowerVal_y  = {0.20, 0.30, 0.20, 0.30, 0.20  };
   // std::vector<double> upperVal_y = {0.38, 0.48, 0.38, 0.48, 0.38 };

   //       std::vector<double> rightval_x = { 1.04, 1.24, 1.44, 1.64, 1.84 };
   // std::vector<double> leftVal_x = { 0.80, 1.00, 1.20, 1.40, 1.60 };

   // std::vector<double> lowerVal_y  = {0.10, 0.20, 0.10, 0.20, 0.10  };
   // std::vector<double> upperVal_y = {0.38, 0.48, 0.38, 0.48, 0.38 };

   double val = 0.0;
   if( x[0] > leftVal_x[BoxIterator] && x[0] < rightval_x[BoxIterator] &&
       x[1] > lowerVal_y[BoxIterator] && x[1] < upperVal_y[BoxIterator] )
   {
      val = 1.0;
   }
   return val;
}

double desin_function(const Vector & x)
{
   // double val = ((sin((x[0]+0.2)/2.0*2.0*M_PI)*cos((x[1]+0.48)/0.48*2.0*M_PI)
   //                   +sin((x[1]+0.48)/0.48*2.0*M_PI)*cos(0)
   //                   +sin(0)*cos((x[0]+0.2)/2.0*2.0*M_PI))+1.0)*0.5*0.25 + 0.12;

   double val = 0.3;

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

   double dirichletVal = 0.0816; //0.389807; //0.389874;
   bool leftvel = true; //0.389874;
   double tLengthScale = 2.0e-2;
   double tThreshold = 0.3;
   double tDensity = 1.0e3;
   double tRefVelocity = 1.0e-3; 
   double tKinViscosity = 1.0e-6; 
   double ReynoldsNumber = tLengthScale * tRefVelocity / tKinViscosity;

   double tPoriosity = 0.1; 
   double tPermeability = 1.0e-3; 
   double tDa = tPoriosity / tPermeability; 

   static double DomainX = 0.16;
   double MultInX = DomainX;
   double MultInY = 0.08;
   double MultInZ = 0.04;
   double Lx = 1.0 * MultInX;
   double Ly = 1.0 * MultInY;
   double Lz = 1.0 * MultInZ;
   int NX = 36;
   int NY = 16;
   int NZ = 8;


   Mesh mesh = Mesh::MakeCartesian3D(NX, NY,NZ, Element::HEXAHEDRON, Lx, Ly,Lz);



   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   // Create translation vectors defining the periodicity
   Vector x_translation({10.0, 0.0, 0.0});
   Vector y_translation({0.0, 1.0*MultInY, 0.0});
   Vector z_translation({0.0, 0.0, 1.0*MultInZ});
   std::vector<Vector> translations = {x_translation, y_translation, z_translation};

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
   ::mfem::ParFiniteElementSpace anaVelFESpace_scalar_H1(pmesh, &desFECol_H1,dim );

   //----------------------------------------------------------
   // initialize desing field
   mfem::ParGridFunction desingVarVec(&desFESpace_scalar_H1); desingVarVec=tThreshold;

   // Allocate the nonlinear diffusion solver
   mfem::NLDiffusion* solver=new mfem::NLDiffusion(pmesh,orderAnalysis);


   mfem::ParGridFunction PinitialGF(&desFESpace_scalar_H1);

   if(true)
   {
      mfem::Vector locationVector(dim);
      mfem::Vector & tSol = solver->GetSol();

      //std::cout<<"SolSize: "<<tSol.Size()<<std::endl;

      for(int Ij = 0; Ij< desFESpace_scalar_H1.GetTrueVSize(); Ij++)
      {
         pmesh->GetNode(Ij, &locationVector[0]);

         const double * pCoords(static_cast<const double*>(locationVector));

         tSol[Ij] = dirichletVal - pCoords[0]*dirichletVal/MultInX ;

         // tSol[Ij] = rand() / double(RAND_MAX)*0.001;
      }
   }

   // {
   //    std::string tStringIn = "./PreassureGF/FinalPresGF";
   //    int n = 6;
   //    std::string tWorlsRank = std::to_string( mpi.WorldRank());
    
   //    int precision = n - tWorlsRank.size();
   //    std::string s = std::string(precision, '0').append(tWorlsRank);

   //    tStringIn= tStringIn +"."+s;

   //    std::ifstream inp(tStringIn);
   //    mfem::ParGridFunction FieldGF(pmesh, inp);

   //    PinitialGF = FieldGF;

   //    PinitialGF *= 0.3;
   // }

   // mfem::Coefficient * PInitialCoeff = new mfem::GridFunctionCoefficient(&PinitialGF);

   //add boundary conditions
   //solver->AddDirichletBC(2,0.0);

   if(leftvel)
   {
      solver->AddDirichletBC(3,0.0);
      solver->AddDirichletBC(5,dirichletVal);
   }
   else{
      solver->AddDirichletBC(5,0.0);
      solver->AddDirichletBC(3,dirichletVal);
   }


   
   //solver->AddDirichletBC(4,*PInitialCoeff);

 
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
   if(true)
   {
      mfem::FunctionCoefficient desin_function_coeff(desin_function);

      desingVarVec.ProjectCoefficient(desin_function_coeff);
   }

   //----------------------------------------------------------
   //solve
   int NumTimeSteps = 1;
   solver->SetNewtonSolver(1e-12, 1e-13,25, 1, 1.0);
   solver->SetLinearSolver(1e-12, 1e-13, 1000, 0);
   double tFinalLoad = 0.00;
   for(int Ik = 1; Ik <= NumTimeSteps; Ik++)
   {
      double tLoad = tFinalLoad / NumTimeSteps * Ik;
      solver->AddNeumannLoadVal( tLoad );
      solver->FSolve();
   }

   std::cout<<"---solved----"<<std::endl;
   mfem::ParGridFunction tPreassureGF;
   solver->GetSol(tPreassureGF);

   //----------------------------------------------------------
   //postprocess
   mfem::ParaViewDataCollection paraview_dc("3DValidationHomoginization", pmesh);
   paraview_dc.SetPrefixPath("ParaView3D");
   paraview_dc.SetLevelsOfDetail(orderAnalysis);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(1);
   paraview_dc.SetTime(1.0);
   paraview_dc.RegisterField("design",&desingVarVec);
   paraview_dc.RegisterField("pressure",&tPreassureGF);
   paraview_dc.RegisterField("pressureinitial",&PinitialGF);

   ::mfem::VectorCoefficient* VelVoeff = new mfem::VelCoefficient(
            tMatCoeff,
            &tPreassureGF,
            &desingVarVec  ); 

   ::mfem::ParGridFunction velGF(&anaVelFESpace_scalar_H1);
   velGF.ProjectCoefficient(*VelVoeff);

   paraview_dc.RegisterField("velocity",&velGF);     
   paraview_dc.Save();

   // double BoxPVal = 0.0;
   // double BoxVVal = 0.0;

   // int NumBox = 5;

   //    for( int Ik = 0; Ik < NumBox; Ik++)
   //    {
   //       mfem::Coefficient * tPreasusreCoeff = new mfem::GridFunctionCoefficient( &tPreassureGF);
   //       mfem::Coefficient * tVelInnerProdCoeff = new mfem::InnerProductCoefficient (*VelVoeff, *VelVoeff);
   //       mfem::Coefficient * tIndicatorCoeff = new mfem::FunctionCoefficient( BoxFunction);
   //       mfem::Coefficient * tFinalPCoeff = new mfem::ProductCoefficient( *tPreasusreCoeff, *tIndicatorCoeff);
   //       mfem::Coefficient * tFinalVCoeff = new mfem::ProductCoefficient( *tVelInnerProdCoeff, *tIndicatorCoeff);
   
   //       mfem::ParLinearForm BoxPQILinearForm(&desFESpace_scalar_H1);
   //       BoxPQILinearForm.AddDomainIntegrator( new mfem::DomainLFIntegrator( *tFinalPCoeff ) );
   //       BoxPQILinearForm.Assemble();
   //       BoxPQILinearForm.ParallelAssemble();

   //       mfem::ParLinearForm BoxVQILinearForm(&desFESpace_scalar_H1);
   //       BoxVQILinearForm.AddDomainIntegrator( new mfem::DomainLFIntegrator( *tFinalVCoeff ) );
   //       BoxVQILinearForm.Assemble();
   //       BoxVQILinearForm.ParallelAssemble();

   //       mfem::ParGridFunction OneGridGunction(&desFESpace_scalar_H1); OneGridGunction =1.0;
   //       BoxPVal = BoxPQILinearForm * OneGridGunction;
   //       BoxVVal = BoxVQILinearForm * OneGridGunction;
   //       double TotalPBoxVal = 0.0;
   //       double TotalVBoxVal = 0.0;

   //       MPI_Allreduce(
   //          &BoxPVal,
   //          &TotalPBoxVal, 
   //          1, 
   //          MPI_DOUBLE, 
   //          MPI_SUM,
   //          MPI_COMM_WORLD);

            
   //       MPI_Allreduce(
   //          &BoxVVal,
   //          &TotalVBoxVal, 
   //          1, 
   //          MPI_DOUBLE, 
   //          MPI_SUM,
   //          MPI_COMM_WORLD);

            
   // FiniteElementSpace *fes = velGF.FESpace();
   // int vdim = fes->GetVDim();

   // // initilaize integradl of velozity vector
   // Vector tVelVal(vdim);
   // tVelVal = 0.0;

   //  for (int e = 0; e < fes->GetNE(); ++e)
   // {
   //    const FiniteElement *fe = fes->GetFE(e);
   //    const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
   //                                             fe->GetOrder());
   //    ElementTransformation *tr = fes->GetElementTransformation(e);

   //    for (int i = 0; i < ir.GetNPoints(); ++i)
   //    {
   //       const IntegrationPoint &ip = ir.IntPoint(i);
   //       tr->SetIntPoint(&ip);

   //       double w = tr->Weight() * ip.weight;

   //       Vector tVal;

   //       velGF.GetVectorValue( e, ip, tVal);

   //       double indicator = tIndicatorCoeff->Eval(*tr, ip);

   //       tVal *= w * indicator;
   //       tVelVal += tVal;

   
   //    }

   // }

   // for( int Ik = 0; Ik < vdim; Ik ++)
   // {
   //    double tVal = tVelVal(Ik); 
   //    double tTotalVal;

   //    MPI_Allreduce(
   //       &tVal,
   //       &tTotalVal, 
   //       1, 
   //       MPI_DOUBLE, 
   //       MPI_SUM,
   //       pmesh->GetComm());

   //       tVelVal(Ik) = tTotalVal;
   // }


   

   //       if (mpi.Root())
   //       {
   //          std::cout<<"--------------------------------------------------"<<std::endl;
   //          std::cout<<"BoxVal Preassures: "<< Ik<<" |P: "<<TotalPBoxVal<<" |V: "<<TotalVBoxVal <<" | vel2: "<<std::sqrt(tVelVal*tVelVal)<<std::endl;
   //          std::cout<<"--------------------------------------------------"<<std::endl;
   //       }

   //       BoxIterator = Ik + 1;

   //       delete tFinalPCoeff;
   //       delete tFinalVCoeff;
   //       delete tVelInnerProdCoeff;
   //       delete tIndicatorCoeff;
   //       delete tPreasusreCoeff;
   //    }




   
   delete solver;
   delete pmesh;

   return 0;
}
