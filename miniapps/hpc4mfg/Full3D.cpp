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
#include "advection_diffusion_solver.hpp"
#include <fstream>
#include <ctime>
#include <cstdlib> 
#include <unistd.h>
#include "ascii.hpp"

using namespace mfem;

//s_NavierContext ctx;
 //ctx.order = 2;
 //ctx.kin_vis = 0.1;
// ctx.t_final = 1.0;
// ctx.dt = 1e-4;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   if (xi <= 1e-8)
   {
      u(0) = 1.0e-12;
   }
   else{
      u(0) = 1.0e-12;
   }
   u(1) = 0.0;
   u(2) = 0.0;

}

double pres(const Vector &x, double t)
{
   return 0.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   //const char *mesh_file = "bar3d.msh";
   int run_id = 0;

   int serial_refinements = 2;

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

   bool tPerturbed = false;
   double PerturbationSize = 0.01;
   bool LoadSolVecFromFile = false;
   enum stokes::DensityCoeff::PatternType tGeometry = stokes::DensityCoeff::PatternType::Grains;
   enum stokes::DensityCoeff::ProjectionType tProjectionType = stokes::DensityCoeff::ProjectionType::zero_one;
  
   double tLengthScale = 1.0e-2;
   double tThreshold = 0.65;
   double tDensity = 1.0e3;
   double tRefVelocity = 1.0e-3; 
   double tKinViscosity = 1.0e-0; 
   double ReynoldsNumber = tLengthScale * tRefVelocity / tKinViscosity;

   double tPoriosity = 0.1; 
   double tPermeability = 1.0e-3; 
   double tDa = tPoriosity / tPermeability; 
   double tBrinkmann = 0; 

   // s_NavierContext ctx;
   // ctx.order = 2;
   // ctx.kin_vis = 1.0 / ReynoldsNumber;
   // ctx.t_final = 3.0;
   // ctx.dt = 1e-45;

   //mesh->EnsureNCMesh(true);
   ParMesh *pmesh = nullptr;
   {

   Mesh mesh("native_heat_ex_wide.msh");

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh.UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }


   pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   for (int lev = 0; lev < 1; lev++)
   {
      if (mpi.Root())
      {
            std::cout << "Refining the mesh in parallel" << '\n';
      }
      pmesh->UniformRefinement();
   }

   }

   //delete mesh;
   if (mpi.Root())
   {
      std::cout << "Mesh of elements: " << pmesh->GetNE() << std::endl;
   }

   if( true )
   {

      mfem::ParaViewDataCollection mPvdc("3D_Mesh", pmesh);
      mPvdc.SetCycle(0);
      mPvdc.SetTime(0.0);
      mPvdc.Save();
      
      // mfem::FiniteElementCollection * fec = new H1_FECollection(1,3);
      // mfem::ParFiniteElementSpace	* fes = new ParFiniteElementSpace(pmesh,fec,3);   

      // for (int e = 0; e < fes->GetNE(); ++e)
      // {
      //    const FiniteElement *fe = fes->GetFE(e);
      //    const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
      //                                          fe->GetOrder());
      //    ElementTransformation *tr = fes->GetElementTransformation(e);

      //    for (int i = 0; i < ir.GetNPoints(); ++i)
      //    {
      //       const IntegrationPoint &ip = ir.IntPoint(i);
      //       tr->SetIntPoint(&ip);

      //       double w = tr->Weight();

      //       MFEM_ASSERT(w >= 0.0, "neg detJ found.");   
      //    }

      // }

      // delete fec;
      // delete fes;

   }

   //----------------------------------------------------------

   double preasureGrad = 0.01 + 0.001 *(tForce_Magnitude - 1.0);

   tForce_Magnitude = preasureGrad * tLengthScale /(tDensity * std::pow(tRefVelocity ,2));
 
   //----------------------------------------------------------


   mfem::stokes::Stokes* solver=new mfem::stokes::Stokes(pmesh,2);

   solver-> SetParams(0.0, 0.0, 0.0, 0.0, tThreshold );
   solver->SetDensityCoeff(tGeometry,tProjectionType);

   solver->AddPressDirichletBC(4, 0.0);
   solver->AddPressDirichletBC(5, 0.0);

   mfem::Vector VelBC(3); VelBC=0.0 ; VelBC[0]=0.001;
   mfem::VectorCoefficient * VelBCCoeff= new VectorConstantCoefficient(VelBC);

      mfem::Vector VelBC1(3); VelBC1=0.0 ; 
   mfem::VectorCoefficient * VelBCCoeff1= new VectorConstantCoefficient(VelBC1);

   solver->AddVelDirichletBC(2, *VelBCCoeff);
   solver->AddVelDirichletBC(3, *VelBCCoeff);

   solver->AddVelDirichletBC(1, *VelBCCoeff1);

   mfem::ParGridFunction u_gf = solver->GetVelSolGF();

   if( true )
   {
      //solve
      solver->FSolve();

      solver->Postprocess();

      // Deep cppy here. not needed 
      mfem::ParGridFunction u_gf = solver->GetVelSolGF();
      std::string tOutputNameGF = "VelGF";
      u_gf.Save( tOutputNameGF.c_str() );

      u_gf *=1e1;

      delete VelBCCoeff;
      delete VelBCCoeff1;
   }
   else
   {
      std::string tStringIn = "VelGF";
      int n = 6;
      std::string tWorlsRank = std::to_string( mpi.WorldRank());
    
      int precision = n - tWorlsRank.size();
      std::string s = std::string(precision, '0').append(tWorlsRank);

      tStringIn= tStringIn +"."+s;

      std::ifstream inp(tStringIn);
      ParGridFunction tLoadGF(pmesh, inp);

      u_gf = tLoadGF;

      u_gf *=1e1;
   }

   //---------------------------------------------------------

   mfem::VectorGridFunctionCoefficient VelCoeff(&u_gf);

   mfem::Advection_Diffusion_Solver * solverAdvDiff = new mfem::Advection_Diffusion_Solver(pmesh,2);

   solverAdvDiff->SetVelocity(&VelCoeff);

   mfem::Vector ConstVector(pmesh->Dimension());   ConstVector = 0.0;   //ConstVector(0) = 1.0;
   mfem::VectorConstantCoefficient avgTemp(ConstVector);
   solverAdvDiff->SetGradTempMean( &avgTemp );

   int dim = 3;

   //add material
   solverAdvDiff->AddMaterial(new mfem::IdentityMatrixCoefficient(dim));

   solverAdvDiff->AddDirichletBC(2 , 0.0);
   solverAdvDiff->AddDirichletBC(3 , 0.0);

   solverAdvDiff->SetDensityCoeff(tGeometry,tProjectionType);

   solverAdvDiff->FSolve();

   solverAdvDiff->Postprocess();

   delete solverAdvDiff;
   delete solver;

          
   

   delete pmesh;

   return 0;
}
