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
#include "ascii.hpp"

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

   if (xi <= -1.599)
   {
      u(0) = 1.0e-0;
   }
   else{
      u(0) = 0;
   }
   u(1) = 0.0;
   u(2) = 0.0;

}

double pres(const Vector &x, double t)
{
      double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   if (xi <= 1.599)
   {
      return 0.0;
   }

   return 0.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   //const char *mesh_file = "bar3d.msh";
   int run_id = 0;

   int serial_refinements = 0;

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
   enum DensityCoeff::PatternType tGeometry = DensityCoeff::PatternType::Gyroids;
   enum DensityCoeff::ProjectionType tProjectionType = DensityCoeff::ProjectionType::zero_one;
  
   double tLengthScale = 1.0e-2;
   double tThreshold = 0.2;
   double tDensity = 1.0e3;
   double tRefVelocity = 1.0e-3; 
   double tKinViscosity = 1.0e-0; 
   double ReynoldsNumber = tLengthScale * tRefVelocity / tKinViscosity;

   double tPoriosity = 0.1; 
   double tPermeability = 1.0e-3; 
   double tDa = tPoriosity / tPermeability; 
   double tBrinkmann = 0; 

   s_NavierContext ctx;
   ctx.order = 2;
   ctx.kin_vis = 1.0 / ReynoldsNumber;
   ctx.t_final = 3.0;
   ctx.dt = 1e-45;

   //mesh->EnsureNCMesh(true);

   Mesh mesh("heat_ex1_v4.mesh");

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
   {
      DensityCoeff* DensCoeff = new DensityCoeff;
      DensCoeff->SetThreshold(tThreshold);
      DensCoeff->SetPatternType(tGeometry);
      DensCoeff->SetProjectionType(tProjectionType);

      NavierSolver* mFlowsolver = new NavierSolver(pmesh, ctx.order, ctx.kin_vis);
      mFlowsolver->EnablePA(true);
      mFlowsolver->EnableNI(true);

      Array<int> domain_attr(pmesh->attributes.Max());
      domain_attr = 1;
      BrinkPenalAccel* Bp = new BrinkPenalAccel(pmesh->Dimension() );
      Bp->SetDensity(DensCoeff);
      Bp->SetBrinkmannPenalization(tBrinkmann);
      Bp->SetVel(mFlowsolver->GetCurrentVelocity());
      Bp->SetParams( 0.0, 0.0, 0.0, 0.0);
      mFlowsolver->AddAccelTerm(Bp,domain_attr);

      std::cout<<"pmesh->bdr_attributes.Max(): "<<pmesh->bdr_attributes.Max() <<std::endl;

      Array<int> attrVel(pmesh->bdr_attributes.Max());  attrVel = 0;
      attrVel[0] = 1;
      mFlowsolver->AddVelDirichletBC(vel, attrVel);

      Array<int> attrPres(pmesh->bdr_attributes.Max()); attrPres = 0;
      attrPres[0] = 1;
      mFlowsolver->AddPresDirichletBC(pres, attrPres);

      mFlowsolver->Setup(ctx.dt);


         std::cout<<"setup  done: "<<std::endl;
      ParGridFunction *u_gf = mFlowsolver->GetCurrentVelocity();
      ParGridFunction *p_gf = mFlowsolver->GetCurrentPressure();
      ParGridFunction *d_gf = new ParGridFunction(*p_gf);
      //DensCoeff->SetProjectionType(DensityCoeff::ProjectionType::continuous);
      d_gf->ProjectCoefficient(*DensCoeff);
      DensCoeff->SetProjectionType(DensityCoeff::ProjectionType::zero_one);

      ParaViewDataCollection* mPvdc = new ParaViewDataCollection("3DFull", pmesh);
      mPvdc->SetDataFormat(VTKFormat::BINARY32);
      mPvdc->SetHighOrderOutput(true);
      mPvdc->SetLevelsOfDetail(ctx.order);
      mPvdc->SetCycle(0);
      mPvdc->SetTime(0.0);
      mPvdc->RegisterField("velocity", u_gf);
      mPvdc->RegisterField("pressure", p_gf);
      mPvdc->RegisterField("density",  d_gf);
      mPvdc->Save();

      double t = 0.0;
      double dt = ctx.dt;
      double t_final = ctx.t_final;
      bool last_step = false;

      for (int step = 0; !last_step; ++step)
      {
         if (t + dt >= t_final - dt / 2)
         {
            last_step = true;
         }
         
         mFlowsolver->Step(t, dt, step);

         Bp->SetVel(mFlowsolver->GetCurrentVelocity());

         //mFlowsolver->GetCurrentVelocity()->norm2();
         //mBp->SetVel(flowsolver.GetProvisionalVelocity());

         if (step % 2000 == 0)
         {
            mPvdc->SetCycle(step);
            mPvdc->SetTime(t);
            mPvdc->Save();
         }

         if (mpi.Root())
         {
            printf("%11s %11s\n", "Time", "dt");
            printf("%.5E %.5E\n", t, dt);
            fflush(stdout);
         }
      }

      mFlowsolver->PrintTimingData();

      std::string uOutputNameGF = "FinalVelGF";
      std::string pOutputNameGF = "FinalPresGF";

      u_gf->Save( uOutputNameGF.c_str() );
      p_gf->Save( pOutputNameGF.c_str() );

      //---------------------------------------------------------

      // mfem::VectorGridFunctionCoefficient VelCoeff( u_gf);

      // mfem::Advection_Diffusion_Solver * solverAdvDiff = new mfem::Advection_Diffusion_Solver(pmesh,2);

      // solverAdvDiff->SetVelocity(&VelCoeff);

      // mfem::Vector ConstVector(pmesh->Dimension());   ConstVector = 0.0;   //ConstVector(0) = 1.0;
      // mfem::VectorConstantCoefficient avgTemp(ConstVector);

      // solverAdvDiff->SetGradTempMean( &avgTemp );

      // int dim = 3;

      // //add material
      // solverAdvDiff->AddMaterial(new mfem::IdentityMatrixCoefficient(dim));

      // solverAdvDiff->AddDirichletBC(1 , 0.0);
      // solverAdvDiff->AddDirichletBC(3 , 0.0);

      // solverAdvDiff->FSolve();

      // solverAdvDiff->Postprocess();

      // delete solverAdvDiff;
      // delete solver;

           //---------------------------------------------------------

      // FiniteElementSpace *fes = u_gf->FESpace();
      // int vdim = fes->GetVDim();

      // initilaize integradl of velozity vector
      // Vector tVelVal(vdim);
      // tVelVal = 0.0;

      // double tVolume = 0.0;

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


      //       double w = tr->Weight() * ip.weight;

      //       Vector tVal;

      //       u_gf->GetVectorValue( e, ip, tVal);

      //       tVal *= w;
      //       tVelVal += tVal;

      //       tVolume += w;
      //    }

      // }

      // double tTotalVol;
      // MPI_Allreduce(&tVolume, &tTotalVol, 1, MPI_DOUBLE, MPI_SUM,
      //             pmesh->GetComm());

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

      //       tVelVal(Ik) = tTotalVal / tTotalVol;
      // }

      // if (mpi.Root())
      // {
      //    std::string tString = "./OutputFile_";

      //    Ascii tAsciiWriter( tString, FileMode::NEW );

      //    for( int Ik = 0; Ik < vdim; Ik ++)
      //    {
      //       tAsciiWriter.print(stringify( tVelVal(Ik) ));
      //    }
         
      //    tAsciiWriter.save();
      // }
   }

   delete pmesh;

   return 0;
}
