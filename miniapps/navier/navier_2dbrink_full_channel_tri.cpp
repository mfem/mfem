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

static int BoxIterator = 0;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = 0.015; //1e-4 / 5.0e-4;
   u(1) = 0.0;
}

void vel_0(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = 0.0; //1e-4 / 5.0e-4;
   u(1) = 0.0;
}

double pres(const Vector &x, double t)
{
   return 0.0;
}


double BoxFunction(const Vector &x)
{
   
   // std::vector<double> rightval_x = { 1.08, 1.28, 1.48, 1.68, 1.88 };
   // std::vector<double> leftVal_x = { 1.02, 1.22, 1.42, 1.62, 1.82 };

   // std::vector<double> lowerVal_y  = {0.22, 0.32, 0.22, 0.32, 0.22  };
   // std::vector<double> upperVal_y = {0.28, 0.38, 0.28, 0.38, 0.28 };

   // std::vector<double> rightval_x = { 1.12, 1.32, 1.52, 1.72, 1.92 };
   // std::vector<double> leftVal_x = { 1.00, 1.20, 1.40, 1.60, 1.80 };

   // std::vector<double> lowerVal_y  = {0.20, 0.30, 0.20, 0.30, 0.20  };
   // std::vector<double> upperVal_y = {0.32, 0.42, 0.32, 0.42, 0.32 };

   // std::vector<double> rightval_x = { 1.18, 1.38, 1.58, 1.78, 1.98 };
   // std::vector<double> leftVal_x = { 1.00, 1.20, 1.40, 1.60, 1.80 };

   // std::vector<double> lowerVal_y  = {0.20, 0.30, 0.20, 0.30, 0.20  };
   // std::vector<double> upperVal_y = {0.38, 0.48, 0.38, 0.48, 0.38 };

      std::vector<double> rightval_x = { 1.24, 1.44, 1.64, 1.84, 2.04 };
   std::vector<double> leftVal_x = { 1.00, 1.20, 1.40, 1.60, 1.80 };

   std::vector<double> lowerVal_y  = {0.10, 0.20, 0.10, 0.20, 0.10  };
   std::vector<double> upperVal_y = {0.38, 0.48, 0.38, 0.48, 0.38 };

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

   bool mreadVelField = false;
   bool tPerturbed = false;
   double PerturbationSize = 0.01;
   bool LoadSolVecFromFile = false;
   enum DensityCoeff::PatternType tGeometry = DensityCoeff::PatternType::Triangles;
   enum DensityCoeff::ProjectionType tProjectionType = DensityCoeff::ProjectionType::zero_one;

  
   double tLengthScale = 2.0e-2;
   double tThreshold = 0.4;
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
   ctx.t_final = 2.2;
   ctx.dt = 1e-4;

   //mesh->EnsureNCMesh(true);

   double MultInX = 0.5;
   double MultInY = 0.12;
   double Lx = 1.0 * MultInX;
   double Ly = 1.0 * MultInY;
   int NX = 64;
   int NY = 16;

   Mesh mesh = Mesh::MakeCartesian2D(NX, NY, Element::QUADRILATERAL, true,Lx, Ly);

   //    int tNumVertices  = mesh.GetNV();
   // for (int i = 0; i < tNumVertices; ++i)
   // {
   //     double * Coords = mesh.GetVertex(i);

   //     Coords[ 0 ] = Coords[ 0 ] - 0.2;  
   //     Coords[ 1 ] = Coords[ 1 ] + 0.0;
   // }

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

   if( true )
   {

      mfem::ParaViewDataCollection mPvdc("2D_FullMesh", pmesh);
      mPvdc.SetCycle(0);
      mPvdc.SetTime(0.0);
      mPvdc.Save();
   }
 
   //----------------------------------------------------------
   {
      DensityCoeff* DensCoeff = new DensityCoeff;
      DensCoeff->SetThreshold(tThreshold);
      DensCoeff->SetPatternType(tGeometry);
      DensCoeff->SetProjectionType(tProjectionType);
      DensCoeff->SetBallCoord(0.5,0.5,1.0);

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

      Array<int> attrVel(pmesh->bdr_attributes.Max());
      attrVel[3] = 1;
      attrVel[0] = 0;
      attrVel[1] = 0;
      attrVel[2] = 0;
      mFlowsolver->AddVelDirichletBC(vel, attrVel);

      //Array<int> attrVel1(pmesh->bdr_attributes.Max());
      //attrVel1[0] = 1;
      //mFlowsolver->AddVelDirichletBC(vel_0, attrVel1);

      Array<int> attrPres(pmesh->bdr_attributes.Max());
      attrPres[1] = 1;
      attrPres[2] = 0;
      attrPres[3] = 0;
      attrPres[0] = 0;
      //attrPres[3] = 1;
      mFlowsolver->AddPresDirichletBC(pres, attrPres);

      mFlowsolver->Setup(ctx.dt);


         std::cout<<"setup  done: "<<std::endl;
      ParGridFunction *u_gf = mFlowsolver->GetCurrentVelocity();
      ParGridFunction *p_gf = mFlowsolver->GetCurrentPressure();
      ParGridFunction *d_gf = new ParGridFunction(*p_gf);
      ParGridFunction *u_gf_dim = new ParGridFunction(*u_gf);
      ParGridFunction *p_gf_dim = new ParGridFunction(*p_gf);
      *u_gf_dim *= 1e-4;
      *p_gf_dim *= 1e-5;
      //DensCoeff->SetProjectionType(DensityCoeff::ProjectionType::continuous);
      d_gf->ProjectCoefficient(*DensCoeff);
      DensCoeff->SetProjectionType(DensityCoeff::ProjectionType::zero_one);

      ParaViewDataCollection* mPvdc = new ParaViewDataCollection("2DFull", pmesh);
      mPvdc->SetDataFormat(VTKFormat::BINARY32);
      mPvdc->SetHighOrderOutput(true);
      mPvdc->SetLevelsOfDetail(ctx.order);
      mPvdc->SetCycle(0);
      mPvdc->SetTime(0.0);
      mPvdc->RegisterField("velocity", u_gf);
      mPvdc->RegisterField("pressure", p_gf);
      mPvdc->RegisterField("velocity_dim", u_gf_dim);
      mPvdc->RegisterField("pressure_dim", p_gf_dim);
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

         if (step % 1000 == 0)
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


     if(mreadVelField)
   {
      std::string tStringIn = "./FinalVelGF";
      int n = 6;
      std::string tWorlsRank = std::to_string( mpi.WorldRank());
    
      int precision = n - tWorlsRank.size();
      std::string s = std::string(precision, '0').append(tWorlsRank);

      tStringIn= tStringIn +"."+s;

      std::ifstream inp(tStringIn);
      mfem::ParGridFunction FieldGF(pmesh, inp);

      *u_gf = FieldGF;

   }
   else
   {
      std::string uOutputNameGF = "FinalVelGF";
      std::string pOutputNameGF = "FinalPresGF";

      u_gf->Save( uOutputNameGF.c_str() );
      p_gf->Save( pOutputNameGF.c_str() );
   }


      std::vector<double> filterVal = { 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001 };

      std::vector<mfem::ParGridFunction > filteredVField(7, *u_gf);

      for( int Ik = 0; Ik < 7; Ik++)
      {
        //mfem::ParGridFunction filteredVField(*u_gf);
        mfem::VectorGridFunctionCoefficient inputGFCoef(u_gf);
        mfem::ParLinearForm rhsForm(u_gf->ParFESpace());
        rhsForm.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(inputGFCoef));
        rhsForm.Assemble();

        mfem::HypreParMatrix A;
        mfem::Vector X, B;
        filteredVField[Ik] = 0.0;
        Array<int> ess_tdof_list_;
        ess_tdof_list_.SetSize(0);

         mfem::ParBilinearForm lhsForm_(u_gf->ParFESpace());
         mfem::Coefficient * tFilterRadius = new mfem::ConstantCoefficient(filterVal[Ik]);
         lhsForm_.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*tFilterRadius, 2));
         lhsForm_.AddDomainIntegrator(new mfem::VectorMassIntegrator());
         lhsForm_.Assemble();
         lhsForm_.FormLinearSystem(ess_tdof_list_, filteredVField[Ik], rhsForm, A, X, B);

        mfem::HypreDiagScale jacobi(A);

        mfem::HyprePCG pcg(A);
        pcg.SetTol(1e-12);
        pcg.SetMaxIter(1000);
        pcg.SetPrintLevel(0);
        pcg.SetPreconditioner(jacobi);
        pcg.Mult(B, X);
        lhsForm_.RecoverFEMSolution(X, rhsForm, filteredVField[Ik]);





         delete tFilterRadius;

      }

      ParaViewDataCollection* mPvdc_final = new ParaViewDataCollection("2DFull_final", pmesh);
      mPvdc_final->SetDataFormat(VTKFormat::BINARY32);
      mPvdc_final->SetHighOrderOutput(true);
      mPvdc_final->SetLevelsOfDetail(ctx.order);
      mPvdc_final->RegisterField("velocity", u_gf);

      for( int Ik = 0; Ik < 7; Ik++)
      {
         std::string tFieldName = "filteredVel_" + std::to_string(Ik);
         mPvdc_final->RegisterField(tFieldName, &(filteredVField[Ik]));
         
      }
      mPvdc_final->Save();    



      double BoxPVal = 0.0;
      double BoxVVal = 0.0;

      int NumBox = 5;

      mfem::Coefficient * tPreasusreCoeff = new mfem::GridFunctionCoefficient( p_gf);
      mfem::VectorCoefficient* VelVoeff = new mfem::VectorGridFunctionCoefficient(u_gf);


      for( int Ik = 0; Ik < NumBox; Ik++)
      {

         mfem::Coefficient * tVelInnerProdCoeff = new mfem::InnerProductCoefficient (*VelVoeff, *VelVoeff);
         mfem::Coefficient * tIndicatorCoeff = new mfem::FunctionCoefficient( BoxFunction);
         mfem::Coefficient * tFinalPCoeff = new mfem::ProductCoefficient( *tPreasusreCoeff, *tIndicatorCoeff);
         mfem::Coefficient * tFinalVCoeff = new mfem::ProductCoefficient( *tVelInnerProdCoeff, *tIndicatorCoeff);
   
         mfem::ParLinearForm BoxPQILinearForm(p_gf->ParFESpace());
         BoxPQILinearForm.AddDomainIntegrator( new mfem::DomainLFIntegrator( *tFinalPCoeff ) );
         BoxPQILinearForm.Assemble();
         BoxPQILinearForm.ParallelAssemble();

         mfem::ParLinearForm BoxVQILinearForm(p_gf->ParFESpace());
         BoxVQILinearForm.AddDomainIntegrator( new mfem::DomainLFIntegrator( *tFinalVCoeff ) );
         BoxVQILinearForm.Assemble();
         BoxVQILinearForm.ParallelAssemble();


   FiniteElementSpace *fes = u_gf->FESpace();
   int vdim = fes->GetVDim();

   // initilaize integradl of velozity vector
   Vector tVelVal(vdim);
   tVelVal = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);

         double w = tr->Weight() * ip.weight;

         Vector tVal;

         u_gf->GetVectorValue( e, ip, tVal);

         double indicator = tIndicatorCoeff->Eval(*tr, ip);

         tVal *= w * indicator;
         tVelVal += tVal;

   
      }

   }

   for( int Ik = 0; Ik < vdim; Ik ++)
   {
      double tVal = tVelVal(Ik); 
      double tTotalVal;

      MPI_Allreduce(
         &tVal,
         &tTotalVal, 
         1, 
         MPI_DOUBLE, 
         MPI_SUM,
         pmesh->GetComm());

         tVelVal(Ik) = tTotalVal;
   }




         mfem::ParGridFunction OneGridGunction(p_gf->ParFESpace()); OneGridGunction =1.0;
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
            std::cout<<"BoxVal Preassure_star: "<< Ik<<" | "<<TotalPBoxVal<< " | Preassure: "<< TotalPBoxVal* tDensity * std::pow(tRefVelocity ,2)<< " | Vel: "<< TotalVBoxVal<<" | vel2: "<<std::sqrt(tVelVal*tVelVal)<<std::endl;
            std::cout<<"--------------------------------------------------"<<std::endl;
         }

         BoxIterator = Ik + 1;
      }


      int MaxBdrAttr = pmesh->bdr_attributes.Max();
      Array<int> BdrAttr(pmesh->bdr_attributes.Max());
      BdrAttr[3] = 1;
      BdrAttr[0] = 0;
      BdrAttr[1] = 0;
      BdrAttr[2] = 0;

      mfem::ParLinearForm BoundaryLF(p_gf->ParFESpace());
      BoundaryLF.AddBoundaryIntegrator( new BoundaryLFIntegrator(*tPreasusreCoeff), BdrAttr);
      BoundaryLF.Assemble();
      BoundaryLF.ParallelAssemble();

      mfem::ParGridFunction OneGridGunction(p_gf->ParFESpace()); OneGridGunction =1.0;
      double BrdVal = BoundaryLF * OneGridGunction;
      double TotalBdrVal = 0.0;

      MPI_Allreduce(
            &BrdVal,
            &TotalBdrVal, 
            1, 
            MPI_DOUBLE, 
            MPI_SUM,
            MPI_COMM_WORLD);

      if (mpi.Root())
      {
         std::cout<<"--------------------------------------------------"<<std::endl;
         std::cout<<"Bdr Preassure: "<<  TotalBdrVal << " | Preassure: "<< TotalBdrVal/MultInY<<std::endl;
         std::cout<<"--------------------------------------------------"<<std::endl;
       }


      delete mPvdc;
      delete d_gf;
      delete DensCoeff;
      delete mFlowsolver;
   }

   delete pmesh;

   return 0;
}
