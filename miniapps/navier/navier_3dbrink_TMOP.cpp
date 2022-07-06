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



   //Mesh *mesh = new Mesh("bar3d.msh");
   Mesh *mesh = new Mesh("./cube.mesh");
   //mesh->EnsureNCMesh(true);

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   if (mpi.Root())
   {
      std::cout << "Mesh of elements: " << pmesh->GetNE() << std::endl;
   }

   if(true)
   {
         // Mesh morphing based on level set field
         const int quadOrder = 8; // integration order
         const int targetId = 2;
         const int metricId =  303;
         const int maxSolvItr = 60;
         const bool moveBnd = false;
         const int verboLev = 2;
         const double surfaceFit = 1e3;

         mfem::MeshMorphingSolver  tmopMeshMorphing_( pmesh, quadOrder, targetId, metricId, maxSolvItr, moveBnd, verboLev, surfaceFit);



         const int dim = pmesh->Dimension();
         mfem::H1_FECollection FEColl(2, dim);
         mfem::ParFiniteElementSpace FEScalarSpace(pmesh, &FEColl, 1, pmesh->GetNodalFESpace()->GetOrdering());
         mfem::ParFiniteElementSpace FEVectorSpace(pmesh, &FEColl, dim, pmesh->GetNodalFESpace()->GetOrdering());
         pmesh->SetNodalFESpace(&FEVectorSpace);

         // Set the nodal grid function
         mfem::ParGridFunction x(&FEVectorSpace);
         pmesh->SetNodalGridFunction(&x);

         mfem::ParGridFunction nodeDisp(&FEVectorSpace);



         mfem::ParGridFunction LSF_(&FEScalarSpace);

         mfem::navier::DensityCoeff * mDensCoeff = new DensityCoeff;
         mDensCoeff->SetThreshold(0.6);
         mDensCoeff->SetPatternType(DensityCoeff::PatternType::Gyroid);
         mDensCoeff->SetProjectionType(DensityCoeff::ProjectionType::continuous);

         LSF_.ProjectCoefficient(* mDensCoeff );

      if( true )
      {

         mfem::ParaViewDataCollection mPvdc("TMOPGyroid_1", pmesh);

        mPvdc.RegisterField("LSF", &LSF_);
        mPvdc.SetCycle(1);
        mPvdc.SetTime(1);
        mPvdc.Save();

        std::cout<<"-----------------------field output------------------------------"<<std::endl;
      }


         tmopMeshMorphing_.Solve(LSF_);

      if( true )
      {
         LSF_.ProjectCoefficient(* mDensCoeff );

         mfem::ParaViewDataCollection mPvdc("TMOPGyroid", pmesh);
         mPvdc.SetCycle(0);
         mPvdc.SetTime(0.0);
         mPvdc.RegisterField("LevelSet", &LSF_);
         mPvdc.Save();
      }
   }

   //----------------------------------------------------------

   //tForce_Magnitude = tForce_Magnitude * 1.154e-3 *5.0 /(1e3 * std::pow(1.75e-2 ,2));
   //tForce_Magnitude = tForce_Magnitude * 3.25e-3 /(1e3 * std::pow(8.0e-3 ,2));
   tForce_Magnitude = tForce_Magnitude * 2.64e-3 /(1.18 * std::pow(1.0 ,2));


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
        tRand[4] = 0.1;//0.65;  //0.4365
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

      tWorkflow.SetDensityCoeff(  );

      tWorkflow.SetupFlowSolver(  );

       tWorkflow.SetInitialConditions(  vel );

      tWorkflow.SetupOutput(  );

      tWorkflow.Perform(  );

      tWorkflow.Postprocess(  run_id );


   std::cout<<  "perform executed" << std::endl;
   }

   // //Refine the mesh
   // if(0)
   // {
   //     int nclimit=1;
   //     for (int iter = 0; iter<3; iter++)
   //     {
   //         Array<Refinement> refs;
   //         for (int i = 0; i < pmesh->GetNE(); i++)
   //         {
   //            bool refine = false;
   //            Geometry::Type geom = pmesh->GetElementBaseGeometry(i);
   //            ElementTransformation *T = pmesh->GetElementTransformation(i);
   //            RefinedGeometry *RefG = mfem::GlobGeometryRefiner.Refine(geom, 2, 1);
   //            IntegrationRule &ir = RefG->RefPts;

   //            // Refine any element where different materials are detected. A more
   //            // sophisticated logic can be implemented here -- e.g. don't refine
   //            // the interfaces between certain materials.
   //            Array<int> mat(ir.GetNPoints());
   //            double matsum = 0.0;
   //            for (int j = 0; j < ir.GetNPoints(); j++)
   //            {
   //               //T->Transform(ir.IntPoint(j), pt);
   //               //int m = material(pt, xmin, xmax);
   //               int m = dens.Eval(*T,ir.IntPoint(j));
   //               mat[j] = m;
   //               matsum += m;
   //               if ((int)matsum != m*(j+1))
   //               {
   //                  refine = true;
   //               }
   //            }

   //            // Mark the element for refinement
   //            if (refine)
   //            {
   //                refs.Append(Refinement(i));
   //            }

   //         }

   //         //pmesh->GeneralRefinement(refs, -1, nclimit);
   //         pmesh->GeneralRefinement(refs, 0, nclimit);
   //         //pmesh->GeneralRefinement(refs);
   //     }

   //     //pmesh->Rebalance();
   // }

   delete pmesh;

   return 0;
}
