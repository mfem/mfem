// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//

#include "SDF_Generator.hpp"
#include <fstream>
#include "mfem.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

void OptimizeMeshWithAMRAroundZeroLevelSet(ParMesh &pmesh,
                                           GridFunctionCoefficient &ls_coeff,
                                           int amr_iter,
                                           ParGridFunction &distance_s,
                                           const int quad_order = 5,
                                           Array<ParGridFunction *> *pgf_to_update = NULL)
{
   mfem::H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   mfem::ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   mfem::ParGridFunction x(&h1fespace);

   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int iter = 0; iter < amr_iter; iter++)
   {
       std::cout << iter << " k10amriter\n";
      el_to_refine = 0.0;
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         DenseMatrix x_grad;
         h1fespace.GetElementDofs(e, dofs);
         const IntegrationRule &ir = irRules.Get(pmesh.GetElementGeometry(e),
                                                 quad_order);
         x.GetValues(e, ir, x_vals);
         double min_val = x_vals.Min();
         double max_val = x_vals.Max();
         // If the zero level set cuts the elements, mark it for refinement
         if (min_val < 0 && max_val >= 0)
         {
            el_to_refine(e) = 1.0;
         }
      }

      // Refine an element if its neighbor will be refined
      for (int inner_iter = 0; inner_iter < 0; inner_iter++)
      {
         el_to_refine.ExchangeFaceNbrData();
         GridFunctionCoefficient field_in_dg(&el_to_refine);
         lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
         for (int e = 0; e < pmesh.GetNE(); e++)
         {
            Array<int> dofs;
            Vector x_vals;
            lhfespace.GetElementDofs(e, dofs);
            const IntegrationRule &ir =
               irRules.Get(pmesh.GetElementGeometry(e), quad_order);
            lhx.GetValues(e, ir, x_vals);
            double max_val = x_vals.Max();
            if (max_val > 0)
            {
               el_to_refine(e) = 1.0;
            }
         }
      }

      // Make the list of elements to be refined
      Array<int> el_to_refine_list;
      for (int e = 0; e < el_to_refine.Size(); e++)
      {
         if (el_to_refine(e) > 0.0)
         {
            el_to_refine_list.Append(e);
         }
      }

      int loc_count = el_to_refine_list.Size();
      int glob_count = loc_count;
      MPI_Allreduce(&loc_count, &glob_count, 1, MPI_INT, MPI_SUM,
                    pmesh.GetComm());
      MPI_Barrier(pmesh.GetComm());
      if (glob_count > 0)
      {
         pmesh.GeneralRefinement(el_to_refine_list, 1);
      }

      if (pgf_to_update != NULL)
      {
         for (int i = 0; i < pgf_to_update->Size(); i++)
         {
            (*pgf_to_update)[i]->ParFESpace()->Update();
            (*pgf_to_update)[i]->Update();
         }
      }

      // Update
      h1fespace.Update();
      x.Update();

      x.ProjectCoefficient(ls_coeff);

      l2fespace.Update();
      el_to_refine.Update();

      lhfespace.Update();
      lhx.Update();

      distance_s.ParFESpace()->Update();
      distance_s.Update();
   }
}

void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.SetFromTrueDofs(fieldtrue);

   delete S;
   delete Lap;
}

void ComputeScalarDistanceFromLevelSet(ParMesh &pmesh,
                                       Coefficient &ls_coeff,
                                       ParGridFunction &distance_s,
                                       const int nDiffuse = 2,
                                       const int pLapOrder = 4,
                                       const int pLapNewton = 50)
{
   mfem::H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   mfem::ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   mfem::ParGridFunction x(&h1fespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   //Now determine distance
   const double dx = mfem::common::AvgElementSize(pmesh);
   mfem::common::DistanceSolver *dist_solver = NULL;

   const int p = pLapOrder;
   const int newton_iter = pLapNewton;
   auto ds = new mfem::common::PLapDistanceSolver(p, newton_iter);
   //   auto ds = new NormalizationDistanceSolver();
   dist_solver = ds;

   ParFiniteElementSpace pfes_s(*distance_s.ParFESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   ParGridFunction filt_gf(&pfes_s);
   mfem::common::PDEFilter filter(pmesh, 1.0 * dx);
   filter.Filter(ls_coeff, filt_gf);
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver->ComputeScalarDistance(ls_filt_coeff, distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

   DiffuseField(distance_s, nDiffuse);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();
}

// make SDF && mpirun -np 6 SDF -m GEBracket.obj -nx 9 -ny 9 -nz 9 -amr 4 -rs 2 -sid 11
// make SDF && mpirun -np 6 SDF -m sphere.obj -nx 4 -ny 4 -nz 4 -amr 3 -rs 2 -sid 21 -dist
// make SDF && mpirun -np 6 SDF -m ./GEBracket.obj  -rs 2
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   int myid = Mpi::WorldRank();
   const int num_procs = Mpi::WorldSize();

   int refinement = 0;
   const char *obj_file = "./Utah_Teapot.obj";
   int nx = 50;
   int ny = 15;
   int nz = 15;
   int amr_iter = 0;
   int save_id = 0;
   bool comp_dist = false;

   OptionsParser args(argc, argv);
   args.AddOption(&obj_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&refinement,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&nx,
                  "-nx",
                  "--nx",
                  "number of elements in x.");
   args.AddOption(&ny,
                  "-ny",
                  "--ny",
                  "number of elements in y.");
   args.AddOption(&nz,
                  "-nz",
                  "--nz",
                  "number of elements in z.");
   args.AddOption(&amr_iter,
                  "-amr",
                  "--amr",
                  "number of elements in z.");
   args.AddOption(&save_id,
                  "-sid",
                  "--sid",
                  "file save name");
   args.AddOption(&comp_dist, "-dist", "--comp-dist",
                     "-no-dist","--no-comp-dist", "Compute distance from 0 level set or not.");
   args.Parse();

   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   mfem::Vector tObjectOffset(3); tObjectOffset= 0.0;

   sdf::SDF_Generator tSDFGenerator( obj_file, tObjectOffset, true);
//   sdf::SDF_Generator tSDFGenerator( "./Utah_teapot.obj", tObjectOffset, true );
//   std::cout << " k100\n";

   sdf::Object *mObject = tSDFGenerator.GetObject();
   sdf::Data tData( *mObject );

   int nSElem = tData.GetNSurfaceElements();
   int ntri =  tData.GetNTriangles();
   Vector TMinX = tData.GetMinCoordX(),
          TMinY = tData.GetMinCoordY(),
          TMinZ = tData.GetMinCoordZ(),
          TMaxX = tData.GetMaxCoordX(),
          TMaxY = tData.GetMaxCoordY(),
          TMaxZ = tData.GetMaxCoordZ();
   MFEM_VERIFY(nSElem > 0 || tData.GetNVolumeElements() > 0 || ntri > 0,
               "CAD elements 0\n");
   if (myid == 0)
   {
       std::cout << nSElem << " " << tData.GetNVolumeElements() << " k101-CADELEMInfo\n";
       std::cout << TMinX.Min() << " " << TMaxX.Max() << " " <<
                    TMinY.Min() << " " << TMaxY.Max() << " " <<
                    TMinZ.Min() << " " << TMaxZ.Max() << " " <<  " k10bb\n";
   }

   //      // create core
   //      Core tCore( tMesh, tData, mVerboseFlag );

   double inc_fac = 1.2;

   // Create mesh
   double Lx = (1+inc_fac)*(TMaxX.Max()-TMinX.Min());
   double Ly = (1+inc_fac)*(TMaxY.Max()-TMinY.Min());
   double Lz = (1+inc_fac)*(TMaxZ.Max()-TMinZ.Min());
   if (myid == 0)
   {
       std::cout << Lx << " " <<  Ly << " " << Lz << " k10lxyz\n";
   }

   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(nx, ny, nz,
                                                 mfem::Element::HEXAHEDRON, Lx, Ly, Lz, true);

   double cx = 0.5*(TMaxX.Max()+TMinX.Min());
   double cy = 0.5*(TMaxY.Max()+TMinY.Min());
   double cz = 0.5*(TMaxZ.Max()+TMinZ.Min());


   int dim = mesh.Dimension();

   int tNumVertices  = mesh.GetNV();
   for (int i = 0; i < tNumVertices; ++i)
   {
      double * Coords = mesh.GetVertex(i);
      //  Coords[ 0 ] = Coords[ 0 ] - 5.3;
      //  Coords[ 1 ] = Coords[ 1 ] + 3.05;
      //  Coords[ 2 ] = Coords[ 2 ] - 1.7;

      Coords[ 0 ] = Coords[ 0 ] - (0.5*Lx-cx);
      Coords[ 1 ] = Coords[ 1 ] - (0.5*Ly-cy);
      Coords[ 2 ] = Coords[ 2 ] - (0.5*Lz-cz);
   }

   for (int i = 0; i < refinement; ++i)
   {
      mesh.UniformRefinement();
   }

   Vector bMin(dim), bMax(dim);
   mesh.GetBoundingBox(bMin, bMax);
   if (myid == 0)
   {
       std::cout << " mesh bounding box 0\n";
       bMin.Print();
       bMax.Print();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   mesh.EnsureNCMesh(true);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   // build h1 desing space
   int orderDesing = 1;
   ::mfem::H1_FECollection desFECol_H1(orderDesing, dim);
   ::mfem::ParFiniteElementSpace desFESpace_scalar_H1(pmesh, &desFECol_H1 );

   // desing variable vector
   mfem::ParGridFunction SDF_GridFunc(&desFESpace_scalar_H1);
   SDF_GridFunc=0.0;
   for (int i = 0; i < amr_iter; i++) {
       tSDFGenerator.DoAmrOnMeshBasedOnIntersections(pmesh, 1);
       desFESpace_scalar_H1.Update();
       SDF_GridFunc.Update();
   }
   tSDFGenerator.calculate_sdf( SDF_GridFunc );

   double mingf = SDF_GridFunc.Min(),
          maxgf = SDF_GridFunc.Max();
   MPI_Allreduce(MPI_IN_PLACE, &mingf, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &maxgf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   if (myid == 0)
   {
       std::cout << mingf << " " << maxgf << " k10minmaxgf\n";
   }
//   pmesh->SetCurvature(1, false, -1, 0);

   {
       pmesh->GetBoundingBox(bMin, bMax);
   }

   pmesh->DeleteGeometricFactors();
   tNumVertices  = pmesh->GetNV();
   for (int i = 0; i < tNumVertices; ++i)
   {
      double * Coords = pmesh->GetVertex(i);
      for (int d = 0; d < dim; d++)
      {
          Coords[ d ] = Coords[ d ] - 0.5*(bMax(d)+bMin(d));
      }
   }
   pmesh->DeleteGeometricFactors();


   std::cout<<"SDF done"<<std::endl;
   {
       Vector bMin(dim), bMax(dim);
       pmesh->GetBoundingBox(bMin, bMax);
       if (myid == 0)
       {
           std::cout << " mesh bounding box 2\n";
           bMin.Print();
           bMax.Print();
       }
   }
//   MFEM_ABORT(" ");

   VisItDataCollection visit_dc("sdf", pmesh);
   {
       visit_dc.SetTime(0.0);
       visit_dc.SetCycle(0);
       visit_dc.RegisterField("design",&SDF_GridFunc);
       visit_dc.RegisterField("signeddist",&SDF_GridFunc);
       visit_dc.Save();
   }
//   MFEM_ABORT(" ");
   int neglob = pmesh->GetGlobalNE();
   if (myid == 0) {
       std::cout << "Num Elems: " << neglob << std::endl;
   }

   {
      ostringstream mesh_name;
      mesh_name << "sdfmesh_" + std::to_string(save_id) + ".mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }
   {
      ostringstream mesh_name;
      mesh_name << "sdfsol_" + std::to_string(save_id) + ".gf";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      SDF_GridFunc.SaveAsOne(mesh_ofs);
   }
   if (!comp_dist) { delete pmesh; return 0; }

   mfem::ParGridFunction Dist_GridFunc(SDF_GridFunc);
   GridFunctionCoefficient gfc(&SDF_GridFunc);
   Array<ParGridFunction *> pgf_to_update(1);
   pgf_to_update[0] = &SDF_GridFunc;

   if (comp_dist)
   {
       ComputeScalarDistanceFromLevelSet(*pmesh, gfc,
                                         Dist_GridFunc);
       if (myid == 0)
       {
           std::cout << "done dist\n";
       }
   }

   {
//       VisItDataCollection visit_dc("sdf", pmesh);
       visit_dc.SetTime(1.0);
       visit_dc.SetCycle(1);
       visit_dc.RegisterField("design",&SDF_GridFunc);
       visit_dc.RegisterField("signeddist",&Dist_GridFunc);
       visit_dc.Save();
   }

   {
      ostringstream mesh_name;
      mesh_name << "sdfmesh_" + std::to_string(save_id) + ".mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }
   {
      ostringstream mesh_name;
      mesh_name << "sdfsold_" + std::to_string(save_id) + ".gf";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      Dist_GridFunc.SaveAsOne(mesh_ofs);
   }


    delete pmesh;

   return 0;
}
