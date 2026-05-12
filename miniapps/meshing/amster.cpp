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
//    ---------------------------------------------------------------------
//    Amsterdam 2023 code -- AMSTER (Automatic Mesh SmooThER)
//    ---------------------------------------------------------------------
//
//
// Compile with: make amster
//
// Sample runs:
//
//    2D untangling:
//      mpirun -np 4 ./amster -m jagged.mesh -o 2 -qo 4 -no-fit
//    2D untangling + worst-case:
//      mpirun -np 4 ./amster -m amster_q4warp.mesh -o 2 -qo 6 -no-fit
//    2D fitting:
//      mpirun -np 6 amster -m amster_q4warp.mesh -rs 1 -o 3 -amr 7
//
//    2D orders prec:
//      mpirun -np 6 amster -m ../../data/star.mesh -rs 0 -o 1 -amr 7 -vis
//
//    3D untangling:
//      mpirun -np 6 amster -m ../../../mfem_data/cube-holes-inv.mesh -o 3 -qo 4 -no-fit

// Some new sample runs:
// make amster -j4 && mpirun -np 4 ./amster -m jagged.mesh -o 2 -qo 8 -vis -rs 0 -umid 66 -ni 1000 -utid 1 -visit -no-final
//  make amster -j4 && mpirun -np 1 ./amster -m jagged.mesh -qo 8 -vis -rs 0 -umid 4 -ni 1000 -utid 1 -visit -no-final -bound  -o 1
//  make amster -j4 && mpirun -np 1 ./amster -m jagged.mesh -qo 8 -vis -rs 0 -umid 4 -ni 1000 -utid 1 -visit -no-final -bound  -o 2
//  make amster -j4 && mpirun -np 1 ./amster -m jagged.mesh -qo 8 -vis -rs 0 -umid 4 -ni 1000 -utid 1 -visit -no-final -bound  -o 3
// untangling + skew improve
// make amster -j4 && mpirun -np 1 ./amster -m jagged.mesh -qo 8 -vis -rs 0 -umid 4 -ni 1000 -utid 1 -visit -bound  -o 1 -bnd -mid 49 -tid 1

// Blade
// Original - no bound on Jacobian and no tangential relaxation
// make amster -j4 && mpirun -np 4 ./amster -m blade.mesh -o 4 -qo 8 -vis -rs 0 -mid 2 -tid 1 -visit -ni 1000 -st 0 -no-bound -bnd
// Blade + bound on Jacobian
// make amster -j4 && mpirun -np 4 ./amster -m blade.mesh -o 4 -qo 8 -vis -rs 0 -mid 2 -tid 1 -bdropt 1 -visit -ni 1000 -bnd
// Blade + increase quad_order
// make amster -j4 && mpirun -np 4 ./amster -m blade.mesh -o 4 -qo 24 -vis -rs 0 -mid 2 -tid 1 -bdropt 1 -visit -ni 1000 -bnd
// Blade + increase quad_order + no bound on Jacobian
// make amster -j4 && mpirun -np 4 ./amster -m blade.mesh -o 4 -qo 24 -vis -rs 0 -mid 2 -tid 1 -bdropt 1 -visit -ni 1000 -bnd -no-bound
// Blade + bound + adaptive quadrature order
// make amster -j4 && mpirun -np 4 ./amster -m blade.mesh -o 4 -qo 8 -vis -rs 0 -mid 2 -tid 1 -bdropt 1 -visit -ni 1000 -bnd -aqp

// Ale tangled - curvilinear right and top boundaries
// make amster -j4 && mpirun -np 3 amster -m aletangled.mesh -o 2 -qo 8 -vis -rs 0 -umid 4 -utid 1 -mid 80 -tid 2 -bdropt 2 -visit -ni 5000 -bnd
// Ale tangled - rotated square hole
// make amster -j4 && mpirun -np 3 amster -m Laghos_2D_square_hole_800_mesh -o 2 -qo 8 -vis -rs 0 -umid 4 -utid 1 -mid 80 -tid 2 -bdropt 3 -visit -ni 200 -bnd
// Ale tangled - circular hole
// make amster -j4 && mpirun -np 3 amster -m Laghos_2D_circular_hole_650_mesh -o 2 -qo 8 -vis -rs 0 -umid 4 -utid 1 -mid 80 -tid 2 -bdropt 4 -visit -ni 200 -bnd -sm square-disc-q4.mesh

// 3D
// make amster -j4 && mpirun -np 10 amster -m hex6.mesh -o 2 -qo 8 -vis -rs 0 -mid 301 -tid 1 -bdropt 5 -visit -ni 200 -bnd -no-bound
// make amster -j4 && mpirun -np 10 amster -m kershaw_laghos_morphed_2.mesh -o 2 -qo 8 -vis -rs 0 -mid 301 -tid 1 -bdropt 8 -visit -ni 500 -bnd -no-bound

// amster_q4warp
// make amster -j4 && mpirun -np 10 amster -m amster_q4warp.mesh -o 4 -qo 8 -vis -rs 0 -mid 80 -tid 2 -bdropt 7 -visit -ni 2000


#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "mesh-fitting.hpp"
#include "amster.hpp"

using namespace mfem;
using namespace std;

void TransferLowToHigh(const ParGridFunction &l, ParGridFunction &h);
void TransferHighToLow(const ParGridFunction &h, ParGridFunction &l);

void Untangle(ParGridFunction &x, double min_detA, int quad_order,
              int metric_id, int target_id, GridFunction::PLBound *plb,
              ParGridFunction *detgf, int solver_iter, bool move_bnd,
              bool adapt_qp);
void WorstCaseOptimize(ParGridFunction &x, int quad_order,
                       int metric_id, int target_id, GridFunction::PLBound *plb,
                       ParGridFunction *detgf, int solver_iter,
                       double &min_det);
void GetDeterminantJacobianGF(ParMesh *mesh, ParGridFunction *detgf);

int main (int argc, char *argv[])
{
#ifndef MFEM_USE_GSLIB
   cout << "AMSTER requires GSLIB!" << endl; return 1;
#endif

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "jagged.mesh";
   const char *surf_mesh_file = "null.mesh";
   int rs_levels         = 0;
   int mesh_poly_deg     = 2;
   int solver_iter       = 50;
   int quad_order        = 8;
   int bg_amr_steps      = 6;
   double surface_fit_const = 10.0;
   double surface_fit_adapt = 10.0;
   double surface_fit_threshold = 1e-5;
   int metric_id         = 2;
   int target_id         = 1;
   int u_metric_id       = 2;
   int u_target_id       = 1;
   bool vis              = false;
   int wc_metric_id      = 4;
   int wc_target_id      = 1;
   int bdr_opt_case      = 0;
   bool visit            = false;
   bool move_bnd         = false;
   bool final_pass       = true;
   bool bound            = true;
   int solver_type       = 1;
   bool transform        = true;
   bool adapt_qp         = false;
   int job_id            = 0;

   // Parse command-line input file.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&surf_mesh_file, "-sm", "--surfmesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&bg_amr_steps, "-amr", "--amr-bg-steps",
                  "Number of AMR steps on the background mesh.");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface preservation constant.");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric 1/2/50/58 in 2D:\n\t");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Mesh optimization metric 1/2/3 in 2D:\n\t");
   args.AddOption(&u_metric_id, "-umid", "--u-metric-id",
                  "Mesh optimization metric 1/2/50/58 in 2D:\n\t");
   args.AddOption(&u_target_id, "-utid", "--u-target-id",
                  "Mesh optimization metric 1/2/3 in 2D:\n\t");
   args.AddOption(&vis, "-vis", "--vis", "-no-vis", "--no-vis",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&wc_metric_id, "-wcmid", "--wc-metric-id",
                  "Mesh optimization metric 1/2/50/58 in 2D:\n\t");
   args.AddOption(&wc_target_id, "-wctid", "--wc-target-id",
                  "Mesh optimization metric 1/2/3 in 2D:\n\t");
   args.AddOption(&bdr_opt_case, "-bdropt", "--bdr-opt",
                  "Boundary attribute for tangential relaxation:\n\t");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&final_pass, "-final", "--final", "-no-final",
                  "--no-final",
                  "Enable final mesh optimization pass.");
   args.AddOption(&bound, "-bound", "--bound", "-no-bound",
                  "--no-bound",
                  "Enable bounds.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  "0 - Newton, 1 - LBFGS:\n\t");
   args.AddOption(&transform, "-transform", "--transform", "-no-transform",
                  "--no-transform",
                  "Enable transforms.");
   args.AddOption(&adapt_qp, "-aqp", "--aqp", "-no-aqp",
                  "--no-aqp",
                  "Enable adaptive quad order.");
   args.AddOption(&job_id, "-jid", "--job-id",
                  "Job id to append to final output filename.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   mesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   Mesh *smesh = nullptr;
   if (strcmp(surf_mesh_file, "null.mesh") != 0)
   {
      smesh = new Mesh(surf_mesh_file, 1, 1, false);
      for (int lev = 0; lev < rs_levels; lev++) { smesh->UniformRefinement(); }
      smesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   }
   else
   {
      smesh = mesh;
   }

   if ((bdr_opt_case == 5 || bdr_opt_case == 6 ) && transform)
   {
      ModifyBoundaryAttributesForNodeMovement(mesh, *(mesh->GetNodes()));
      mesh->SetAttributes();

      // Kershaw transformation
      if (bdr_opt_case == 5)
      {
         common::KershawTransformation kershawT(mesh->Dimension(), 0.3, 0.3, 3);
         mesh->Transform(kershawT);

         // Do a rotation in 3D
         VectorFunctionCoefficient fcr(3, rotationTransformation);
         mesh->Transform(fcr);
      }
      else if (bdr_opt_case == 6)
      {
         VectorFunctionCoefficient fcw(3, warpingTransformation);
         mesh->Transform(fcw);
      }
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   const int dim = pmesh->Dimension();

   auto setTwoBits = [](int j, int k)
   {
      return (1 << (j-1)) | (1 << (k-1));
   };

   // Setup edge-mesh for tangential relaxation
   Array<ParMesh *> surf_mesh_arr;
   Array<int> surf_mesh_attr, surf_mesh_edge_attr;
   if (bdr_opt_case == 1) //blade
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   else if (bdr_opt_case == 2) // Ale tangled - curvilinear right and top
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(2);
      surf_mesh_arr.SetSize(2);
      surf_mesh_attr[0] = 3;
      surf_mesh_attr[1] = 4;
   }
   else if (bdr_opt_case == 3) // Ale tangled - rotated square hole
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(4);
      surf_mesh_arr.SetSize(4);
      surf_mesh_attr[0] = 3;
      surf_mesh_attr[1] = 4;
      surf_mesh_attr[2] = 5;
      surf_mesh_attr[3] = 6;
   }
   else if (bdr_opt_case == 4) // Ale tangled - rotated circular hole
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   else if (bdr_opt_case == 5 || bdr_opt_case == 6 || bdr_opt_case == 8) // 3D case - kershaw
   {
      MFEM_VERIFY(dim == 3,"3D case");
      surf_mesh_attr.SetSize(3);
      surf_mesh_attr[0] = 1;
      surf_mesh_attr[1] = 2;
      surf_mesh_attr[2] = 3;
      surf_mesh_edge_attr.SetSize(3);
      surf_mesh_edge_attr[0] = setTwoBits(1,2);
      surf_mesh_edge_attr[1] = setTwoBits(2,3);
      surf_mesh_edge_attr[2] = setTwoBits(1,3);

      surf_mesh_arr.SetSize(surf_mesh_attr.Size()+surf_mesh_edge_attr.Size());
   }
   else if (bdr_opt_case == 7) // amster_q4warp.mesh
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(4);
      surf_mesh_arr.SetSize(4);
      surf_mesh_attr[0] = 1;
      surf_mesh_attr[1] = 2;
      surf_mesh_attr[2] = 3;
      surf_mesh_attr[3] = 4;
   }
   else if (bdr_opt_case == 9) // jagged.mesh
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(5);
      surf_mesh_arr.SetSize(5);
      surf_mesh_attr[0] = 1;
      surf_mesh_attr[1] = 4;
      surf_mesh_attr[2] = 5;
      surf_mesh_attr[3] = 6;
      surf_mesh_attr[4] = 7;
   }
   double bbox_fac = 2.0; //2.0;

   Vector surf_measure_before(surf_mesh_attr.Size() + surf_mesh_edge_attr.Size());
   surf_measure_before = 0.0;

   auto getTwoSetBits = [](int val) -> std::pair<int, int>
   {
      std::pair<int, int> result = {-1, -1};
      int bitIndex = 1, count = 0;

      while (val && count < 2)
      {
         if (val & 1)
         {
            if (count == 0) { result.first = bitIndex; }
            else { result.second = bitIndex; }
            count++;
         }
         val >>= 1;
         bitIndex++;
      }

      // Optional: handle cases where not exactly two bits are set
      return result;
   };

   if (bdr_opt_case >= 1)
   {
      if (myid == 0)
      {
         std::cout << smesh->GetNE() << " elements in the mesh." << std::endl;
      }
      H1_FECollection fec_temp(mesh_poly_deg, dim);
      FiniteElementSpace fes_temp(smesh, &fec_temp);
      GridFunction attr_count_ser(&fes_temp);
      Array<int> attr_marker_ser;
      SetupSerialDofAttributes(attr_count_ser, attr_marker_ser);
      for (int i = 0; i < surf_mesh_attr.Size(); i++)
      {
         if (dim == 2)
         {
            // Mesh *meshsurf = SetupEdgeMesh2D(smesh, surf_mesh_attr[i]);
            Mesh *meshsurf = SetupEdgeMesh2D(smesh, attr_count_ser, attr_marker_ser, surf_mesh_attr[i]);
            if (myid == 0)
            {
               for (int e = 0; e < meshsurf->GetNE(); e++)
               { surf_measure_before(i) += meshsurf->GetElementVolume(e); }
            }
            surf_mesh_arr[i] = new ParMesh(MPI_COMM_WORLD, *meshsurf);
            delete meshsurf;
         }
         else if (dim == 3)
         {
            Mesh *meshsurf = SetupFaceMesh3D(smesh, surf_mesh_attr[i]);
            if (myid == 0)
            {
               for (int e = 0; e < meshsurf->GetNE(); e++)
               { surf_measure_before(i) += meshsurf->GetElementVolume(e); }
            }
            int *part = meshsurf->GeneratePartitioning(nranks, 0);
            surf_mesh_arr[i] = new ParMesh(MPI_COMM_WORLD, *meshsurf, part);
            delete meshsurf;
         }
      }
      for (int i = 0; i < surf_mesh_edge_attr.Size(); i++)
      {
         auto result = getTwoSetBits(surf_mesh_edge_attr[i]);
         int eattr1 = result.first;
         int eattr2 = result.second;
         Mesh *meshsurf = SetupEdgeMesh3D(smesh, attr_count_ser, attr_marker_ser, eattr1, eattr2);
         if (myid == 0)
         {
            for (int e = 0; e < meshsurf->GetNE(); e++)
            { surf_measure_before(surf_mesh_attr.Size() + i) += meshsurf->GetElementVolume(e); }
         }
         surf_mesh_arr[i+surf_mesh_attr.Size()] = new ParMesh(MPI_COMM_WORLD, *meshsurf);
         delete meshsurf;
      }
      for (int i = 0; i < surf_mesh_arr.Size(); i++)
      {
         {
            ostringstream mesh_name;
            mesh_name << "3D-face-edge-extracted" + std::to_string(i) + ".mesh";
            ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            surf_mesh_arr[i]->PrintAsOne(mesh_ofs);
         }
         if (visit)
         {
            VisItDataCollection dc(("amster-input-bdr_" + std::to_string(job_id) + "_" + std::to_string(i)).c_str(), surf_mesh_arr[i]);
            dc.SetFormat(DataCollection::SERIAL_FORMAT);
            dc.Save();
         }
         FindPointsGSLIB finder;
         finder.SetupSurf(*surf_mesh_arr[i], bbox_fac);
         Mesh *mesh_abb  = finder.GetBoundingBoxMeshSurf(0);
         Mesh *mesh_obb  = finder.GetBoundingBoxMeshSurf(1);
         if (myid==0 && visit)
         {
            VisItDataCollection dc0(("amster-input-bdr-aabb_" + std::to_string(job_id) + "_" + std::to_string(i)).c_str(), mesh_abb);
            dc0.SetFormat(DataCollection::SERIAL_FORMAT);
            dc0.Save();
            VisItDataCollection dc1(("amster-input-bdr-obb_" + std::to_string(job_id) + "_" + std::to_string(i)).c_str(), mesh_obb);
            dc0.SetFormat(DataCollection::SERIAL_FORMAT);
            dc1.Save();
         }
         finder.FreeData();
      }
      if (visit && myid == 0)
      {
         VisItDataCollection dc(("amster-input_" + std::to_string(job_id)).c_str(), mesh);
         dc.SetFormat(DataCollection::SERIAL_FORMAT);
         dc.Save();
      }
      {
         ostringstream mesh_name;
         mesh_name << "amster-input.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintAsSerial(mesh_ofs);
      }
   }

   delete mesh;

   // Define a finite element space on the mesh.
   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes(pmesh, &fec, dim);
   ParFiniteElementSpace spfes(pmesh, &fec);
   pmesh->SetNodalFESpace(&pfes);

   // Get the mesh nodes as a finite element grid function in fespace.
   ParGridFunction x(&pfes);
   pmesh->SetNodalGridFunction(&x);

   // Save the starting (prior to the optimization) mesh to a file.
   {
      ostringstream mesh_name;
      mesh_name << "amster_in.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(&pfes);
   x0 = x;
   ParGridFunction dxx = x0;

   // Setup visualization
   int vis_count = 0;
   VisItDataCollection dc(("amster_" + std::to_string(job_id)).c_str(), pmesh);
   dc.SetFormat(DataCollection::SERIAL_FORMAT);
   if (visit)
   {
      dc.SetCycle(vis_count);
      dc.Save();
   }

   // Metric.
   TMOP_QualityMetric *umetric = GetMetric(u_metric_id);
   TargetConstructor *utarget_c = GetTargetConstructor(u_target_id, x0);

   TMOP_QualityMetric *wcmetric = GetMetric(wc_metric_id);
   TargetConstructor *wctarget_c = GetTargetConstructor(wc_target_id, x0);

   TMOP_QualityMetric *metric = GetMetric(metric_id);
   TargetConstructor *target_c = GetTargetConstructor(target_id, x0);

   int det_order = dim*mesh_poly_deg-1;
   L2_FECollection fec_det(det_order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fespace_det(pmesh, &fec_det);
   ParGridFunction detgf(&fespace_det);
   GetDeterminantJacobianGF(pmesh, &detgf);

   L2_FECollection fec_pw(0, dim);
   ParFiniteElementSpace fespace_pw(pmesh, &fec_pw);
   ParGridFunction gf_pw(&fespace_pw);
   ParGridFunction gf_pw2(&fespace_pw);

   Vector detgf_lower, detgf_upper;
   int ref_factor = dim == 2 ? 10 : 8;
   GridFunction::PLBound *plb = nullptr;
   GridFunction::PLBound plbt = detgf.GetBounds(detgf_lower, detgf_upper,
                                                ref_factor);
   if (bound)
   {
      plb = &plbt;
   }

   double min_detA0, volume0, min_det_cur;
   GetMinDet(pmesh, x0, quad_order, min_detA0, volume0);
   min_det_cur = min_detA0;

   // Visualize the starting mesh.
   if (vis)
   {
      std::string title = "Initial mesh, mu_" + std::to_string(metric_id);
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh,
                        title.c_str(), 0, 0, 300);
   }

   // Untangle the mesh
   if (min_detA0 > 0.0)
   {
      if (myid == 0)
      {
         std::cout << "********************************\n";
         std::cout << "***Mesh is already untangled, no need to untangle.***\n";
      }
   }
   else
   {
      real_t min_detA_u0, min_muT_u0, max_muT_u0, avg_muT_u0, volume_u0;
      real_t min_detA_u, min_muT_u, max_muT_u, avg_muT_u, volume_u;
      GetMeshStats(pmesh, x0, umetric, utarget_c, quad_order,
                   min_detA_u0, min_muT_u0, max_muT_u0, avg_muT_u0, volume_u0);


      real_t dbounds = 1e-3;
      plbt.AdjustBounds(dbounds);
      detgf.GetBounds(detgf_lower, detgf_upper, plbt);
      plbt.AdjustBounds(-dbounds);
      double min_det_bound0 = detgf_lower.Min();

      // Visualize the starting mesh.
      if (vis && u_metric_id != metric_id)
      {
         std::string title = "Initial mesh, untangling mu_" +
                             std::to_string(u_metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *umetric, *utarget_c, *pmesh,
                           title.c_str(), 300, 0, 300);
      }
      if (myid == 0)
      {
         cout << "Minimum det J bound is " << min_det_bound0 << endl;
         cout << min_detA0 << endl;
      }

      // Untangle the mesh
      plbt.AdjustBounds(dbounds);
      Untangle(x, min_detA_u0, quad_order, u_metric_id, u_target_id,
               plb, &detgf, solver_iter, move_bnd, adapt_qp);
      plbt.AdjustBounds(-dbounds);
      {
         ostringstream mesh_name;
         mesh_name << "amster_untangled_out2.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintAsSerial(mesh_ofs);
      }

      if (vis)
      {
         std::string title = "After Untangling, mu_" +
                             std::to_string(u_metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *umetric, *utarget_c, *pmesh,
                           title.c_str(), 600, 0, 300);
      }

      // Average quality and worst-quality for the mesh.
      GetMeshStats(pmesh, x, metric, target_c, quad_order,
                   min_detA_u, min_muT_u, max_muT_u, avg_muT_u, volume_u);
      min_det_cur = min_detA_u;

      GetDeterminantJacobianGF(pmesh, &detgf);
      detgf.GetBounds(detgf_lower, detgf_upper, ref_factor);
      double min_det_bound = detgf_lower.Min();
      if (myid == 0)
      {
         cout << "\n*** Stats of original mesh and untangled mesh\n";
         cout << "Minimum det(J):       " << min_detA_u0 << " " <<
              min_detA_u << endl
              << "Minimum det(J) bound: " << min_det_bound0 << " " <<
              min_det_bound << endl
              << "Minimum muT:          " << min_muT_u0 << " " <<
              min_muT_u << endl
              << "Maximum muT:          " << max_muT_u0 << " " <<
              max_muT_u << endl
              << "Average muT:          " << avg_muT_u0 << " " <<
              avg_muT_u << endl;
      }

      // Visualize the mesh displacement.
      if (vis)
      {
         socketstream vis;
         dxx = x0;
         dxx -= x;
         common::VisualizeField(vis, "localhost", 19916, dxx,
                                "Displacements", 900, 0, 300, 300, "jRmclA");
      }

      if (visit)
      {
         dc.SetCycle(++vis_count);
         dc.Save();
      }
   }

   // Do regular mesh optimization
   if (min_det_cur > 0.0 && final_pass)
   {
      detgf.GetBounds(detgf_lower, detgf_upper, ref_factor);
      real_t min_detA_m0, min_muT_m0, max_muT_m0, avg_muT_m0, volume_m0;
      real_t min_detA_m, min_muT_m, max_muT_m, avg_muT_m, volume_m;
      GetMeshStats(pmesh, x, metric, target_c, quad_order,
                   min_detA_m0, min_muT_m0, max_muT_m0, avg_muT_m0, volume_m0);
      double min_det_bound0 = detgf_lower.Min();
      if (vis && min_detA0 > 0.0)
      {
         std::string title = "Initial mesh, mu_" +
                             std::to_string(metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh,
                           title.c_str(), 300, 600, 300);
      }

      MeshOptimizer meshopt(pmesh);
      ParGridFunction attr_count(&pfes);
      ParGridFunction attr_count_s(&spfes);
      Array<int> attr_marker;
      SetupDofAttributes(attr_count, attr_marker);
      SetupDofAttributes(attr_count_s, attr_marker);
      Array<int> aux_ess_dofs = IdentifyAuxiliaryEssentialDofs2(attr_count);
      Array<int> aux_ess_dofs_s = IdentifyAuxiliaryEssentialDofs2(attr_count_s);
      // Get Dofs for all faces/edges
      Array<Array<int> *> bdr_face_dofs(surf_mesh_attr.Size());
      for (int i = 0; i < bdr_face_dofs.Size(); i++)
      {
         // bdr_face_dofs[i] = GetBdrFaceDofsForAttr(attr_count_s, attr_marker,
                                                //   surf_mesh_attr[i], true);
         bdr_face_dofs[i] = GetDofMatchingBdrFaceAttributes(attr_count_s,
                                                            attr_marker,
                                                            surf_mesh_attr[i]);
      }

      Array<Array<int> *> bdr_edge_dofs(surf_mesh_edge_attr.Size());
      for (int i = 0; i < bdr_edge_dofs.Size(); i++)
      {
         auto result = getTwoSetBits(surf_mesh_edge_attr[i]);
         int eattr1 = result.first;
         int eattr2 = result.second;
         // bdr_edge_dofs[i] = GetBdrEdgeDofsForAttr(attr_count_s, attr_marker,
                                                //   eattr1, eattr2, true);
         bdr_edge_dofs[i] = GetDofMatchingBdrEdgeAttributes(attr_count_s,
                                                            attr_marker,
                                                            eattr1, eattr2);
         for (int j = 0; j < bdr_face_dofs.Size(); j++)
         {
            RemoveDofsFromBdrFaceDofs(*bdr_face_dofs[j], *bdr_edge_dofs[i]);
         }
         RemoveDofsFromBdrFaceDofs(*bdr_edge_dofs[i], aux_ess_dofs_s);
      }

      meshopt.Setup(x, &min_detA0, quad_order, metric_id, target_id,
                    plb, &detgf, solver_iter, move_bnd, surf_mesh_attr,
                    aux_ess_dofs, solver_type, adapt_qp);
      Array<FindPointsGSLIB *> finder_arr;
      if (bdr_opt_case >= 1)
      {
         for (int i = 0; i < surf_mesh_attr.Size(); i++)
         {
            FindPointsGSLIB *finder = new FindPointsGSLIB();
            finder_arr.Append(finder);
            finder->SetupSurf(*surf_mesh_arr[i], bbox_fac);
            finder->SetDistanceToleranceForPointsFoundOnBoundary(10);
            meshopt.SetupTangentialRelaxationForFacEdg(&pfes, bdr_face_dofs[i],
                                                       finder,
                                                       surf_mesh_arr[i]->GetNodes());
         }
         int noff = surf_mesh_attr.Size();
         for (int i = 0; i < surf_mesh_edge_attr.Size(); i++)
         {
            FindPointsGSLIB *finder = new FindPointsGSLIB();
            finder_arr.Append(finder);
            finder->SetupSurf(*surf_mesh_arr[i+noff], bbox_fac);
            finder->SetDistanceToleranceForPointsFoundOnBoundary(10);
            meshopt.SetupTangentialRelaxationForFacEdg(&pfes, bdr_edge_dofs[i],
                                                       finder, surf_mesh_arr[i+noff]->GetNodes());
         }
         meshopt.EnableTangentialRelaxation();
      }
      if (visit)
      {
         int vis_frequency = 2;
         if (bdr_opt_case == 1)
         {
            vis_frequency = 5;
         }
         if (bdr_opt_case == 5 || bdr_opt_case == 6)
         {
            vis_frequency = 100;
         }
         meshopt.EnableVisualization(&dc, vis_count, vis_frequency);
      }

      {
         ParGridFunction ess_identifier1(&pfes);
         ess_identifier1 = 0.0;
         for (int i = 0; i < aux_ess_dofs.Size(); i++)
         {
            ess_identifier1[aux_ess_dofs[i]] = 1.0;
         }
         ParGridFunction ess_identifier2(&spfes);
         ess_identifier2 = 0.0;
         for (int i = 0; i < bdr_face_dofs.Size(); i++)
         {
            for (int j = 0; j < bdr_face_dofs[i]->Size(); j++)
            {
               ess_identifier2[(*bdr_face_dofs[i])[j]] = 1.0;
            }
         }
         ParGridFunction ess_identifier3(&spfes);
         ess_identifier3 = 0.0;
         for (int i = 0; i < bdr_edge_dofs.Size(); i++)
         {
            for (int j = 0; j < bdr_edge_dofs[i]->Size(); j++)
            {
               ess_identifier3[(*bdr_edge_dofs[i])[j]] = 1.0;
            }
         }

         if (visit)
         {
            VisItDataCollection dc(("amsterbdr_" + std::to_string(job_id)).c_str(), pmesh);
            dc.SetFormat(DataCollection::SERIAL_FORMAT);
            dc.RegisterField("essn_dof_identifier", &ess_identifier1);
            dc.RegisterField("face_dof_identifier", &ess_identifier2);
            dc.RegisterField("edge_dof_identifier", &ess_identifier3);
            dc.SetCycle(0);
            dc.Save();
         }
      }

      meshopt.OptimizeNodes(x);

      {
         ostringstream mesh_name;
         mesh_name << "amster_final_out_" << std::to_string(job_id) << ".mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintAsSerial(mesh_ofs);
      }

      if (visit)
      {
         int vcount = meshopt.GetTMOPIntegrator()->GetVisCounter();
         dc.SetCycle(++vcount);
         dc.Save();
      }

      if (vis)
      {
         std::string title = "After mesh-optimization, mu_" +
                             std::to_string(metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh,
                           title.c_str(), 600, 0, 300);
      }

      // Visualize the mesh displacement.
      if (vis)
      {
         socketstream vis;
         dxx = x0;
         dxx -= x;
         common::VisualizeField(vis, "localhost", 19916, dxx,
                                "Displacements", 900, 0, 300, 300, "jRmclA");
      }

      // Average quality and worst-quality for the mesh.
      Array<int> *adapted_quad_order = nullptr;
      if (adapt_qp)
      {
         auto aqp= meshopt.GetTMOPIntegrator()->GetElementWiseAdaptedQuadOrder();
         adapted_quad_order = new Array<int>(aqp);
      }
      GetMeshStats(pmesh, x, metric, target_c, quad_order,
                   min_detA_m, min_muT_m, max_muT_m, avg_muT_m, volume_m,
                   adapted_quad_order);

      GetDeterminantJacobianGF(pmesh, &detgf);
      detgf.GetBounds(detgf_lower, detgf_upper, ref_factor);
      double min_det_bound = detgf_lower.Min();
      if (myid == 0)
      {
         cout << "\n*** Stats of final mesh optimization pass\n";
         cout << "Minimum det(J):       " << min_detA_m0 << " " <<
              min_detA_m << endl
              << "Minimum det(J) bound: " << min_det_bound0 << " " <<
              min_det_bound << endl
              << "Minimum muT:          " << min_muT_m0 << " " <<
              min_muT_m << endl
              << "Maximum muT:          " << max_muT_m0 << " " <<
              max_muT_m << endl
              << "Average muT:          " << avg_muT_m0 << " " <<
              avg_muT_m << endl
              << "Integral muT:         " << meshopt.init_energy << " " <<
              meshopt.final_energy << endl;
      }

      if (bdr_opt_case >= 1)
      {
         Vector surf_measure_after(surf_mesh_attr.Size() + surf_mesh_edge_attr.Size());
         surf_measure_after = 0.0;

         Mesh smesh_post = pmesh->GetSerialMesh(0);
         if (myid == 0)
         {
            smesh_post.SetCurvature(mesh_poly_deg, false, -1, 0);
            H1_FECollection fec_post(mesh_poly_deg, dim);
            FiniteElementSpace fes_post(&smesh_post, &fec_post);
            GridFunction attr_count_post(&fes_post);
            Array<int> attr_marker_post;
            SetupSerialDofAttributes(attr_count_post, attr_marker_post);

            for (int i = 0; i < surf_mesh_attr.Size(); i++)
            {
               Mesh *ms = (dim == 2)
                  ? SetupEdgeMesh2D(&smesh_post, attr_count_post, attr_marker_post, surf_mesh_attr[i])
                  : SetupFaceMesh3D(&smesh_post, surf_mesh_attr[i]);
               for (int e = 0; e < ms->GetNE(); e++)
               { surf_measure_after(i) += ms->GetElementVolume(e); }
               delete ms;
            }
            for (int i = 0; i < surf_mesh_edge_attr.Size(); i++)
            {
               auto result = getTwoSetBits(surf_mesh_edge_attr[i]);
               int eattr1 = result.first;
               int eattr2 = result.second;
               Mesh *ms = SetupEdgeMesh3D(&smesh_post, attr_count_post, attr_marker_post, eattr1, eattr2);
               for (int e = 0; e < ms->GetNE(); e++)
               { surf_measure_after(surf_mesh_attr.Size() + i) += ms->GetElementVolume(e); }
               delete ms;
            }

            const char *label = (dim == 3) ? "Area" : "Length";
            cout << "\n*** Surface " << label << " before/after tangential relaxation\n";
            cout << std::setw(20) << "surface"
                 << std::setw(22) << "before"
                 << std::setw(22) << "after"
                 << std::setw(22) << "delta"
                 << std::setw(22) << "delta %" << "\n";
            cout.setf(std::ios::scientific);
            cout << std::setprecision(12);
            for (int i = 0; i < surf_mesh_attr.Size(); i++)
            {
               double vi = surf_measure_before(i);
               double vf = surf_measure_after(i);
               cout << std::setw(20) << ("bdr attr " + std::to_string(surf_mesh_attr[i]))
                    << std::setw(22) << vi
                    << std::setw(22) << vf
                    << std::setw(22) << (vf - vi)
                    << std::setw(22) << ((vf - vi) / vi * 100.0) << "\n";
            }
            for (int i = 0; i < surf_mesh_edge_attr.Size(); i++)
            {
               auto result = getTwoSetBits(surf_mesh_edge_attr[i]);
               double vi = surf_measure_before(surf_mesh_attr.Size() + i);
               double vf = surf_measure_after(surf_mesh_attr.Size() + i);
               cout << std::setw(20) << ("edge " + std::to_string(result.first) + "/" + std::to_string(result.second))
                    << std::setw(22) << vi
                    << std::setw(22) << vf
                    << std::setw(22) << (vf - vi)
                    << std::setw(22) << ((vf - vi) / vi * 100.0) << "\n";
            }
            cout.unsetf(std::ios::scientific);
         }
      }

      // Vis ratio of min det at quad point vs min bound
      GetElementWiseMinDetAtQPs(pmesh, x, quad_order, gf_pw);
      ParGridFunction gf_pw_b(&fespace_pw);
      detgf.GetElementBounds(gf_pw_b, detgf_upper, ref_factor);
      for (int e = 0; e < pmesh->GetNE(); e++)
      {
         double ratio = gf_pw(e)/gf_pw_b(e);
         gf_pw(e) = gf_pw_b(e) < 0.0 ? fabs(ratio) : ratio;
      }

      if (vis)
      {
         socketstream vis;
         common::VisualizeField(vis, "localhost", 19916, gf_pw,
                                "min det ratio", 1200, 600, 300, 300, "jRmclA");
      }
      if (plb)
      {
         if (adapt_qp)
         {
            meshopt.GetTMOPIntegrator()->GetElementWiseAdaptedQuadOrder(gf_pw2);
         }
         else {
            gf_pw2 = quad_order*1.0;
         }

         if (vis)
         {
            socketstream vis;
            common::VisualizeField(vis, "localhost", 19916, gf_pw2,
                                   "quad_order", 1500, 600, 300, 300, "jRmclAppppp");
         }
      }
      if (visit)
      {
         VisItDataCollection dcq(("amster-det-ratio_" + std::to_string(job_id)).c_str(), pmesh);
         dcq.RegisterField("det_ratio", &gf_pw);
         dcq.RegisterField("quad_order", &gf_pw2);
         dcq.SetFormat(DataCollection::SERIAL_FORMAT);
         dcq.Save();
      }

      // output invalid elements only
      if (true)
      {
         for (int e = 0; e < pmesh->GetNE(); e++)
         {
            pmesh->SetAttribute(e, 1);
            if (gf_pw_b(e) < 0.0)
            {
               pmesh->SetAttribute(e, 2);
            }
         }
         pmesh->SetAttributes();
         // Make ParSubMesh with elements of attribute 2
         Array<int> cond_attr;
         cond_attr.Append(2);
         ParSubMesh psm(ParSubMesh::CreateFromDomain(*pmesh, cond_attr));

         // Print as serial mesh
         ostringstream mesh_name;
         mesh_name << "amster_invalid_elements.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         psm.PrintAsSerial(mesh_ofs);

         auto pvdc = new VisItDataCollection(("amster-invalid-elements_" + std::to_string(job_id)).c_str(), &psm);
         pvdc->SetFormat(DataCollection::SERIAL_FORMAT);
         pvdc->Save();
         delete pvdc;
      }
   }


   delete target_c;
   delete metric;
   delete utarget_c;
   delete umetric;
   delete wctarget_c;
   delete wcmetric;
   delete pmesh;

   return 0;
}

void Untangle(ParGridFunction &x, double min_detA, int quad_order,
              int metric_id, int target_id, GridFunction::PLBound *plb,
              ParGridFunction *detgf, int solver_iter, bool move_bnd,
              bool adapt_qp)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();
   // const int dim = pfes.GetParMesh()->Dimension();

   if (pfes.GetMyRank() == 0) { cout << "*** \nUntangle Phase\n***\n"; }
   ParMesh *pmesh = pfes.GetParMesh();
   const int dim = pfes.GetMesh()->Dimension();

   // The metrics work in terms of det(T).
   const DenseMatrix &Wideal =
      Geometries.GetGeomToPerfGeomJac(pfes.GetFE(0)->GetGeomType());
   // Slightly below the minimum to avoid division by 0.
   double min_detT = min_detA / Wideal.Det();

   // Metric / target / integrator.
   auto btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Shifted;
   auto wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::None;
   TMOP_QualityMetric *metric = GetMetric(metric_id);
   real_t min_det_threshold = 1e-3;
   double alpha = 1.5;
   double detT_ep = 1e-3;
   double muT_ep = 0.0;
   double bound = false;
   if (plb)
   {
      alpha = 1.0;
      detT_ep = 1e-3;
      bound = true;
   }
   TMOP_WorstCaseUntangleOptimizer_Metric u_metric(*metric, 1.0, 1.0, 1,
                                                   alpha, detT_ep, muT_ep,
                                                   btype, wctype,
                                                   min_det_threshold);
   TargetConstructor *target_c = GetTargetConstructor(target_id, x);
   auto tmop_integ = new TMOP_Integrator(&u_metric, target_c, nullptr);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);
   if (plb)
   {
      tmop_integ->SetPLBoundsForDeterminant(plb, detgf);
      int max_quad_order = 100;
      if (adapt_qp)
      {
         tmop_integ->EnableAdaptiveIntegrationRule(pfes.GetNE(), quad_order,
         max_quad_order);
      }
   }
   tmop_integ->ComputeUntangleMetricQuantiles(x, pfes);
   // tmop_integ->EnableFiniteDifferences(x);

   // Nonlinear form.
   ParNonlinearForm nlf(&pfes);
   nlf.AddDomainIntegrator(tmop_integ);

   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   if (!move_bnd)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      nlf.SetEssentialBC(ess_bdr);
   }
   else
   {
      int n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfes.GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         if (attr == 1 || attr == 2 || (attr == 3 && dim == 3)) { n += nd; }
         if (attr >= 4 || (dim == 2 && attr == 3)) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n);
      n = 0;
      Array<int> vdofs;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfes.GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfes.GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3 && dim == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr >= 4 || (attr == 3 && dim == 2)) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      nlf.SetEssentialVDofs(ess_vdofs);
   }

   // Linear solver.
   MINRESSolver minres(pfes.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   minres.SetPrintLevel(minres_pl.FirstAndLast().Summary());

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes.GetComm(), ir);
   solver.SetIntegrationRules(IntRulesLo, quad_order);
   solver.SetOperator(nlf);
   solver.SetPreconditioner(minres);
   solver.SetMinDetPtr(&min_detT);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(1e-12);
   solver.SetAbsTol(0.0);
   solver.SetMinimumDeterminantThreshold(min_det_threshold);
   if (plb) { solver.SetDeterminantBound(true); }
   IterativeSolver::PrintLevel newton_pl;
   solver.SetPrintLevel(newton_pl.Iterations().Summary());
   if (plb && adapt_qp)
   {
      solver.EnableQuadOrderIncrement(8, 5.0); /// new
   }

   const real_t init_energy = nlf.GetParGridFunctionEnergy(x);

   // Optimize.
   x.SetTrueVector();
   Vector b;
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   const real_t final_energy = nlf.GetParGridFunctionEnergy(x);
   if (pfes.GetMyRank() == 0)
   {
      cout << "Initial energy: " << init_energy << endl
           << "Final energy:   " << final_energy << endl;
   }

   delete target_c;
   delete metric;

   return;
}


void Interpolate(const ParGridFunction &src, const Array<int> &y_fixed_marker,
                 ParGridFunction &y)
{
   const int dim = y.ParFESpace()->GetVDim();
   Array<int> dofs;
   for (int e = 0; e < y.ParFESpace()->GetNE(); e++)
   {
      const IntegrationRule &ir = y.ParFESpace()->GetFE(e)->GetNodes();
      const int ndof = ir.GetNPoints();
      y.ParFESpace()->GetElementVDofs(e, dofs);

      for (int i = 0; i < ndof; i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         for (int d = 0; d < dim; d++)
         {
            if (y_fixed_marker[dofs[d*ndof + i]] == 0)
            {
               y(dofs[d*ndof + i]) = src.GetValue(e, ip, d+1);
            }
         }
      }
   }
}

IntegrationRule PermuteIR(const IntegrationRule &irule,
                          const Array<int> ordering)
{
   const int np = irule.GetNPoints();
   MFEM_VERIFY(np == ordering.Size(), "Invalid permutation size");
   IntegrationRule ir(np);
   ir.SetOrder(irule.GetOrder());

   for (int i = 0; i < np; i++)
   {
      IntegrationPoint &ip_new = ir.IntPoint(i);
      const IntegrationPoint &ip_old = irule.IntPoint(ordering[i]);
      ip_new.Set(ip_old.x, ip_old.y, ip_old.z, ip_old.weight);
   }

   return ir;
}

void GetDeterminantJacobianGF(ParMesh *mesh, ParGridFunction *detgf)
{
   int dim = mesh->Dimension();
   FiniteElementSpace *fespace = detgf->FESpace();
   Array<int> dofs;

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const FiniteElement *fe = fespace->GetFE(e);
      const IntegrationRule ir = fe->GetNodes();
      ElementTransformation *transf = mesh->GetElementTransformation(e);
      DenseMatrix Jac(fe->GetDim());
      const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
                                      (fe);
      const Array<int> &irordering = nfe->GetLexicographicOrdering();
      IntegrationRule ir2 = irordering.Size() ?
                            PermuteIR(ir, irordering) :
                            ir;

      Vector detvals(ir2.GetNPoints());
      Vector loc(dim);
      for (int q = 0; q < ir2.GetNPoints(); q++)
      {
         IntegrationPoint ip = ir2.IntPoint(q);
         transf->SetIntPoint(&ip);
         transf->Transform(ip, loc);
         Jac = transf->Jacobian();
         detvals(q) = Jac.Weight();
      }

      fespace->GetElementDofs(e, dofs);
      if (irordering.Size())
      {
         for (int i = 0; i < dofs.Size(); i++)
         {
            (*detgf)(dofs[i]) = detvals(irordering[i]);
         }
      }
      else
      {
         detgf->SetSubVector(dofs, detvals);
      }
   }
}

void TransferLowToHigh(const ParGridFunction &l, ParGridFunction &h)
{
   Array<int> h_ess_marker(h.Size());
   h_ess_marker = 0;

   Interpolate(l, h_ess_marker, h);
}

void TransferHighToLow(const ParGridFunction &h, ParGridFunction &l)
{
   Array<int> l_ess_vdof_marker(l.Size());
   l_ess_vdof_marker = 0;
   // wrong.
   // PRefinementTransferOperator transfer(*l.ParFESpace(), *h.ParFESpace());
   // transfer.MultTranspose(h, l);

   // Projects, doesn't interpolate.
   //l.ProjectGridFunction(h);

   Interpolate(h, l_ess_vdof_marker, l);
}
