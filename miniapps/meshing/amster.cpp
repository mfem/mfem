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
//      mpirun -np 4 amster -m jagged.mesh -o 2 -qo 4 -no-wc -no-fit
//    2D untangling + worst-case:
//      mpirun -np 4 amster -m amster_q4warp.mesh -o 2 -qo 6 -no-fit
//    2D fitting:
//      mpirun -np 6 amster -m amster_q4warp.mesh -rs 1 -o 3 -no-wc -amr 7
//
//    2D orders prec:
//      mpirun -np 6 amster -m ../../data/star.mesh -rs 0 -o 1 -no-wc -amr 7 -vis
//
//    3D untangling:
//      mpirun -np 6 amster -m ../../../mfem_data/cube-holes-inv.mesh -o 3 -qo 4 -no-wc -no-fit

// Some new sample runs:
// make amster -j4 && mpirun -np 4 amster -m jagged.mesh -o 2 -qo 8 -vis -rs 0 -umid 66 -ni 1000 -no-wc -wcmid 66 -utid 1 -wctid 2 -visit -no-final

// Blade
// make amster -j4 && mpirun -np 4 amster -m blade.mesh -o 4 -qo 8 -vis -rs 0 -no-wc -mid 2 -tid 1 -bdropt 1 -visit -ni 1000 -bnd
// Blade + increase quad_order
// make amster -j4 && mpirun -np 4 amster -m blade.mesh -o 4 -qo 24 -vis -rs 0 -no-wc -mid 2 -tid 1 -bdropt 1 -visit -ni 1000 -bnd
// Blade + increase quad_order + no bound on Jacobian
// make amster -j4 && mpirun -np 4 amster -m blade.mesh -o 4 -qo 24 -vis -rs 0 -no-wc -mid 2 -tid 1 -bdropt 1 -visit -ni 1000 -bnd -no-bound

// Ale tangled - curvilinear right and top boundaries
// make amster -j4 && mpirun -np 3 amster -m aletangled.mesh -o 2 -qo 8 -vis -rs 0 -umid 4 -no-wc -wcmid 66 -utid 1 -wctid 2 -mid 80 -tid 2 -bdropt 2 -visit -ni 5000 -bnd
// Ale tangled - rotated square hole
// make amster -j4 && mpirun -np 3 amster -m Laghos_800_mesh -o 2 -qo 8 -vis -rs 0 -umid 4 -no-wc -wcmid 66 -utid 1 -wctid 2 -mid 80 -tid 2 -bdropt 3 -visit -ni 1000 -bnd


#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"
#include "amster.hpp"

using namespace mfem;
using namespace std;

void TransferLowToHigh(const ParGridFunction &l, ParGridFunction &h);
void TransferHighToLow(const ParGridFunction &h, ParGridFunction &l);

void Untangle(ParGridFunction &x, double min_detA, int quad_order,
              int metric_id, int target_id, GridFunction::PLBound *plb,
              ParGridFunction *detgf, int solver_iter, bool move_bnd);
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
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "jagged.mesh";
   int rs_levels         = 0;
   int mesh_poly_deg     = 2;
   bool worst_case       = true;
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

   // Parse command-line input file.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&worst_case, "-wc", "--worst-case",
                  "-no-wc", "--no-worst-case",
                  "Enable worst case optimization step.");
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
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   const int dim = pmesh->Dimension();

   // Setup edge-mesh for tangential relaxation
   Array<ParMesh *> surf_mesh_arr;
   Array<int> surf_mesh_attr;
   if (bdr_opt_case == 1)
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   else if (bdr_opt_case == 2)
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(2);
      surf_mesh_arr.SetSize(2);
      surf_mesh_attr[0] = 3;
      surf_mesh_attr[1] = 4;
   }
   else if (bdr_opt_case == 3)
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(4);
      surf_mesh_arr.SetSize(4);
      surf_mesh_attr[0] = 3;
      surf_mesh_attr[1] = 4;
      surf_mesh_attr[2] = 5;
      surf_mesh_attr[3] = 6;
   }
   int bbox_fac = 2.0;
   if (bdr_opt_case >= 1)
   {
      if (!mesh->GetNodes())
      {
         mesh->SetCurvature(mesh_poly_deg, -1, -1, 0);
      }
      for (int i = 0; i < surf_mesh_attr.Size(); i++)
      {
         Mesh *meshsurf = SetupEdgeMesh(mesh, surf_mesh_attr[i]);
         surf_mesh_arr[i] = new ParMesh(MPI_COMM_WORLD, *meshsurf);
         delete meshsurf;
         {
            ostringstream mesh_name;
            mesh_name << "edge_extracted" + std::to_string(i) + ".mesh";
            ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            surf_mesh_arr[i]->PrintAsOne(mesh_ofs);
         }
         if (visit)
         {
            VisItDataCollection dc("amster-input-bdr"+ std::to_string(i), surf_mesh_arr[i]);
            dc.SetFormat(DataCollection::SERIAL_FORMAT);
            dc.Save();
         }
         FindPointsGSLIB finder;
         finder.SetupSurf(*surf_mesh_arr[i], bbox_fac);
         Mesh *mesh_abb  = finder.GetBoundingBoxMeshSurf(0);
         Mesh *mesh_obb  = finder.GetBoundingBoxMeshSurf(1);
         if (myid==0 && visit)
         {
            VisItDataCollection dc0("amster-input-bdr-aabb"+ std::to_string(i), mesh_abb);
            dc0.SetFormat(DataCollection::SERIAL_FORMAT);
            dc0.Save();
            VisItDataCollection dc1("amster-input-bdr-obb"+ std::to_string(i), mesh_obb);
            dc1.SetFormat(DataCollection::SERIAL_FORMAT);
            dc1.Save();
         }
         finder.FreeData();
      }
      if (visit && myid == 0)
      {
         VisItDataCollection dc("amster-input", mesh);
         dc.SetFormat(DataCollection::SERIAL_FORMAT);
         dc.Save();
      }
   }

   delete mesh;

   // Define a finite element space on the mesh.
   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes(pmesh, &fec, dim);
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
      pmesh->PrintAsOne(mesh_ofs);
   }

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(&pfes);
   x0 = x;

   // Setup visualization
   int vis_count = 0;
   VisItDataCollection dc("amster", pmesh);
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

   Vector detgf_lower, detgf_upper;
   int ref_factor = 5;
   GridFunction::PLBound *plb = nullptr;
   GridFunction::PLBound plbt = detgf.GetBounds(detgf_lower, detgf_upper,
                                                ref_factor);
   if (bound) {
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
      double min_det_bound0 = detgf_lower.Min();

      // Visualize the starting mesh.
      if (vis && u_metric_id != metric_id)
      {
         std::string title = "Initial mesh, untangling mu_" +
                             std::to_string(u_metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *umetric, *utarget_c, *pmesh,
                           title.c_str(), 300, 0, 300);
      }

      // Untangle the mesh
      Untangle(x, min_detA_u0, quad_order, u_metric_id, u_target_id,
               plb, &detgf, solver_iter, move_bnd);

      {
         ostringstream mesh_name;
         mesh_name << "amster_untangled_out.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintAsOne(mesh_ofs);
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
         x0 -= x;
         common::VisualizeField(vis, "localhost", 19916, x0,
                                "Displacements", 900, 0, 300, 300, "jRmclA");
      }

      if (visit)
      {
         dc.SetCycle(++vis_count);
         dc.Save();
      }
   }

   // Worst case optimization
   if (!worst_case)
   {
      if (myid == 0)
      {
         std::cout << "********************************\n";
         std::cout << "*** Worst case optimization not requested.***\n";
      }
   }
   else
   {
      *plb = detgf.GetBounds(detgf_lower, detgf_upper, ref_factor);
      real_t min_detA_wc0, min_muT_wc0, max_muT_wc0, avg_muT_wc0, volume_wc0;
      real_t min_detA_wc, min_muT_wc, max_muT_wc, avg_muT_wc, volume_wc;
      GetMeshStats(pmesh, x0, wcmetric, wctarget_c, quad_order,
                   min_detA_wc0, min_muT_wc0, max_muT_wc0, avg_muT_wc0, volume_wc0);
      double min_det_bound0 = detgf_lower.Min();


      // Visualize the starting mesh.
      if (vis &&
          ((wc_metric_id != u_metric_id && min_detA0 <= 0.0) ||
           (wc_metric_id != metric_id && min_detA0 > 0.0)))
      {
         std::string title = "Initial mesh, worst case mu_" +
                             std::to_string(wc_metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *wcmetric, *wctarget_c, *pmesh,
                           title.c_str(), 300, 300, 300);
      }

      WorstCaseOptimize(x, quad_order, metric_id, wc_target_id, plb, &detgf,
                        solver_iter, min_detA_wc0);

      {
         ostringstream mesh_name;
         mesh_name << "amster_worst_case.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintAsOne(mesh_ofs);
      }

      if (vis)
      {
         std::string title = "After worst-case, mu_" +
                             std::to_string(wc_metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *wcmetric, *wctarget_c, *pmesh,
                           title.c_str(), 600, 300, 300);
      }

      // Average quality and worst-quality for the mesh.
      GetMeshStats(pmesh, x, wcmetric, wctarget_c, quad_order,
                   min_detA_wc, min_muT_wc, max_muT_wc, avg_muT_wc, volume_wc);
      min_det_cur = min_detA_wc;

      GetDeterminantJacobianGF(pmesh, &detgf);
      detgf.GetBounds(detgf_lower, detgf_upper, ref_factor);
      double min_det_bound = detgf_lower.Min();
      if (myid == 0)
      {
         cout << "\n*** Stats of input mesh and worst-case optimized mesh\n";
         cout << "Minimum det(J):       " << min_detA_wc0 << " " <<
              min_detA_wc << endl
              << "Minimum det(J) bound: " << min_det_bound0 << " " <<
              min_det_bound << endl
              << "Minimum muT:          " << min_muT_wc0 << " " <<
              min_muT_wc << endl
              << "Maximum muT:          " << max_muT_wc0 << " " <<
              max_muT_wc << endl
              << "Average muT:          " << avg_muT_wc0 << " " <<
              avg_muT_wc << endl;
      }

      // Visualize the mesh displacement.
      if (vis)
      {
         socketstream vis;
         x0 -= x;
         common::VisualizeField(vis, "localhost", 19916, x0,
                                "Displacements", 900, 300, 300, 300, "jRmclA");
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
      if (vis && min_detA0 > 0.0 && worst_case)
      {
         std::string title = "Initial mesh, mu_" +
                             std::to_string(metric_id);
         vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh,
                           title.c_str(), 300, 600, 300);
      }

      MeshOptimizer meshopt(pmesh);
      Array<int> ess_vdofs_aux;
      Array<int> aux_ess_dofs = IdentifyAuxiliaryEssentialDofs(&pfes);

      meshopt.Setup(x, &min_detA0, quad_order, metric_id, target_id,
                    plb, &detgf, solver_iter, move_bnd, surf_mesh_attr,
                    aux_ess_dofs);
      Array<FindPointsGSLIB *> finder_arr;
      if (bdr_opt_case == 1)
      {
         FindPointsGSLIB *finder = new FindPointsGSLIB();
         finder_arr.Append(finder);
         finder->SetupSurf(*surf_mesh_arr[0], bbox_fac);
         finder->SetDistanceToleranceForPointsFoundOnBoundary(10);
         meshopt.SetupTangentialRelaxationFor2DEdge(&pfes, surf_mesh_attr[0],
                                                    finder,
                                                    surf_mesh_arr[0]->GetNodes());
         meshopt.EnableTangentialRelaxation();
      }
      else if (bdr_opt_case == 2 || bdr_opt_case == 3)
      {
         for (int i = 0; i < surf_mesh_attr.Size(); i++)
         {
            FindPointsGSLIB *finder = new FindPointsGSLIB();
            finder_arr.Append(finder);
            finder->SetupSurf(*surf_mesh_arr[i], bbox_fac);
            finder->SetDistanceToleranceForPointsFoundOnBoundary(10);
            meshopt.SetupTangentialRelaxationFor2DEdge(&pfes, surf_mesh_attr[i],
                                                       finder,
                                                       surf_mesh_arr[i]->GetNodes());
         }
         meshopt.EnableTangentialRelaxation();
      }
      if (visit)
      {
         int vis_frequency = 100;
         if (bdr_opt_case == 1)
         {
            vis_frequency = 10;
         }
         meshopt.EnableVisualization(&dc, vis_count, vis_frequency);
      }
      meshopt.OptimizeNodes(x);

      {
         ostringstream mesh_name;
         mesh_name << "amster_final_out.mesh";
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->PrintAsOne(mesh_ofs);
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
                           title.c_str(), 600, 600, 300);
      }

      // Visualize the mesh displacement.
      if (vis)
      {
         socketstream vis;
         x0 -= x;
         common::VisualizeField(vis, "localhost", 19916, x0,
                                "Displacements", 900, 600, 300, 300, "jRmclA");
      }

      // Average quality and worst-quality for the mesh.
      GetMeshStats(pmesh, x, metric, target_c, quad_order,
                   min_detA_m, min_muT_m, max_muT_m, avg_muT_m, volume_m);

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
              avg_muT_m << endl;
      }
   }

   delete target_c;
   delete metric;
   delete pmesh;

   return 0;
}

void Untangle(ParGridFunction &x, double min_detA, int quad_order,
              int metric_id, int target_id, GridFunction::PLBound *plb,
              ParGridFunction *detgf, int solver_iter, bool move_bnd)
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
   real_t min_det_threshold = 0.01;
   TMOP_WorstCaseUntangleOptimizer_Metric u_metric(*metric, 1.0, 1.0, 1, 1.0,
                                                   1e-3, 0.001,
                                                   btype, wctype, true,
                                                   min_det_threshold);
   TargetConstructor *target_c = GetTargetConstructor(target_id, x);
   auto tmop_integ = new TMOP_Integrator(&u_metric, target_c, nullptr);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);
   if (plb)
   {
      tmop_integ->SetPLBoundsForDeterminant(plb, detgf);
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

void WorstCaseOptimize(ParGridFunction &x, int quad_order,
                       int metric_id, int target_id,
                       GridFunction::PLBound *plb,
                       ParGridFunction *detgf, int solver_iter,
                       double &min_det)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();
   // const int dim = pfes.GetParMesh()->Dimension();

   if (pfes.GetMyRank() == 0) { cout << "*** \nWorst Quality Phase\n***\n"; }

   // Metric / target / integrator.
   auto btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::None;
   auto wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::Beta;
   TMOP_QualityMetric *metric = GetMetric(metric_id);
   TMOP_WorstCaseUntangleOptimizer_Metric u_metric(*metric, 1.0, 1.0, 2, 1.5,
                                                   0.001, 0.001,
                                                   btype, wctype);
   TargetConstructor *target_c = GetTargetConstructor(target_id, x);
   auto tmop_integ = new TMOP_Integrator(&u_metric, target_c, nullptr);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);
   if (plb) { tmop_integ->SetPLBoundsForDeterminant(plb, detgf); }
   tmop_integ->ComputeUntangleMetricQuantiles(x, pfes);

   // Nonlinear form.
   ParNonlinearForm nlf(&pfes);
   nlf.AddDomainIntegrator(tmop_integ);

   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

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
   solver.EnableWorstCaseOptimization();
   solver.SetPreconditioner(minres);
   solver.SetMinDetPtr(&min_det);
   solver.SetMaxIter(1000);
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0.0);
   if (plb) { solver.SetDeterminantBound(true); }
   IterativeSolver::PrintLevel newton_pl;
   solver.SetPrintLevel(newton_pl.Iterations().Summary());

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
