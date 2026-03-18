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
// Compile with: make tangential-relaxation
//
// Sample runs:
// Blade - no bound on Jacobian + tangential relaxation
// make tangential-relaxation -j4 && mpirun -np 4 tangential-relaxation -m blade.mesh -o 4 -qo 8 -vis -rs 0 -mid 2 -tid 1  -ni 400 -st 1 -no-bound -bnd

// Blade - bound on Jacobian + tangential relaxation
// make tangential-relaxation -j4 && mpirun -np 4 tangential-relaxation -m blade.mesh -o 4 -qo 16 -vis -rs 0 -mid 2 -tid 1  -ni 400 -st 1 -bnd -bdropt 1 -bound

// Ale tangled - curvilinear right and top boundaries
// mpirun -np 3 tangential-relaxation -m 2Dmorphedsquare.mesh -o 2 -qo 8 -vis -mid 80 -tid 2 -bdropt 2  -ni 400 -bnd -bound

// circular hole
// make tangential-relaxation -j4 && mpirun -np 3 tangential-relaxation -m 2Dcircularhole.mesh -o 2 -qo 8 -vis -rs 0 -mid 80 -tid 2 -bdropt 4 -ni 200 -bnd -bound

// 3D
// make tangential-relaxation -j4 && mpirun -np 10 tangential-relaxation -m kershaw_laghos_morphed_2.mesh -o 2 -qo 8 -vis -rs 0 -mid 301 -tid 1 -bdropt 8  -ni 500 -bnd -no-bound


#include "mfem.hpp"
#include "tangential-relaxation.hpp"

using namespace mfem;
using namespace std;

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

   int rs_levels         = 0;
   int mesh_poly_deg     = 2;
   int solver_iter       = 50;
   int quad_order        = 8;
   int metric_id         = 2;
   int target_id         = 1;
   int bdr_opt_case      = 1;
   bool vis              = false;
   bool move_bnd         = false;
   bool final_pass       = true;
   bool bound            = true;
   int solver_type       = 1;
   bool transform        = true;
   bool adapt_qp         = false;

   // Parse command-line input file.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");

   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric 1/2/50/58 in 2D:\n\t");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Mesh optimization metric 1/2/3 in 2D:\n\t");
   args.AddOption(&vis, "-vis", "--vis", "-no-vis", "--no-vis",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&bdr_opt_case, "-bdropt", "--bdr-opt",
                  "Boundary attribute for tangential relaxation:\n\t");
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
   else if (bdr_opt_case == 4) // Ale tangled - rotated circular hole
   {
      MFEM_VERIFY(dim == 2,"Only 2D meshes supported for tangential relaxation.");
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   else if (bdr_opt_case == 8) // 3D case - kershaw
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
   double bbox_fac = 2.0; //2.0;

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
         std::cout << mesh->GetNE() << " elements in the mesh." << std::endl;
      }
      H1_FECollection fec_temp(mesh_poly_deg, dim);
      FiniteElementSpace fes_temp(mesh, &fec_temp);
      GridFunction attr_count_ser(&fes_temp);
      Array<int> attr_marker_ser;
      SetupSerialDofAttributes(attr_count_ser, attr_marker_ser);
      for (int i = 0; i < surf_mesh_attr.Size(); i++)
      {
         if (dim == 2)
         {
            Mesh *meshsurf = SetupEdgeMesh2D(mesh, attr_count_ser, attr_marker_ser,
                                             surf_mesh_attr[i]);
            surf_mesh_arr[i] = new ParMesh(MPI_COMM_WORLD, *meshsurf);
            delete meshsurf;
         }
         else if (dim == 3)
         {
            Mesh *meshsurf = SetupFaceMesh3D(mesh, surf_mesh_attr[i]);
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
         Mesh *meshsurf = SetupEdgeMesh3D(mesh, attr_count_ser, attr_marker_ser, eattr1,
                                          eattr2);
         surf_mesh_arr[i+surf_mesh_attr.Size()] = new ParMesh(MPI_COMM_WORLD, *meshsurf);
         delete meshsurf;
      }
      for (int i = 0; i < surf_mesh_arr.Size(); i++)
      {
         {
            ostringstream mesh_name;
            mesh_name << "bdr-extracted" + std::to_string(i) + ".mesh";
            ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            surf_mesh_arr[i]->PrintAsOne(mesh_ofs);
         }
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
   x.SetTrueVector();

   // Save the starting (prior to the optimization) mesh to a file.
   {
      ostringstream mesh_name;
      mesh_name << "tangential-in.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(&pfes);
   x0 = x;

   // Metric.
   TMOP_QualityMetric *metric = GetMetric(metric_id);
   TargetConstructor *target_c = GetTargetConstructor(target_id, x0);
   auto detgf = pmesh->GetJacobianDeterminantGF();

   Vector detgf_lower, detgf_upper;
   int ref_factor = dim == 2 ? 10 : 8;
   PLBound *plb = nullptr;
   PLBound plbt = detgf->GetBounds(detgf_lower, detgf_upper,
                                   ref_factor);
   if (bound)
   {
      plb = &plbt;
   }

   // Visualize the starting mesh.
   if (vis)
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, x0,
                             "Initial mesh", 600, 0, 300, 300, "jRmlAe");
   }

   MeshOptimizer meshopt(pmesh);
   ParGridFunction attr_count(&pfes);
   ParGridFunction attr_count_s(&spfes);
   Array<int> attr_marker;
   SetupDofAttributes(attr_count, attr_marker);
   SetupDofAttributes(attr_count_s, attr_marker);
   Array<int> aux_ess_dofs = IdentifyAuxiliaryEssentialDofs(attr_count);
   Array<int> aux_ess_dofs_s = IdentifyAuxiliaryEssentialDofs(attr_count_s);
   // Get Dofs for all faces/edges
   Array<Array<int> *> bdr_face_dofs(surf_mesh_attr.Size());
   for (int i = 0; i < bdr_face_dofs.Size(); i++)
   {
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
      bdr_edge_dofs[i] = GetDofMatchingBdrEdgeAttributes(attr_count_s,
                                                         attr_marker,
                                                         eattr1, eattr2);
      for (int j = 0; j < bdr_face_dofs.Size(); j++)
      {
         RemoveDofsFromBdrFaceDofs(*bdr_face_dofs[j], *bdr_edge_dofs[i]);
      }
      RemoveDofsFromBdrFaceDofs(*bdr_edge_dofs[i], aux_ess_dofs_s);
   }

   real_t min_detA_m0 = GetMinDet(pmesh, x0, quad_order);

   meshopt.Setup(x, &min_detA_m0, quad_order, metric_id,
                 target_id,
                 plb, detgf.get(), solver_iter, move_bnd, surf_mesh_attr,
                 aux_ess_dofs, solver_type, adapt_qp);

   Array<FindPointsGSLIB *> finder_arr;
   if (bdr_opt_case)
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

   meshopt.OptimizeNodes(x);

   {
      ostringstream mesh_name;
      mesh_name << "tangential-out.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // Visualize the final mesh.
   if (vis)
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, x0,
                             "Final mesh", 600, 0, 300, 300, "jRmlAe");
   }

   // 13. Free the used memory.
   delete target_c;
   delete metric;
   delete pmesh;

   return 0;
}
