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
//       Tangential Relaxation Miniapp: TMOP-based Mesh Optimization with
//                 Tangential Relaxation of Boundary Nodes
//    ---------------------------------------------------------------------
//
// This miniapp performs high-order mesh optimization while allowing selected
// boundary nodes to slide tangentially along extracted curves and surfaces.
// The mesh quality objective is based on the Target-Matrix Optimization
// Paradigm (TMOP), while tangential relaxation is enforced by projecting the
// selected boundary nodes to lower-dimensional reference meshes constructed
// from the initial geometry.
//
// Compile with: make tangential-relaxation
//
// Sample runs:
// Blade - bound on Jacobian + tangential relaxation
// make tangential-relaxation -j4 && mpirun -np 4 tangential-relaxation -m blade.mesh -o 4 -qo 16 -vis -rs 0 -mid 2 -tid 1  -ni 400 -st 1 -bnd -bdropt 1 -bound

// Ale tangled - curvilinear right and top boundaries
// mpirun -np 3 tangential-relaxation -m 2Dmorphedsquare.mesh -o 2 -qo 8 -vis -mid 80 -tid 2 -bdropt 2  -ni 400 -bnd -bound

// circular hole
// make tangential-relaxation -j4 && mpirun -np 3 tangential-relaxation -m 2Dcircularhole.mesh -o 2 -qo 8 -vis -rs 0 -mid 80 -tid 2 -bdropt 4 -ni 200 -bnd -bound

// 3D
// make tangential-relaxation -j4 && mpirun -np 10 tangential-relaxation -m kershaw_laghos_morphed_2.mesh -o 2 -qo 8 -vis -rs 0 -mid 301 -tid 1 -bdropt 8  -ni 500 -bnd -no-bound


#include "mfem.hpp"
#include "../common/mfem-common.hpp"
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
   bool bound            = true;
   bool transform        = true;

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
   args.AddOption(&bound, "-bound", "--bound", "-no-bound",
                  "--no-bound",
                  "Enable bounds.");
   args.AddOption(&transform, "-transform", "--transform", "-no-transform",
                  "--no-transform",
                  "Enable transforms.");
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

   // Helper for encoding pairs of boundary attributes in 3D edge cases.
   auto setTwoBits = [](int j, int k)
   {
      return (1 << (j-1)) | (1 << (k-1));
   };

   // Select the boundary attributes that will participate in tangential
   // relaxation and build the corresponding lower-dimensional meshes.
   Array<ParMesh *> surf_mesh_arr;
   Array<int> surf_mesh_attr, surf_mesh_edge_attr;
   if (bdr_opt_case == 1) //blade
   {
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   else if (bdr_opt_case == 2) // Ale tangled - curvilinear right and top
   {
      surf_mesh_attr.SetSize(2);
      surf_mesh_arr.SetSize(2);
      surf_mesh_attr[0] = 3;
      surf_mesh_attr[1] = 4;
   }
   else if (bdr_opt_case == 4) // Ale tangled - rotated circular hole
   {
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   else if (bdr_opt_case == 8) // 3D case - kershaw
   {
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
   double bbox_fac = 2.0;

   // Decode the stored boundary-attribute pairs.
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

      return result;
   };

   // Extract the boundary curves/surfaces from the initial serial mesh. These
   // meshes define the geometry on which boundary nodes are allowed to slide.
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
         Mesh *meshsurf = SetupEdgeMesh3D(mesh, attr_count_ser,
                                          attr_marker_ser, eattr1, eattr2);
         surf_mesh_arr[i+surf_mesh_attr.Size()] = new ParMesh(MPI_COMM_WORLD,
                                                              *meshsurf);
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

   // Compute Jacobian determinant bounds from the initial mesh if requested.
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

   // Mark boundary dofs by attribute and identify the subset that will be used
   // for tangential relaxation.
   ParGridFunction attr_count(&pfes);
   ParGridFunction attr_count_s(&spfes);
   Array<int> attr_marker;
   SetupDofAttributes(attr_count, attr_marker);
   SetupDofAttributes(attr_count_s, attr_marker);
   Array<int> aux_ess_dofs = IdentifyAuxiliaryEssentialDofs(attr_count);
   Array<int> aux_ess_dofs_s = IdentifyAuxiliaryEssentialDofs(attr_count_s);
   // Get dofs for all selected boundary faces and edges.
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

   // Define the TMOP quality metric and target.
   real_t min_det_m0 = GetMinDet(pmesh, x0, quad_order);
   TMOP_QualityMetric *metric = nullptr;
   switch (metric_id)
   {
      case 2: metric = new TMOP_Metric_002; break;
      case 49: metric = new TMOP_AMetric_049(0.6); break;
      case 80: metric = new TMOP_Metric_080(0.25); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 322: metric = new TMOP_Metric_322; break;
      case 360: metric = new TMOP_Metric_360; break;
      default: MFEM_ABORT("Unknown metric_id");
   }

   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default: MFEM_ABORT("Unknown target_id");
   }
   TargetConstructor *target_c =
      new TargetConstructor(target_t, x.ParFESpace()->GetComm());
   target_c->SetNodes(x);

   // Setup the TMOP nonlinear form.
   auto *tmop_integ = new TMOP_Integrator(metric, target_c, nullptr);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);
   if (plb)
   {
      tmop_integ->EnableDeterminantPLBounds(detgf.get(), 3, 2);
   }

   auto *nlf = new ParNonlinearForm(&pfes);
   nlf->AddDomainIntegrator(tmop_integ);

   // Fix non-sliding boundary dofs and preserve corners and junctions.
   if (!move_bnd)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      nlf->SetEssentialBC(ess_bdr);
   }
   else
   {
      int n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfes.GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         if (surf_mesh_attr.Find(attr) == -1)
         {
            if (attr == 1 || attr == 2 || (attr == 3 && dim == 3)) { n += nd; }
            if (attr >= dim+1) { n += nd * dim; }
         }
      }

      Array<int> ess_vdofs(n);
      n = 0;
      Array<int> vdofs;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfes.GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfes.GetBdrElementVDofs(i, vdofs);
         if (surf_mesh_attr.Find(attr) == -1)
         {
            if (attr == 1)
            {
               for (int j = 0; j < nd; j++)
               {
                  ess_vdofs[n++] = vdofs[j];
               }
            }
            else if (attr == 2)
            {
               for (int j = 0; j < nd; j++)
               {
                  ess_vdofs[n++] = vdofs[j+nd];
               }
            }
            else if (attr == 3 && dim == 3)
            {
               for (int j = 0; j < nd; j++)
               {
                  ess_vdofs[n++] = vdofs[j+2*nd];
               }
            }
            else if (attr >= dim+1)
            {
               for (int j = 0; j < vdofs.Size(); j++)
               {
                  ess_vdofs[n++] = vdofs[j];
               }
            }
         }
      }
      for (int i = 0; i < aux_ess_dofs.Size(); i++)
      {
         ess_vdofs.Append(aux_ess_dofs[i]);
      }
      nlf->SetEssentialVDofs(ess_vdofs);
   }

   // Setup solver
   const  int solver_type       = 1; // LBFGS
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   auto *solver = new TMOPNewtonSolver(pfes.GetComm(), ir, solver_type);
   solver->SetIntegrationRules(IntRulesLo, quad_order);
   if (plb) { solver->EnsurePositiveDeterminantBound(); }
   solver->SetOperator(*nlf);
   solver->SetMinDetPtr(&min_det_m0);
   solver->SetMaxIter(solver_iter);
   solver->SetRelTol(1e-8);
   solver->SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver->SetPrintLevel(newton_pl.Iterations().Summary());

   // Setup FindPoints data for the extracted boundary meshes and enable
   // tangential relaxation for the selected boundary dofs.
   Array<FindPointsGSLIB *> finder_arr;
   Array<Array<int> *> tang_dofs_arr;
   Array<GridFunction *> nodes0_arr;
   if (bdr_opt_case)
   {
      for (int i = 0; i < surf_mesh_attr.Size(); i++)
      {
         FindPointsGSLIB *finder = new FindPointsGSLIB();
         finder_arr.Append(finder);
         finder->SetupSurf(*surf_mesh_arr[i], bbox_fac);
         finder->SetDistanceToleranceForPointsFoundOnBoundary(10);
         tang_dofs_arr.Append(bdr_face_dofs[i]);
         nodes0_arr.Append(surf_mesh_arr[i]->GetNodes());
      }
      int noff = surf_mesh_attr.Size();
      for (int i = 0; i < surf_mesh_edge_attr.Size(); i++)
      {
         FindPointsGSLIB *finder = new FindPointsGSLIB();
         finder_arr.Append(finder);
         finder->SetupSurf(*surf_mesh_arr[i+noff], bbox_fac);
         finder->SetDistanceToleranceForPointsFoundOnBoundary(10);
         tang_dofs_arr.Append(bdr_edge_dofs[i]);
         nodes0_arr.Append(surf_mesh_arr[i+noff]->GetNodes());
      }
      solver->SetTangentialRelaxationFlag(true);
      tmop_integ->EnableTangentialRelaxation(finder_arr, tang_dofs_arr,
                                             nodes0_arr);
   }

   // Optimize the mesh nodes.
   const real_t init_energy = nlf->GetParGridFunctionEnergy(x);
   x.SetTrueVector();
   Vector b;
   solver->Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   const real_t final_energy = nlf->GetParGridFunctionEnergy(x);
   if (pfes.GetMyRank() == 0)
   {
      std::cout << "Initial energy: " << init_energy << endl
                << "Final energy:   " << final_energy << endl;
   }

   // Save the final optimized mesh.
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
   delete solver;
   delete nlf;
   delete target_c;
   delete metric;
   for (int i = 0; i < finder_arr.Size(); i++) { delete finder_arr[i]; }
   for (int i = 0; i < bdr_face_dofs.Size(); i++) { delete bdr_face_dofs[i]; }
   for (int i = 0; i < bdr_edge_dofs.Size(); i++) { delete bdr_edge_dofs[i]; }
   for (int i = 0; i < surf_mesh_arr.Size(); i++) { delete surf_mesh_arr[i]; }
   delete pmesh;

   return 0;
}
