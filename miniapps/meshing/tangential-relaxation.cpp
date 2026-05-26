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
// Note about boundary attribute requirement:
// Tangential relaxation identifies sliding boundary surfaces, edges, and
// vertices from the mesh boundary attributes. Adjacent boundary surfaces that
// are expected to slide independently must therefore use different attributes;
// otherwise the miniapp cannot correctly classify the face- and edge-based
// boundary nodes used by the relaxation step.
//
// See the following SIAM IMR 2026 paper, Mittal et al., "High-Order Mesh
// r-Adaptivity with Tangential Relaxation and Guaranteed Mesh Validity",
// arXiv. https://doi.org/10.48550/arXiv.2601.17708 for technical details.
//
// Compile with: make tangential-relaxation
//
// Sample runs:
// Blade - bound on Jacobian + tangential relaxation
// make tangential-relaxation -j4 && mpirun -np 4 tangential-relaxation -m blade.mesh -o 4 -qo 16 -vis -rs 0 -mid 2 -tid 1  -ni 400 -bnd -bdropt 1 -bound
// curvilinear surfaces
// make tangential-relaxation -j4 &&  mpirun -np 4 tangential-relaxation -m square01.mesh -o 2 -qo 8 -vis -mid 80 -tid 2 -bdropt 2 -rs 2  -ni 400 -bnd -transform 1
// make tangential-relaxation -j4 &&  mpirun -np 4 tangential-relaxation -m square01-tri.mesh -o 2 -qo 8 -vis -mid 80 -tid 2 -bdropt 2 -rs 1  -ni 400 -bnd -transform 1
// * mpirun -np 4 tangential-relaxation -m cube.mesh -o 2 -qo 12 -vis -mid 303 -tid 1 -rs 1 -ni 1000 -bnd -transform 1 -bdropt 3

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include "tangential-relaxation.hpp"

using namespace mfem;
using namespace std;

// Transformation from the unit square/cube to a smooth shield-like domain.
void UnitSquareShieldTransformation(const Vector &xin, Vector &xout)
{
   MFEM_VERIFY(xin.Size() == 2 || xin.Size() == 3,
               "This transformation is only defined in 2D and 3D.");

   real_t x = xin(0);
   real_t y = xin(1);
   x = x + x*(1-x)*0.4;
   y = y + y*(1-y)*0.4;

   // a adds deformation inside.
   // b pulls the top-right corner out.
   // c adds boundary deformation.
   real_t a = 0.2, b = 0.5, c = 1.5;

   if (xin.Size() == 2)
   {
      xout.SetSize(2);
      xout(0) = x + a * sin(0.5 * M_PI * x) * sin(c * M_PI * y) + b * x * y;
      xout(1) = y + a * sin(c * M_PI * x) * sin(0.5 * M_PI * y) + b * x * y;
      return;
   }

   real_t z = xin(2);
   z = z + z*(1-z)*0.25;
   const real_t zb = xin(2) * (1.0 - xin(2));
   const real_t xy_warp_x =
      a * sin(0.5 * M_PI * x) * sin(c * M_PI * y) + b * x * y;
   const real_t xy_warp_y =
      a * sin(c * M_PI * x) * sin(0.5 * M_PI * y) + b * x * y;

   xout.SetSize(3);
   xout(0) = x + (1.0 + 0.6 * zb) * xy_warp_x
             + 0.08 * a * sin(M_PI * xin(2)) * sin(c * M_PI * y);
   xout(1) = y + (1.0 + 0.6 * zb) * xy_warp_y
             + 0.08 * a * sin(M_PI * xin(2)) * sin(c * M_PI * x);
   xout(2) = z + 0.18 * a * sin(c * M_PI * x) * sin(M_PI * xin(2))
             + 0.18 * a * sin(c * M_PI * y) * sin(M_PI * xin(2))
             + 0.12 * b * x * y * sin(M_PI * xin(2));
}

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
   bool bound            = false;
   int transform         = 0;

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
                  "Supported values: 2D - 2, 49, 80 \n\t"
                  "3D - 301, 303, 323, 347.");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size");
   args.AddOption(&vis, "-vis", "--vis", "-no-vis", "--no-vis",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&bdr_opt_case, "-bdropt", "--bdr-opt",
                  "Boundary-attribute case for tangential relaxation. "
                  "See the boundary-attribute note in the file header.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries."
                  " Boundaries are identified by attributes: \n\t"
                  "1: parallel to y-axis in 2D, y-z plane in 3D\n\t"
                  "2: parallel to x-axis in 2D, x-z plane in 3D\n\t"
                  "3: parallel to x-y plane in 3D\n\t"
                  "Attributes based on bdr_opt_case are handled separately.");
   args.AddOption(&bound, "-bound", "--bound", "-no-bound",
                  "--no-bound",
                  "Enable bounds.");
   args.AddOption(&transform, "-transform", "--transform",
                  "Transformation type:\n\t"
                  "0: No transformation (default)\n\t"
                  "1: Smooth shield-like transformation of a unit-square mesh");
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

   // Apply the requested mesh transformation.
   if (transform == 1)
   {
      VectorFunctionCoefficient fcu(mesh->Dimension(),
                                    UnitSquareShieldTransformation);
      mesh->Transform(fcu);
   }
   else if (transform != 0)
   {
      MFEM_ABORT("Unknown transform option.");
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   const int dim = pmesh->Dimension();

   // Helper for encoding pairs of boundary attributes in 3D edge cases.
   auto setEdgeBits = [](int j, int k)
   {
      return (1 << (j-1)) | (1 << (k-1));
   };

   // Select the boundary attributes that will participate in tangential
   // relaxation and build the corresponding lower-dimensional meshes. See the
   // boundary-attribute note in the file header: adjacent sliding surfaces
   // must use distinct attributes so face, edge, and vertex dofs can be
   // classified correctly. This helps preserve the geometry during tangential
   // relaxation.
   Array<ParMesh *> surf_mesh_arr;
   Array<int> surf_mesh_attr, surf_mesh_edge_attr;
   if (bdr_opt_case == 1) //blade
   {
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   else if (bdr_opt_case == 2) // morphed - curvilinear right and top
   {
      surf_mesh_attr.SetSize(2);
      surf_mesh_arr.SetSize(2);
      surf_mesh_attr[0] = 1;
      surf_mesh_attr[1] = 2;
   }
   else if (bdr_opt_case == 3) // 3D case - kershaw
   {
      surf_mesh_attr.SetSize(3);
      surf_mesh_attr[0] = 1;
      surf_mesh_attr[1] = 2;
      surf_mesh_attr[2] = 3;
      surf_mesh_edge_attr.SetSize(3);
      surf_mesh_edge_attr[0] = setEdgeBits(1,2); // Edges between faces 1 and 2.
      surf_mesh_edge_attr[1] = setEdgeBits(2,3); // Edges between faces 2 and 3.
      surf_mesh_edge_attr[2] = setEdgeBits(1,3); // Edges between faces 1 and 3.

      surf_mesh_arr.SetSize(surf_mesh_attr.Size()+surf_mesh_edge_attr.Size());
   }
   else if (bdr_opt_case == 4) // Ale tangled - rotated circular hole
   {
      surf_mesh_attr.SetSize(1);
      surf_mesh_arr.SetSize(1);
      surf_mesh_attr[0] = 4;
   }
   double bbox_fac = 0.2;

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

   bool tensor_product_only =
      pmesh->GetNE() == 0 ||
      (pmesh->GetNumGeometries(dim) == 1 &&
       (pmesh->GetElementType(0)==Element::QUADRILATERAL ||
        pmesh->GetElementType(0) == Element::HEXAHEDRON));
   MPI_Allreduce(MPI_IN_PLACE, &tensor_product_only, 1, MFEM_MPI_CXX_BOOL,
                 MPI_LAND, MPI_COMM_WORLD);
   PLBound *plb = nullptr;
   if (tensor_product_only && bound)
   {
      plb = new PLBound(detgf->FESpace(),
                        ref_factor*(detgf->FESpace()->GetMaxElementOrder()+1));
   }

   // Visualize the starting mesh.
   if (vis)
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, x0,
                             "Initial mesh", 0, 0, 500, 500, "jRmlA");
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
      case 303: metric = new TMOP_Metric_303; break;
      case 323: metric = new TMOP_Metric_323; break;
      case 347: metric = new TMOP_Metric_347(0.5); break;
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
      for (int i = 0; i < surf_mesh_attr.Size(); i++)
      {
         const int attr = surf_mesh_attr[i];
         if (attr >= 1 && attr <= ess_bdr.Size())
         {
            ess_bdr[attr - 1] = 0;
         }
      }
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

   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   auto *solver = new TMOPNewtonSolver(pfes.GetComm(), ir, 1); // use LBFGS
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
         finder->SetDistanceToleranceForPointsFoundOnBoundary(0.2);
         tang_dofs_arr.Append(bdr_face_dofs[i]);
         nodes0_arr.Append(surf_mesh_arr[i]->GetNodes());
      }
      int noff = surf_mesh_attr.Size();
      for (int i = 0; i < surf_mesh_edge_attr.Size(); i++)
      {
         FindPointsGSLIB *finder = new FindPointsGSLIB();
         finder_arr.Append(finder);
         finder->SetupSurf(*surf_mesh_arr[i+noff], bbox_fac);
         finder->SetDistanceToleranceForPointsFoundOnBoundary(0.2);
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
                             "Final mesh", 600, 0, 500, 500, "jRmlA");
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
