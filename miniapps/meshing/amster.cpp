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
//    mpirun -np 4 amster -m ../../data/star.mesh
//    mpirun -np 4 amster -m blade.mesh -o 4 -battr 3
//    mpirun -np 4 amster -m bone3D.mesh
//

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"
#include "amster.hpp"

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "jagged.mesh";
   int mesh_poly_deg     = 2;
   bool fdscheme         = false;
   int quad_order        = 8;
   int bg_amr_steps      = 6;
   double surface_fit_const = 10.0;
   double surface_fit_adapt = 1.0;
   double surface_fit_threshold = 1e-7;
   int bdr_attr_to_fit      = 1;

   // Parse command-line input file.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&fdscheme, "-fd", "--fd_approx", "-no-fd", "--no-fd-approx",
                  "Enable finite difference based derivative computations.");
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
   args.AddOption(&bdr_attr_to_fit, "-battr", "--battr",
                  "Boundary attribute to fit.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }


   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   const int dim = pmesh->Dimension();
   delete mesh;

   // Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);
   pmesh->SetNodalFESpace(pfespace);

   // Get the mesh nodes as a finite element grid function in fespace.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   // Save the starting (prior to the optimization) mesh to a file.
   ostringstream mesh_name;
   mesh_name << "amster_in.mesh";
   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->PrintAsOne(mesh_ofs);

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // Metric.
   TMOP_QualityMetric *metric = NULL;
   if (dim == 2) { metric = new TMOP_Metric_004; }
   else          { metric = new TMOP_Metric_302; }

   TargetConstructor::TargetType target_t =
      TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   auto target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   target_c->SetNodes(x0);

   // Visualize the starting mesh and metric values.
   char title[] = "Initial metric values";
   vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);

   // Try to to pull back nodes from inverted elements next to the boundary.
   //   for (int f = 0; f < pfespace->GetNBE(); f++)
   //   {
   //      FaceElementTransformations *trans_f = pmesh->GetBdrFaceTransformations(f);
   //      const IntegrationRule &ir_f = pfespace->GetBE(f)->GetNodes();
   //      Vector n(dim);
   //      Array<int> vdofs_e;
   //      pfespace->GetElementVDofs(trans_f->Elem1No, vdofs_e);
   //      for (int i = 0; i < ir_f.GetNPoints(); i++)
   //      {
   //         const IntegrationPoint &ip = ir_f.IntPoint(i);
   //         trans_f->SetIntPoint(&ip);
   //         CalcOrtho(trans_f->Jacobian(), n);

   //         Vector coord_face_node;
   //         x.GetVectorValue(*trans_f, ip, coord_face_node);
   //         int ndof = vdofs_e.Size() / dim;

   //         for (int j = 0; j < ndof; j++)
   //         {
   //            Vector vec(dim);
   //            for (int d = 0; d < dim; d++)
   //            {
   //               vec(d) = coord_face_node(d) - x(vdofs_e[ndof*d + j]);
   //            }

   //            if (vec * n < -1e-8)
   //            {
   //               // The node is on the wrong side.
   //               cout << "wrong side "
   //                    << f << " " << trans_f->Elem1No << " " << j << endl;
   //               // Pull it inside;
   //               for (int d = 0; d < dim; d++)
   //               {
   //                  x(vdofs_e[ndof*d + j]) =
   //                        x(vdofs_e[ndof*d + j]) + 1.05 * vec(d);
   //               }
   //            }
   //         }
   //      }
   //   }
   //   // Visualize the starting mesh and metric values.
   //   char t[] = "Fixed outside nodes";
   //   vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, t, 0);

   // Detect boundary nodes.
   Array<int> vdofs;
   ParFiniteElementSpace sfespace = ParFiniteElementSpace(pmesh, fec);
   ParGridFunction domain(&sfespace);
   domain = 1.0;
   for (int i = 0; i < sfespace.GetNBE(); i++)
   {
      sfespace.GetBdrElementDofs(i, vdofs);
      for (int j = 0; j < vdofs.Size(); j++) { domain(vdofs[j]) = 0.0; }
   }

   // Distance to the boundary, on the original mesh.
   GridFunctionCoefficient coeff(&domain);
   ParGridFunction dist(&sfespace);
   ComputeScalarDistanceFromLevelSet(*pmesh, coeff, dist, false);
   dist *= -1.0;
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, dist,
                             "Dist to Boundary", 0, 700, 300, 300, "Rj");
   }

   // Create the background mesh.
   ParMesh *pmesh_bg = NULL;
   Mesh *mesh_bg = NULL;
   if (dim == 2)
   {
      mesh_bg = new Mesh(Mesh::MakeCartesian2D(5, 5, Element::QUADRILATERAL, true));
   }
   else if (dim == 3)
   {
      mesh_bg = new Mesh(Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON, true));
   }
   mesh_bg->EnsureNCMesh();
   pmesh_bg = new ParMesh(MPI_COMM_WORLD, *mesh_bg);
   delete mesh_bg;
   //   pmesh_bg->SetCurvature(mesh_poly_deg, false, -1, 0);
   // Set curvature to linear because we use it with FindPoints for interpolating a
   // linear function later
   pmesh_bg->SetCurvature(1, false, -1, 0);

   // Make the background mesh big enough to cover the original domain.
   Vector p_min(dim), p_max(dim);
   pmesh->GetBoundingBox(p_min, p_max);
   GridFunction &x_bg = *pmesh_bg->GetNodes();
   const int num_nodes = x_bg.Size() / dim;
   for (int i = 0; i < num_nodes; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         double length_d = p_max(d) - p_min(d),
                extra_d = 0.2 * length_d;
         x_bg(i + d*num_nodes) = p_min(d) - extra_d +
                                 x_bg(i + d*num_nodes) * (length_d + 2*extra_d);
      }
   }

#ifndef MFEM_USE_GSLIB
   MFEM_ABORT("GSLIB needed for this functionality.");
#endif

   // The background level set function is always linear to avoid oscillations.
   H1_FECollection bg_fec(1, dim);
   ParFiniteElementSpace bg_pfes(pmesh_bg, &bg_fec);
   ParGridFunction bg_domain(&bg_pfes);

   // Refine the background mesh around the boundary.
   OptimizeMeshWithAMRForAnotherMesh(*pmesh_bg, dist, bg_amr_steps, bg_domain);
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, bg_domain,
                             "Dist on Background", 300, 700, 300, 300, "Rj");
   }

   // Compute min element size.
   double min_dx = std::numeric_limits<double>::infinity();
   for (int e = 0; e < pmesh_bg->GetNE(); e++)
   {
      min_dx = fmin(min_dx, pmesh_bg->GetElementSize(e));
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_dx, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

   // Shift the zero level set by ~ one element inside.
   const double alpha = min_dx;
   bg_domain -= alpha;

   // Compute a distance function on the background.
   GridFunctionCoefficient ls_filt_coeff(&bg_domain);
   ComputeScalarDistanceFromLevelSet(*pmesh_bg, ls_filt_coeff, bg_domain, true);

   // Offset back to the original position of the boundary.
   bg_domain += alpha;
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, bg_domain,
                             "Final LS", 600, 700, 300, 300, "Rjmm");
   }


   //   auto btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Shifted;
   //   auto wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::Beta;

   //   TMOP_QualityMetric *untangler_metric =
   //      new TMOP_WorstCaseUntangleOptimizer_Metric(*metric, 2, 1.5, 0.001,//0.01 for pseudo barrier
   //                                                 0.001, btype, wctype);

   //   TMOP_QualityMetric *metric_to_use = untangler_metric;
   TMOP_Integrator *tmop_integ = new TMOP_Integrator(metric, target_c,
                                                     nullptr);

   // Finite differences for computations of derivatives.
   if (fdscheme) { tmop_integ->EnableFiniteDifferences(x); }

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = &IntRulesLo;
   tmop_integ->SetIntegrationRules(*irules, quad_order);
   if (myid == 0 && dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (myid == 0 && dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   // Surface fitting.
   L2_FECollection mat_coll(0, dim);
   Array<bool> surf_fit_marker(domain.Size());
   ParGridFunction surf_fit_mat_gf(&sfespace);
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   AdaptivityEvaluator *adapt_surface = NULL;
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;

   // Background mesh FECollection, FESpace, and GridFunction
   ParFiniteElementSpace *bg_grad_fes = NULL;
   ParGridFunction *bg_grad = NULL;
   ParFiniteElementSpace *bg_hess_fes = NULL;
   ParGridFunction *bg_hess = NULL;

   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   ParFiniteElementSpace *grad_fes = NULL;
   ParGridFunction *surf_fit_grad = NULL;
   ParFiniteElementSpace *surf_fit_hess_fes = NULL;
   ParGridFunction *surf_fit_hess = NULL;

   if (surface_fit_const > 0.0)
   {
      bg_grad_fes = new ParFiniteElementSpace(pmesh_bg, &bg_fec,
                                              pmesh_bg->Dimension());
      bg_grad = new ParGridFunction(bg_grad_fes);

      int n_hessian_bg = pow(pmesh_bg->Dimension(), 2);
      bg_hess_fes = new ParFiniteElementSpace(pmesh_bg, &bg_fec, n_hessian_bg);
      bg_hess = new ParGridFunction(bg_hess_fes);


      //Setup gradient of the background mesh
      bg_grad->ReorderByNodes();
      for (int d = 0; d < pmesh_bg->Dimension(); d++)
      {
         ParGridFunction bg_grad_comp(&bg_pfes, bg_grad->GetData()+d*bg_domain.Size());
         bg_domain.GetDerivative(1, d, bg_grad_comp);
      }

      //Setup Hessian on background mesh
      bg_hess->ReorderByNodes();
      int id = 0;
      for (int d = 0; d < pmesh_bg->Dimension(); d++)
      {
         for (int idir = 0; idir < pmesh_bg->Dimension(); idir++)
         {
            ParGridFunction bg_grad_comp(&bg_pfes, bg_grad->GetData()+d*bg_domain.Size());
            ParGridFunction bg_hess_comp(&bg_pfes, bg_hess->GetData()+id*bg_domain.Size());
            bg_grad_comp.GetDerivative(1, idir, bg_hess_comp);
            id++;
         }
      }


      //Setup functions on the mesh being optimized
      grad_fes = new ParFiniteElementSpace(pmesh, fec, pmesh->Dimension());
      surf_fit_grad = new ParGridFunction(grad_fes);

      surf_fit_hess_fes = new ParFiniteElementSpace(pmesh, fec,
                                                    pmesh->Dimension()*pmesh->Dimension());
      surf_fit_hess = new ParGridFunction(surf_fit_hess_fes);


      for (int i = 0; i < surf_fit_marker.Size(); i++)
      {
         surf_fit_marker[i] = false;
         surf_fit_mat_gf[i] = 0.0;
      }

      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         if (attr == bdr_attr_to_fit)
         {
            sfespace.GetBdrElementVDofs(i, vdofs);
            for (int j = 0; j < vdofs.Size(); j++)
            {
               surf_fit_marker[vdofs[j]] = true;
               surf_fit_mat_gf(vdofs[j]) = 1.0;
            }
         }
      }

      adapt_surface = new InterpolatorFP;
      adapt_grad_surface = new InterpolatorFP;
      adapt_hess_surface = new InterpolatorFP;

      tmop_integ->EnableSurfaceFittingFromSource(bg_domain,
                                                 domain,
                                                 surf_fit_marker, surf_fit_coeff,
                                                 *adapt_surface,
                                                 *bg_grad, *surf_fit_grad, *adapt_grad_surface,
                                                 *bg_hess, *surf_fit_hess, *adapt_hess_surface);
      if (true)
      {
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, surf_fit_mat_gf,
                                "Dofs to Move",
                                900, 600, 300, 300);
      }
   }

   // Setup the final NonlinearForm.
   ParNonlinearForm a(pfespace);
   a.AddDomainIntegrator(tmop_integ);

   // TODO remove this.
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   ess_bdr[bdr_attr_to_fit-1] = 0;
   a.SetEssentialBC(ess_bdr);

   // Compute the minimum det(J) of the starting mesh.
   double min_detJ = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_detJ, 1, MPI_DOUBLE,
                 MPI_MIN, MPI_COMM_WORLD);
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detJ << endl; }

   if (min_detJ < 0.0)
   {
      MFEM_VERIFY(target_t == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                  "Untangling is supported only for ideal targets.");

      const DenseMatrix &Wideal =
         Geometries.GetGeomToPerfGeomJac(pfespace->GetFE(0)->GetGeomType());
      min_detJ /= Wideal.Det();

      // Slightly below minJ0 to avoid div by 0.
      min_detJ -= 0.01 * min_dx;
   }

   // For HR tests, the energy is normalized by the number of elements.
   const double init_energy = a.GetParGridFunctionEnergy(x);
   //   surf_fit_coeff.constant   = 0.0;
   double init_metric_energy = a.GetParGridFunctionEnergy(x);
   //   surf_fit_coeff.constant  = surface_fit_const;

   // 15. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
   minres->SetMaxIter(100);
   minres->SetRelTol(1e-12);
   minres->SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   minres_pl.FirstAndLast().Summary();
   minres->SetPrintLevel(minres_pl);
   S = minres;

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfespace->GetComm(), ir);
   if (surface_fit_adapt > 0.0)
   {
      solver.SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
   }
   if (surface_fit_threshold > 0)
   {
      solver.SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
   }
   solver.SetIntegrationRules(*irules, quad_order);
   solver.SetPreconditioner(*S);

   // For untangling, the solver will update the min det(T) values.
   solver.SetMinDetPtr(&min_detJ);
   solver.SetMaxIter(50);
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   newton_pl.Iterations().Summary();
   solver.SetPrintLevel(newton_pl);

   solver.SetOperator(a);
   Vector b(0);
   x.SetTrueVector();
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "amster-out.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }

   // Visualize the final mesh and metric values.
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 600);
   }

   //   if (surface_fit_const > 0.0)
   //   {
   //      if (visualization)
   //      {
   //         socketstream vis2, vis3;
   //         common::VisualizeField(vis2, "localhost", 19916, mat,
   //                                "Materials", 600, 900, 300, 300);
   //         common::VisualizeField(vis3, "localhost", 19916, surf_fit_mat_gf,
   //                                "Surface dof", 900, 900, 300, 300);
   //      }
   //      double err_avg, err_max;
   //      tmop_integ->GetSurfaceFittingErrors(err_avg, err_max);
   //      if (myid == 0)
   //      {
   //         std::cout << "Avg fitting error: " << err_avg << std::endl
   //                   << "Max fitting error: " << err_max << std::endl;
   //      }
   //   }

   // 19. Visualize the mesh displacement.
   {
      x0 -= x;
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      x0.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
   }

   // 20. Free the used memory.
   delete S;
   //   delete metric_coeff1;
   //   delete adapt_lim_eval;
   //   delete adapt_surface;
   delete target_c;
   //   delete adapt_coeff;
   delete metric;
   //   delete untangler_metric;
   delete adapt_hess_surface;
   delete adapt_grad_surface;
   delete adapt_surface;
   delete surf_fit_hess;
   delete surf_fit_hess_fes;
   delete surf_fit_grad;
   delete grad_fes;
   delete bg_hess;
   delete bg_hess_fes;
   delete bg_grad;
   delete bg_grad_fes;
   delete pmesh_bg;
   delete pfespace;
   delete fec;
   delete pmesh;

   return 0;
}
