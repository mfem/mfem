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

#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

#if(defined(MFEM_USE_MPI) && defined(MFEM_USE_GSLIB))
void OptimizeMeshWithAMRForAnotherMesh(ParMesh &pmesh,
                                       ParGridFunction &source,
                                       int amr_iter,
                                       ParGridFunction &x)
{
//   H1_FECollection h1fec(source.ParFESpace()->FEColl()->GetOrder(),
//                               pmesh.Dimension());
//   ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
//   ParGridFunction x(&h1fespace);
   const int dim = pmesh.Dimension();

   MFEM_VERIFY(pmesh.GetNodes() != NULL, "Nodes node set for mesh.");
   Vector vxyz = *(pmesh.GetNodes());
   int ordering = pmesh.GetNodes()->FESpace()->GetOrdering();
   const int nodes_cnt = vxyz.Size() / dim;
   Vector interp_vals(nodes_cnt);

   FindPointsGSLIB finder(pmesh.GetComm());
   finder.SetDefaultInterpolationValue(-1.0);
   finder.Setup(*(source.FESpace()->GetMesh()));

   vxyz = *(pmesh.GetNodes());
   finder.Interpolate(vxyz, source, interp_vals, ordering);
   x = interp_vals;

   L2_FECollection l2fec(0, pmesh.Dimension());
   ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   ParGridFunction el_to_refine(&l2fespace);

   H1_FECollection lhfec(1, pmesh.Dimension());
   ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   ParGridFunction lhx(&lhfespace);

   x.ExchangeFaceNbrData();

   for (int iter = 0; iter < amr_iter; iter++)
   {
      vxyz = *(pmesh.GetNodes());
      finder.Interpolate(vxyz, source, interp_vals, ordering);
      x = interp_vals;
      el_to_refine = 0.0;
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         x.ParFESpace()->GetElementDofs(e, dofs);
         const IntegrationRule &ir = x.ParFESpace()->GetFE(e)->GetNodes();
         x.GetValues(e, ir, x_vals);
         double min_val = x_vals.Min();
         double max_val = x_vals.Max();
         // Mark for refinement if the element is cut, or near the boundary.
         if (min_val * max_val < 0.0 ||
             fabs(min_val) < 1e-12 || fabs(max_val) < 1e-12)
         {
            el_to_refine(e) = 1.0;
         }
      }

      // Refine an element if its neighbor will be refined
      for (int inner_iter = 0; inner_iter < 1; inner_iter++)
      {
         el_to_refine.ExchangeFaceNbrData();
         GridFunctionCoefficient field_in_dg(&el_to_refine);
         lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
         for (int e = 0; e < pmesh.GetNE(); e++)
         {
            Array<int> dofs;
            Vector x_vals;
            lhfespace.GetElementDofs(e, dofs);
            //         lhx.GetSubVector(dofs, x_vals);
            const IntegrationRule &ir =
               IntRules.Get(lhx.ParFESpace()->GetFE(e)->GetGeomType(), 5);
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
         if (el_to_refine(e) > 0.0) { el_to_refine_list.Append(e); }
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
      x.ParFESpace()->Update();
      x.Update();

      l2fespace.Update();
      el_to_refine.Update();

      lhfespace.Update();
      lhx.Update();
   }
}

void ComputeScalarDistanceFromLevelSet(ParMesh &pmesh,
                                       GridFunctionCoefficient &ls_coeff,
                                       ParGridFunction &distance_s)
{
   H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   ParGridFunction x(&h1fespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   //Now determine distance
   const int p = 5;
   const int newton_iter = 50;
   PLapDistanceSolver dist_solver(p, newton_iter);
   dist_solver.print_level = 1;

   ParFiniteElementSpace pfes_s(*distance_s.ParFESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   ParGridFunction filt_gf(&pfes_s);
   const double dx = AvgElementSize(pmesh);
   PDEFilter filter(pmesh, 1.0 * dx);
   filter.Filter(ls_coeff, filt_gf);
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver.ComputeScalarDistance(ls_filt_coeff, distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

   //DiffuseField(distance_s, 10);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();
   for (int i = 0; i < distance_s.Size(); i++)
   {
      //distance_s(i) = std::fabs(distance_s(i));
      distance_s(i) *= -1;
   }
}
#endif

// g - c * dx * dx * laplace(g) = g_old.
void DiffuseH1(ParGridFunction &g, double c)
{
   ParFiniteElementSpace &pfes = *g.ParFESpace();

   auto check_h1 = dynamic_cast<const H1_FECollection *>(pfes.FEColl());
   MFEM_VERIFY(check_h1 && pfes.GetVDim() == 1,
               "This solver supports only scalar H1 spaces.");

   // Compute average mesh size (assumes similar cells).
   ParMesh &pmesh = *pfes.GetParMesh();
   double dx, loc_area = 0.0;
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      loc_area += pmesh.GetElementVolume(i);
   }
   double glob_area;
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE,
                 MPI_SUM, pmesh.GetComm());
   const int glob_zones = pmesh.GetGlobalNE();
   switch (pmesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         dx = glob_area / glob_zones; break;
      case Geometry::SQUARE:
         dx = sqrt(glob_area / glob_zones); break;
      case Geometry::TRIANGLE:
         dx = sqrt(2.0 * glob_area / glob_zones); break;
      case Geometry::CUBE:
         dx = pow(glob_area / glob_zones, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         dx = pow(6.0 * glob_area / glob_zones, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!"); dx = 0.0;
   }

   // Set up RHS.
   ParLinearForm b(&pfes);
   GridFunctionCoefficient g_old_coeff(&g);
   b.AddDomainIntegrator(new DomainLFIntegrator(g_old_coeff));
   b.Assemble();

   // Diffusion and mass terms in the LHS.
   ParBilinearForm a_n(&pfes);
   a_n.AddDomainIntegrator(new MassIntegrator);
   ConstantCoefficient c_coeff(c * dx * dx);
   a_n.AddDomainIntegrator(new DiffusionIntegrator(c_coeff));
   a_n.Assemble();

   // Solver.
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(-1);
   OperatorPtr A;
   Vector B, X;

   // Solve with Neumann BC.
   ParGridFunction u_neumann(&pfes);
   Array<int> ess_tdof_list;
   a_n.FormLinearSystem(ess_tdof_list, u_neumann, b, A, X, B);
   auto *prec = new HypreBoomerAMG;
   prec->SetPrintLevel(-1);
   cg.SetPreconditioner(*prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   a_n.RecoverFEMSolution(X, b, u_neumann);
   delete prec;

   g = u_neumann;
}
