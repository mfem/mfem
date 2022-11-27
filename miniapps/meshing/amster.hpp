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
   int nodes_cnt = vxyz.Size() / dim;
   Vector interp_vals(nodes_cnt);

   FindPointsGSLIB finder(source.ParFESpace()->GetComm());
   finder.SetDefaultInterpolationValue(-1.0);
   finder.Setup(*(source.FESpace()->GetMesh()),0.0);

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
               IntRules.Get(lhx.ParFESpace()->GetFE(e)->GetGeomType(), 7);
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
   {
       vxyz = *(pmesh.GetNodes());
       finder.Interpolate(vxyz, source, interp_vals, ordering);
       x = interp_vals;
   }
}

void ComputeScalarDistanceFromLevelSet(ParMesh &pmesh,
                                       GridFunctionCoefficient &ls_coeff,
                                       ParGridFunction &distance_s,
                                       bool filter_input,
                                       const int pLap = 5)
{
   H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                         pmesh.Dimension());
   ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   ParGridFunction x(&h1fespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   //Now determine distance
   const int p = pLap;
   const int newton_iter = 50;
   PLapDistanceSolver dist_solver(p, newton_iter);
   dist_solver.print_level = 1;

   ParFiniteElementSpace pfes_s(*distance_s.ParFESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   ParGridFunction filt_gf(&pfes_s);
   if (filter_input)
   {
      const double dx = AvgElementSize(pmesh);
      PDEFilter filter(pmesh, 1.0 * dx);
      filter.Filter(ls_coeff, filt_gf);
   }
   else { filt_gf.ProjectCoefficient(ls_coeff); }
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver.ComputeScalarDistance(ls_filt_coeff, distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

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
   Array<int> ess_tdof_list;
   a_n.FormLinearSystem(ess_tdof_list, g, b, A, X, B);
   auto *prec = new HypreBoomerAMG;
   prec->SetPrintLevel(-1);
   cg.SetPreconditioner(*prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   a_n.RecoverFEMSolution(X, b, g);
   delete prec;
}

class BackgroundData
{
public:
   int dist_order = 1;
   ParMesh *pmesh_bg = nullptr;
   H1_FECollection *fec_bg = nullptr;
   ParFiniteElementSpace *pfes_bg = nullptr;
   ParGridFunction *dist_bg = nullptr;

   BackgroundData(ParMesh &pmesh_front, ParGridFunction &dist_front,
                  int amr_steps);

   ~BackgroundData()
   {
      delete dist_bg;
      delete pfes_bg;
      delete fec_bg;
      delete pmesh_bg;
   }
};

BackgroundData::BackgroundData(ParMesh &pmesh_front,
                               ParGridFunction &dist_front, int amr_steps)
{
   const int dim = pmesh_front.Dimension();

   // Create the background mesh.
   Mesh *m = NULL;
   if (dim == 2)
   {
      m = new Mesh(Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true));
   }
   else if (dim == 3)
   {
      m = new Mesh(Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON, true));
   }
   m->EnsureNCMesh();
   pmesh_bg = new ParMesh(MPI_COMM_WORLD, *m);
   delete m;
   // TODO does this have to be the same order? Can it be just 1?
   pmesh_bg->SetCurvature(dist_order, false, -1, 0);

   // Make the background mesh big enough to cover the original domain.
   Vector p_min(dim), p_max(dim);
   pmesh_front.GetBoundingBox(p_min, p_max);
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

   fec_bg = new H1_FECollection(dist_order, dim);
   pfes_bg = new ParFiniteElementSpace(pmesh_bg, fec_bg);
   dist_bg = new ParGridFunction(pfes_bg);

   OptimizeMeshWithAMRForAnotherMesh(*pmesh_bg, dist_front,
                                     amr_steps, *dist_bg);
   pmesh_bg->Rebalance();
   pfes_bg->Update();
   dist_bg->Update();
}


class MeshOptimizer
{
private:
   TMOP_QualityMetric *metric = nullptr;
   TargetConstructor *target_c = nullptr;
   ParNonlinearForm *nlf = nullptr;
   IterativeSolver *lin_solver = nullptr;
   TMOPNewtonSolver *solver = nullptr;

public:
   MeshOptimizer() { }

   ~MeshOptimizer()
   {
      delete solver;
      delete lin_solver;
      delete nlf;
      delete target_c;
      delete metric;
   }

   // Must be called before optimization.
   void Setup(ParFiniteElementSpace &pfes, int metric_id, int quad_order);

   void SetAbsTol(double atol) { solver->SetAbsTol(atol); }

   // Optimizes the node positions given in x.
   // When we enter, x contains the initial node positions.
   // When we exit, x contains the optimized node positions.
   // The underlying mesh of x remains unchanged (its positions don't change).
   void OptimizeNodes(ParGridFunction &x);

   TMOP_Integrator *GetIntegrator();

   TMOPNewtonSolver *GetSolver() { return solver; }

   double Residual(ParGridFunction &x);
};

void MeshOptimizer::Setup(ParFiniteElementSpace &pfes,
                          int metric_id, int quad_order)
{
   const int dim = pfes.GetMesh()->Dimension();

   // Metric.
   if (dim == 2)
   {
      switch (metric_id)
      {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 50: metric = new TMOP_Metric_050; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 80: metric = new TMOP_Metric_080(0.1); break;
      }
   }
   else { metric = new TMOP_Metric_302; }

   // Target.
   TargetConstructor::TargetType target =
         TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   target_c = new TargetConstructor(target, pfes.GetComm());

   // Integrator.
   auto tmop_integ = new TMOP_Integrator(metric, target_c, nullptr);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);

   // Nonlinear form.
   nlf = new ParNonlinearForm(&pfes);
   nlf->AddDomainIntegrator(tmop_integ);

   // Boundary.
//   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
//   ess_bdr = 1;
//   nlf->SetEssentialBC(ess_bdr);

   // Linear solver.
   lin_solver = new MINRESSolver(pfes.GetComm());
   lin_solver->SetMaxIter(100);
   lin_solver->SetRelTol(1e-12);
   lin_solver->SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   lin_solver->SetPrintLevel(minres_pl.FirstAndLast().Summary());

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   solver = new TMOPNewtonSolver(pfes.GetComm(), ir);
   solver->SetIntegrationRules(IntRulesLo, quad_order);
   solver->SetOperator(*nlf);
   solver->SetPreconditioner(*lin_solver);
   solver->SetMaxIter(1000);
   solver->SetRelTol(1e-8);
   solver->SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver->SetPrintLevel(newton_pl.Iterations().Summary());
}

double MinDetJ(ParMesh &pmesh, int quad_order)
{
   GridFunction &nodes = *pmesh.GetNodes();
   FiniteElementSpace &pfes = *nodes.FESpace();

   double min_detJ = infinity();
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const IntegrationRule &ir =
         IntRulesLo.Get(pfes.GetFE(e)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         transf->SetIntPoint(&ir.IntPoint(q));
         min_detJ = fmin(min_detJ, transf->Jacobian().Det());
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &min_detJ, 1, MPI_DOUBLE, MPI_MIN,
                 pmesh.GetComm());

   return min_detJ;
}

void MeshOptimizer::OptimizeNodes(ParGridFunction &x)
{
   MFEM_VERIFY(solver, "Setup() has not been called.");

   ParMesh &pmesh = *x.ParFESpace()->GetParMesh();
   int myid = pmesh.GetMyRank();

   GridFunction *ptr_nodes = pmesh.GetNodes();
   GridFunction *ptr_x = &x;
   int dont_own_nodes = 0;
   pmesh.SwapNodes(ptr_x, dont_own_nodes);

   ParFiniteElementSpace &pfes = *x.ParFESpace();

   const int quad_order =
      solver->GetIntegrationRule(*x.ParFESpace()->GetFE(0)).GetOrder();
   const int order = pfes.GetFE(0)->GetOrder();
   double min_detJ = MinDetJ(pmesh, quad_order);
   if (myid == 0)
   {
      cout << "\n*** Optimizing Order " << order << " ***\n\n";
      cout << "Min detJ before opt: " << min_detJ << endl;
   }

   // Optimize.
   x.SetTrueVector();
   Vector b;
   solver->Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   min_detJ = MinDetJ(pmesh, quad_order);
   if (myid == 0)
   {
      cout << "Min detJ after opt: " << min_detJ << endl;
   }

   pmesh.SwapNodes(ptr_nodes, dont_own_nodes);
}

TMOP_Integrator *MeshOptimizer::GetIntegrator()
{
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   return dynamic_cast<TMOP_Integrator *>(integs[0]);
}

double MeshOptimizer::Residual(ParGridFunction &x)
{
   MFEM_VERIFY(solver, "Setup() has not been called.");
   Vector b;
   x.SetTrueVector();
   return solver->GetResidual(b, x.GetTrueVector());
}
