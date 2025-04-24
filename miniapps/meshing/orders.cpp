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
// Compile with: make orders
//
// Sample runs:
//

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

double test_func(const Vector &coord)
{
   return sin(M_PI*coord(0)) * sin(2.0*M_PI*coord(1));
}

double MinDetJ(ParMesh &pmesh, int quad_order);

void TransferLowToHigh(const ParGridFunction &l, ParGridFunction &h);

void TransferHighToLow(const ParGridFunction &h, ParGridFunction &l);

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

   double Residual(ParGridFunction &x);
};

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "blade.mesh";
   int rs_levels         = 0;
   int mesh_poly_deg     = 2;
   int solver_iter       = 50;
   int quad_order        = 8;
   int metric_id         = 2;

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
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   const int dim = pmesh.Dimension();
   delete mesh;

   // Define a finite element space on the mesh.
   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec, dim);
   pmesh.SetNodalFESpace(&pfes);

   // Get the mesh nodes as a finite element grid function in fespace.
   ParGridFunction x(&pfes);
   pmesh.SetNodalGridFunction(&x);

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(&pfes);
   x0 = x;

   {
      socketstream vis_g;
      common::VisualizeMesh(vis_g, "localhost", 19916, pmesh,
                            "Initial mesh", 0, 0, 400, 400, "mpRj");
   }

   MeshOptimizer optimizer;
   optimizer.Setup(pfes, metric_id, 8);
   double res_0 = optimizer.Residual(x);

   for (int o = 1; o < mesh_poly_deg; o++)
   {
      H1_FECollection fec_LO(o, dim);
      ParFiniteElementSpace pfes_LO(&pmesh, &fec_LO, dim);
      ParGridFunction x_LO(&pfes_LO);

      MeshOptimizer optimizer_LO;
      optimizer_LO.Setup(pfes_LO, metric_id, 6);
      TransferHighToLow(x, x_LO);
      optimizer_LO.OptimizeNodes(x_LO);
      TransferLowToHigh(x_LO, x);
   }

   optimizer.SetAbsTol(res_0 * 1e-8);
   optimizer.OptimizeNodes(x);

   {
      socketstream vis_g;
      common::VisualizeMesh(vis_g, "localhost", 19916, pmesh,
                            "HO Optimized", 800, 0, 400, 400, "mpRj");
   }

   return 0;
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


void TransferLowToHigh(const ParGridFunction &l, ParGridFunction &h)
{
   Array<int> ess_bdr(h.ParFESpace()->GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> h_ess_marker(h.Size());
   h.ParFESpace()->GetEssentialVDofs(ess_bdr, h_ess_marker);

   // Doesn't preserve the boundary nodes of h.
   //   TransferOperator transfer(*l.ParFESpace(), *h.ParFESpace());
   //   transfer.Mult(l, h);

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
   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   nlf->SetEssentialBC(ess_bdr);

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

double MeshOptimizer::Residual(ParGridFunction &x)
{
   MFEM_VERIFY(solver, "Setup() has not been called.");
   Vector b;
   x.SetTrueVector();
   return solver->GetResidual(b, x.GetTrueVector());
}
