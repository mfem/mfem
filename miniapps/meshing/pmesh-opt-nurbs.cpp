// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
//    ------------------------------------------------------------------
//      Fitting of Selected Mesh Nodes to Specified Physical Positions
//    ------------------------------------------------------------------
//
// This example fits a selected set of the mesh nodes to given physical
// positions while maintaining a valid mesh with good quality.
//
// Sample runs:
//   make pmesh-opt-nurbs -j4 && mpirun -np 10 pmesh-opt-nurbs -m nurbs-square-disc.mesh -rs 1

// make pmesh-opt-nurbs -j4 && mpirun -np 10 pmesh-opt-nurbs -m nurbs-aniso.mesh -rs 0

// glvis -np 10 -m optimized

// make pmesh-opt-nurbs -j4 && mpirun -np 4 pmesh-opt-nurbs -m invalid.mesh -mod-bndr-attr  -mid 4 -btype 1


#include "mfem.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

char vishost[] = "localhost";
int  wsize     = 350;

void ModifyBoundaryAttributesForNodeMovement(ParMesh *pmesh, ParGridFunction &x)
{
   const int dim = pmesh->Dimension();
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      mfem::Array<int> dofs;
      pmesh->GetNodalFESpace()->GetBdrElementDofs(i, dofs);
      mfem::Vector bdr_xy_data;
      mfem::Vector dof_xyz(dim);
      mfem::Vector dof_xyz_compare;
      mfem::Array<int> xyz_check(dim);
      for (int j = 0; j < dofs.Size(); j++)
      {
         for (int d = 0; d < dim; d++)
         {
            dof_xyz(d) = x(pmesh->GetNodalFESpace()->DofToVDof(dofs[j], d));
         }
         if (j == 0)
         {
            dof_xyz_compare = dof_xyz;
            xyz_check = 1;
         }
         else
         {
            for (int d = 0; d < dim; d++)
            {
               if (std::fabs(dof_xyz(d)-dof_xyz_compare(d)) < 1.e-10)
               {
                  xyz_check[d] += 1;
               }
            }
         }
      }
      if (dim == 2)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
      else if (dim == 3)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else if (xyz_check[2] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 3);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
   }
}

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

int main (int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();

   const char *mesh_file = "square01.mesh";
   int rs_levels     = 0;
   int mesh_poly_deg = 2;
   int quad_order    = 5;
   bool glvis        = true;
   int visport       = 19916;
   real_t lim_const      = 0.0;
   bool mod_bndr_attr    = false;
   int metric_id        = 2;
   int barrier_type      = 0;
   int worst_case_type   = 0;
   bool use_wcu_metric   = false;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&glvis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&mod_bndr_attr, "-mod-bndr-attr",
                  "--modify-boundary-attribute",
                  "-fix-bndr-attr", "--fix-boundary-attribute",
                  "Change boundary attribue based on alignment with Cartesian axes.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric");

   args.AddOption(&barrier_type, "-btype", "--barrier-type",
                  "0 - None,"
                  "1 - Shifted Barrier,"
                  "2 - Pseudo Barrier.");
   args.AddOption(&worst_case_type, "-wctype", "--worst-case-type",
                  "0 - None,"
                  "1 - Beta,"
                  "2 - PMean.");
   args.AddOption(&use_wcu_metric, "-wcu", "--worstcase-untangle-metric",
                  "-no-wcu", "--no-worstcase-untangle-metric",
                  "Enable the TMOP_WorstCaseUntangleOptimizer_Metric wrapper.\n\t"
                  "Controlled by -btype and/or -wctype. If both are 0, defaults\n\t"
                  "to -wctype 1 (Beta).");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   if (use_wcu_metric && barrier_type == 0 && worst_case_type == 0)
   {
      worst_case_type = 1; // Beta
      if (myid == 0)
      {
         cout << "-wcu enabled with -btype 0 and -wctype 0; defaulting to -wctype 1 (Beta)."
              << endl;
      }
   }

   // Read and refine the mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   const int dim = pmesh.Dimension();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "perturbed." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);
   }

   ParGridFunction *coord = dynamic_cast<ParGridFunction*>(pmesh.GetNodes());
   const ParFiniteElementSpace *pfes_mesh = coord->ParFESpace();
   ParGridFunction x0(*coord);
   ParFiniteElementSpace *pfes = const_cast<ParFiniteElementSpace*>(pfes_mesh);

   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(&pmesh, *coord);
      pmesh.SetAttributes();
   }


   NURBSExtension *nurbs = pmesh.NURBSext;
   Array<int> vdofs;
   int n = 0;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int nd = pfes_mesh->GetBE(i)->GetDof();
      const int attr = pmesh.GetBdrElement(i)->GetAttribute();
      if (attr == 1 || attr == 2 || attr == dim) { n += nd; }
      if (attr >= dim+1) { n += nd * dim; }
   }
   Array<int> ess_vdofs(n);
   n = 0;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int nd = pfes_mesh->GetBE(i)->GetDof();
      const int attr = pmesh.GetBdrElement(i)->GetAttribute();
      pfes_mesh->GetBdrElementVDofs(i, vdofs);
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
      else if (attr == dim) // Fix z components.
      {
         for (int j = 0; j < nd; j++)
         { ess_vdofs[n++] = vdofs[j+2*nd]; }
      }
      else if (attr >= dim+1) // Fix all components.
      {
         for (int j = 0; j < vdofs.Size(); j++)
         { ess_vdofs[n++] = vdofs[j]; }
      }
   }

   IntegrationRules *irules = &IntRulesLo;
   real_t min_detJ = infinity();
   const int NE = pmesh.GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfes_mesh->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh.GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   real_t minJ0;
   MPI_Allreduce(&min_detJ, &minJ0, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, MPI_COMM_WORLD);
   min_detJ = minJ0;
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detJ << endl; }


   // TMOP setup.
   TMOP_QualityMetric *metric = nullptr;
   if (dim == 2)
   {
      switch (metric_id)
      {
         // T-metrics
         case 1: metric = new TMOP_Metric_001; break;
         case 2: metric = new TMOP_Metric_002; break;
         case 4: metric = new TMOP_Metric_004; break;
         case 7: metric = new TMOP_Metric_007; break;
         case 9: metric = new TMOP_Metric_009; break;
         case 14: metric = new TMOP_Metric_014; break;
         case 22: metric = new TMOP_Metric_022(min_detJ); break;
         case 50: metric = new TMOP_Metric_050; break;
         case 55: metric = new TMOP_Metric_055; break;
         case 56: metric = new TMOP_Metric_056; break;
         case 58: metric = new TMOP_Metric_058; break;
         case 66: metric = new TMOP_Metric_066(0.5); break;
         case 77: metric = new TMOP_Metric_077; break;
         case 80: metric = new TMOP_Metric_080(0.5); break;
         case 85: metric = new TMOP_Metric_085; break;
         case 90: metric = new TMOP_Metric_090; break;
         case 94: metric = new TMOP_Metric_094; break;
         case 98: metric = new TMOP_Metric_098; break;
         // case 211: metric = new TMOP_Metric_211; break;
         // case 252: metric = new TMOP_Metric_252(min_detJ); break;
         default:
            if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
            return 3;
      }
   }
   else
   {
      metric = new TMOP_Metric_302;
   }

   TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType btype;
   switch (barrier_type)
   {
      case 0: btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::None;
         break;
      case 1: btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Shifted;
         break;
      case 2: btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Pseudo;
         break;
      default: cout << "barrier_type not supported: " << barrier_type << endl;
         return 3;
   }

   TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType wctype;
   switch (worst_case_type)
   {
      case 0: wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::None;
         break;
      case 1: wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::Beta;
         break;
      case 2: wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::PMean;
         break;
      default: cout << "worst_case_type not supported: " << worst_case_type << endl;
         return 3;
   }

   TMOP_QualityMetric *untangler_metric = NULL;
   if (use_wcu_metric || barrier_type > 0 || worst_case_type > 0)
   {
      if (barrier_type > 0)
      {
         MFEM_VERIFY(metric_id == 4 || metric_id == 14 || metric_id == 66,
                     "Metric not supported for shifted/pseudo barriers.");
      }
      untangler_metric = new TMOP_WorstCaseUntangleOptimizer_Metric(*metric,
                                                                    2,
                                                                    1.5,
                                                                    0.001, // 0.01 for pseudo barrier
                                                                    0.001,
                                                                    btype,
                                                                    wctype);
   }

   TMOP_QualityMetric *metric_to_use = (use_wcu_metric || barrier_type > 0 ||
                                        worst_case_type > 0)
                                       ? untangler_metric
                                       : metric;

   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                            pfes_mesh->GetComm());
   auto integ = new TMOP_Integrator(metric_to_use, &target, nullptr);

   if (use_wcu_metric || barrier_type > 0 || worst_case_type > 0)
   {
      integ->ComputeUntangleMetricQuantiles(*coord, *pfes_mesh);
   }

   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { integ->EnableLimiting(x0, lim_coeff); }

   // Linear solver.
   MINRESSolver minres(pfes_mesh->GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);

   // Nonlinear solver.
   ParNonlinearForm a(pfes);
   a.SetEssentialVDofs(ess_vdofs);
   a.AddDomainIntegrator(integ);
   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   TMOPNewtonSolver solver(pfes_mesh->GetComm(), ir, 0);
   solver.SetOperator(a);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(2000);
   solver.SetRelTol(1e-10);
   solver.SetAbsTol(0.0);
   solver.SetMinDetPtr(&min_detJ);

   real_t init_energy = a.GetParGridFunctionEnergy(*coord);
   std::cout << "Initial energy: " << init_energy << std::endl;

   // Solve.
   Vector b(0);
   coord->SetTrueVector();
   solver.Mult(b, coord->GetTrueVector());
   coord->SetFromTrueVector();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "optimized." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);
   }

   if (glvis)
   {
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Final mesh",
                            400, 0, 400, 400);
   }

   delete metric;
   delete untangler_metric;
   return 0;
}
