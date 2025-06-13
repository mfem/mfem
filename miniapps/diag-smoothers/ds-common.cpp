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

#include "ds-common.hpp"

using namespace std;
using namespace mfem;

namespace ds_common
{

int MONITOR_DIGITS = 20;
int MG_MAX_ITER = 10;
real_t MG_REL_TOL = std::sqrt(1e-10);

int dim = 0;
int space_dim = 0;
real_t freq = 1.0;
real_t kappa = 1.0;

// Custom monitor that prints a csv-formatted file
DataMonitor::DataMonitor(string file_name, int ndigits)
   : os(file_name),
     precision(ndigits)
{
   if (Mpi::Root())
   {
      mfem::out << "Saving iterations into: " << file_name << endl;
   }
   os << "it,res,sol" << endl;
   os << fixed << setprecision(precision);
}

void DataMonitor::MonitorResidual(int it, real_t norm, const Vector &x,
                                  bool final)
{
   os << it << "," << norm << ",";
}

void DataMonitor::MonitorSolution(int it, real_t norm, const Vector &x,
                                  bool final)
{
   os << norm << endl;
}


// Abs-L(1) general geometric multigrid method, derived from GeometricMultigrid
AbsL1GeometricMultigrid::AbsL1GeometricMultigrid(
   ParFiniteElementSpaceHierarchy& fes_hierarchy,
   Array<int>& ess_bdr,
   IntegratorType it,
   SolverType st,
   AssemblyLevel al)
   : GeometricMultigrid(fes_hierarchy, ess_bdr),
     integrator_type(it),
     solver_type(st),
     assembly_level(al),
     coarse_pc(nullptr),
     one(1.0)
{
   // BilinearForm::FormSystemMatrix does not handle the ownership of A_l.
   // GeometricMultigrid owns the forms, and deletes them.
   mg_owned = !(AssemblyLevel::LEGACY == assembly_level);

   ConstructCoarseOperatorAndSolver(fes_hierarchy.GetFESpaceAtLevel(0));
   for (int l = 1; l < fes_hierarchy.GetNumLevels(); ++l)
   {
      ConstructOperatorAndSmoother(fes_hierarchy.GetFESpaceAtLevel(l), l);
   }
}

void AbsL1GeometricMultigrid::ConstructCoarseOperatorAndSolver(
   ParFiniteElementSpace& coarse_fespace)
{
   ConstructBilinearForm(coarse_fespace);

   OperatorPtr coarse_mat;
   coarse_mat.SetType(Operator::ANY_TYPE);
   bfs[0]->FormSystemMatrix(*essentialTrueDofs[0], coarse_mat);
   coarse_mat.SetOperatorOwner(false);

   // Create smoother
   Vector local_ones(coarse_mat->Height());
   Vector result(coarse_mat->Height());

   local_ones = 1.0;
   coarse_mat->AbsMult(local_ones, result);

   coarse_pc = new OperatorJacobiSmoother(result, *essentialTrueDofs[0]);

   Solver* coarse_solver = nullptr;
   switch (solver_type)
   {
      case sli:
         coarse_solver = new SLISolver(MPI_COMM_WORLD);
         break;
      case cg:
         coarse_solver = new CGSolver(MPI_COMM_WORLD);
         break;
      default:
         mfem_error("Invalid solver type!");
   }
   coarse_solver->SetOperator(*coarse_mat);

   IterativeSolver *it_solver = dynamic_cast<IterativeSolver*>(coarse_solver);
   if (it_solver)
   {
      it_solver->SetRelTol(MG_REL_TOL);
      it_solver->SetMaxIter(MG_MAX_ITER);
      it_solver->SetPrintLevel(-1);
      it_solver->SetPreconditioner(*coarse_pc);
   }

   AddLevel(coarse_mat.Ptr(), coarse_solver, mg_owned, true);
}

void AbsL1GeometricMultigrid::ConstructOperatorAndSmoother(
   ParFiniteElementSpace& fespace, int level)
{
   const Array<int> &ess_tdof_list = *essentialTrueDofs[level];
   ConstructBilinearForm(fespace);

   OperatorPtr level_mat;
   level_mat.SetType(Operator::ANY_TYPE);
   bfs.Last()->FormSystemMatrix(ess_tdof_list, level_mat);
   level_mat.SetOperatorOwner(false);

   // Create smoother
   Vector local_ones(level_mat->Height());
   Vector result(level_mat->Height());

   local_ones = 1.0;
   level_mat->AbsMult(local_ones, result);

   Solver* smoother = new OperatorJacobiSmoother(result, ess_tdof_list);

   AddLevel(level_mat.Ptr(), smoother, mg_owned, true);
}

void AbsL1GeometricMultigrid::ConstructBilinearForm(
   ParFiniteElementSpace &fespace)
{
   ParBilinearForm* form = new ParBilinearForm(&fespace);
   form->SetAssemblyLevel(assembly_level);
   switch (integrator_type)
   {
      case mass:
         form->AddDomainIntegrator(new MassIntegrator(one));
         break;
      case diffusion:
         form->AddDomainIntegrator(new DiffusionIntegrator(one));
         break;
      case maxwell:
         form->AddDomainIntegrator(new CurlCurlIntegrator(one));
         form->AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;
      default:
         mfem_error("Invalid integrator type! Check ParBilinearForm");
   }
   form->Assemble();
   bfs.Append(form);
}


void AssembleElementLpqJacobiDiag(ParBilinearForm& form, real_t p, real_t q,
                                  Vector& diag)
{
   ParBilinearForm temp_form(form.ParFESpace());
   temp_form.AllocateMatrix();
   for (int i = 0; i < form.ParFESpace()->GetNE(); ++i)
   {
      DenseMatrix emat_i;
      form.ComputeElementMatrix(i, emat_i);
      Vector right(emat_i.Height());
      Vector temp(emat_i.Height());
      Vector left(emat_i.Height());

      DenseMatrix temp_emat_i = emat_i;
      for (int j = 0; j < emat_i.Height(); ++j)
      {
         for (int k = 0; k < emat_i.Width(); ++k)
         {
            temp_emat_i(j, k) = std::pow(std::abs(emat_i(j, k)), p);
         }
      }

      if (q!=0.0)
      {
         emat_i.GetDiag(right);
         right.Abs();
         right.Pow(q);
      }
      else
      {
         right = 1.0;
      }

      temp_emat_i.Mult(right, temp);

      if (1.0 + q - p!= 0.0)
      {
         emat_i.GetDiag(left);
         left.Abs();
         left.Pow(1.0 + q - p);
         left *= temp;
      }
      else
      {
         left = temp;
      }

      temp_emat_i.Clear();
      temp_emat_i.Diag(left.GetData(), left.Size());
      temp_form.AssembleElementMatrix(i, temp_emat_i, 1);
   }
   temp_form.Finalize();
   auto mat = temp_form.ParallelAssemble();
   mat->AssembleDiagonal(diag);
   delete mat;
}

real_t diffusion_solution(const Vector &x)
{
   if (dim == 3)
   {
      return sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2)) + 1.0;
   }
   else
   {
      return sin(kappa * x(0)) * sin(kappa * x(1)) + 1.0;
   }
}

real_t diffusion_source(const Vector &x)
{
   if (dim == 3)
   {
      return dim * kappa * kappa * sin(kappa * x(0)) * sin(kappa * x(1)) *
             sin(kappa * x(2));
   }
   else
   {
      return dim * kappa * kappa * sin(kappa * x(0)) * sin(kappa * x(1));
   }
}

void maxwell_solution(const Vector &x, Vector &u)
{
   if (dim == 3)
   {
      u(0) = sin(kappa * x(1));
      u(1) = sin(kappa * x(2));
      u(2) = sin(kappa * x(0));
   }
   else
   {
      u(0) = sin(kappa * x(1));
      u(1) = sin(kappa * x(0));
      if (x.Size() == 3) { u(2) = 0.0; }
   }
}

void maxwell_source(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}

} // end namespace ds_common
