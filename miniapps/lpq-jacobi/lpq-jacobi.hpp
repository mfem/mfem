// Standard header
// This needs to be separated into an hpp and a cpp

#ifndef MFEM_LPQ_JACOBI_HPP
#define MFEM_LPQ_JACOBI_HPP

#include "mfem.hpp"
#include "miniapps/common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

namespace lpq_jacobi
{

int NDIGITS = 20;
int MG_MAX_ITER = 10000;
real_t MG_REL_TOL = 1e-5;

int dim;
int space_dim;
real_t freq = 1.0;
real_t kappa;

// Enumerator for the different solvers to implement
enum SolverType
{
   sli,
   cg,
   num_solvers,  // last
};

// Enumerator for the different integrators to implement
enum IntegratorType
{
   mass,
   diffusion,
   elasticity,
   maxwell,
   num_integrators,  // last
};

// Custom monitor that prints a csv-formatted file
class DataMonitor : public IterativeSolverMonitor
{
private:
   ofstream os;
   int precision;
public:
   DataMonitor(string file_name, int ndigits) : os(file_name), precision(ndigits)
   {
      if (Mpi::Root())
      {
         mfem::out << "Saving iterations into: " << file_name << endl;
      }
      os << "it,res,sol" << endl;
      os << fixed << setprecision(precision);
   }
   void MonitorResidual(int it, real_t norm, const Vector &x, bool final)
   {
      os << it << "," << norm << ",";
   }
   void MonitorSolution(int it, real_t norm, const Vector &x, bool final)
   {
      os << norm << endl;
   }
};

// Custom general geometric multigrid method, derived from GeometricMultigrid
class GeneralGeometricMultigrid : public GeometricMultigrid
{
public:
   // Constructor
   GeneralGeometricMultigrid(ParFiniteElementSpaceHierarchy& fes_hierarchy,
                             Array<int>& ess_bdr,
                             IntegratorType it,
                             SolverType st,
                             real_t p_order,
                             real_t q_order)
      : GeometricMultigrid(fes_hierarchy, ess_bdr),
        integrator_type(it),
        solver_type(st),
        p_order(p_order),
        q_order(q_order),
        coarse_solver(nullptr),
        coarse_pc(nullptr),
        one(1.0),
        level_mats()
   {
      ConstructCoarseOperatorAndSolver(fes_hierarchy.GetFESpaceAtLevel(0));
      for (int l = 1; l < fes_hierarchy.GetNumLevels(); ++l)
      {
         ConstructOperatorAndSmoother(fes_hierarchy.GetFESpaceAtLevel(l), l);
      }
   }

   const Array<int>* ReturnLastEssentialTrueDofs() { return essentialTrueDofs.Last(); }

   ~GeneralGeometricMultigrid() { delete coarse_pc; }

private:
   IntegratorType integrator_type;
   SolverType solver_type;
   real_t p_order;
   real_t q_order;
   Solver* coarse_solver;
   OperatorLpqJacobiSmoother* coarse_pc;
   ConstantCoefficient one;
   Array<HypreParMatrix*> level_mats;

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace& coarse_fespace)
   {
      ConstructBilinearForm(coarse_fespace, false);

      HypreParMatrix* coarse_mat = new HypreParMatrix();
      bfs[0]->FormSystemMatrix(*essentialTrueDofs[0], *coarse_mat);

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

      coarse_pc = new OperatorLpqJacobiSmoother(*coarse_mat,
                                                *essentialTrueDofs[0],
                                                p_order,
                                                q_order);

      IterativeSolver *it_solver = dynamic_cast<IterativeSolver *>(coarse_solver);
      if (it_solver)
      {
         it_solver->SetRelTol(MG_REL_TOL);
         it_solver->SetMaxIter(MG_MAX_ITER);
         it_solver->SetPrintLevel(1);
         it_solver->SetPreconditioner(*coarse_pc);
      }
      coarse_solver->SetOperator(*coarse_mat);

      // Last two variables transfer ownership of the pointers operator and solver
      AddLevel(coarse_mat, coarse_solver, true, true);
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace, int level)
   {
      const Array<int> &ess_tdof_list = *essentialTrueDofs[level];
      ConstructBilinearForm(fespace, false);

      level_mats.Append(new HypreParMatrix());
      bfs.Last()->FormSystemMatrix(ess_tdof_list, *level_mats.Last());

      Solver* smoother = new OperatorLpqJacobiSmoother(*level_mats.Last(),
                                                       ess_tdof_list,
                                                       p_order,
                                                       q_order);

      AddLevel(level_mats.Last(), smoother, true, true);
   }


   void ConstructBilinearForm(ParFiniteElementSpace& fespace,
                              bool partial_assembly = true)
   {
      ParBilinearForm* form = new ParBilinearForm(&fespace);

      if (partial_assembly)
      {
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }

      switch (integrator_type)
      {
         case mass:
            form->AddDomainIntegrator(new MassIntegrator);
            break;
         case diffusion:
            form->AddDomainIntegrator(new DiffusionIntegrator);
            break;
         case elasticity:
            form->AddDomainIntegrator(new ElasticityIntegrator(one, one));
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
};

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
      return dim * kappa * kappa * sin(kappa * x(0)) * sin(kappa * x(1)) * sin(
                kappa * x(2));
   }
   else
   {
      return dim * kappa * kappa * sin(kappa * x(0)) * sin(kappa * x(1));
   }
}

void elasticity_solution(const Vector &x, Vector &u)
{
   if (dim == 3)
   {
      u(0) = sin(kappa * x(0));
      u(1) = sin(kappa * x(1));
      u(2) = sin(kappa * x(2));
   }
   else
   {
      u(0) = sin(kappa * x(0));
      u(1) = sin(kappa * x(1));
      if (x.Size() == 3) { u(2) = 0.0; }
   }
}

void elasticity_source(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = - 3.0 * kappa * kappa * sin(kappa * x(0));
      f(1) = - 3.0 * kappa * kappa * sin(kappa * x(1));
      f(2) = - 3.0 * kappa * kappa * sin(kappa * x(2));
   }
   else
   {
      f(0) = - 3.0 * kappa * kappa * sin(kappa * x(0));
      f(1) = - 3.0 * kappa * kappa * sin(kappa * x(1));
      if (x.Size() == 3) { f(2) = 0.0; }
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

} // end namespace lpq_jacobi
#endif // MFEM_LPQ_JACOBI_HPP
