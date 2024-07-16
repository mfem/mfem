
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

// Custom monitor that prints a csv-like file
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
                             IntegratorType it)
      : GeometricMultigrid(fes_hierarchy, ess_bdr), one(1.0), integrator_type(it)
   {
      ConstructCoarseOperatorAndSolver(fes_hierarchy.GetFESpaceAtLevel(0));
      for (int l = 1; l < fes_hierarchy.GetNumLevels(); ++l)
      {
         ConstructOperatorAndSmoother(fes_hierarchy.GetFESpaceAtLevel(l), l);
      }
   }

   ~GeneralGeometricMultigrid()
   {
      delete solver;
   }

private:
   ConstantCoefficient one;
   Solver* solver;
   HypreBoomerAMG* amg;
   IntegratorType integrator_type;

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace& coarse_fespace)
   {
      ConstructBilinearForm(coarse_fespace, false);

      HypreParMatrix* coarse_mat = new HypreParMatrix();
      bfs[0]->FormSystemMatrix(*essentialTrueDofs[0], *coarse_mat);

      // Here, AMG comes is a preconditioner as a member
      amg = new HypreBoomerAMG(*coarse_mat);
      amg->SetPrintLevel(-1);

      CGSolver* solver = new CGSolver(MPI_COMM_WORLD);
      solver->SetPrintLevel(-1);
      solver->SetMaxIter(10);
      solver->SetRelTol(sqrt(1e-4));
      solver->SetAbsTol(0.0);
      solver->SetOperator(*coarse_mat);
      solver->SetPreconditioner(*amg);

      // Last two variables transfer ownership of the pointers
      // Operator and solver
      AddLevel(coarse_mat, solver, true, true);
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace, int level)
   {
      const Array<int> &ess_tdof_list = *essentialTrueDofs[level];
      ConstructBilinearForm(fespace, true);

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      bfs.Last()->FormSystemMatrix(ess_tdof_list, opr);
      opr.SetOperatorOwner(false);

      Vector diag(fespace.GetTrueVSize());
      bfs.Last()->AssembleDiagonal(diag);

      Solver* smoother = new OperatorChebyshevSmoother(
         *opr, diag, ess_tdof_list, 2, fespace.GetParMesh()->GetComm());

      AddLevel(opr.Ptr(), smoother, true, true);
   }


   // PUt later
   void ConstructBilinearForm(ParFiniteElementSpace& fespace,
                              bool partial_assembly)
   {
      ParBilinearForm* form = new ParBilinearForm(&fespace);
      if (partial_assembly)
      {
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
      form->Assemble();
      bfs.Append(form);
   }
};

}
#endif // MFEM_LPQ_JACOBI_HPP
