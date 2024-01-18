#ifndef MTOP_HOMOGENIZATION_HPP
#define MTOP_HOMOGENIZATION_HPP

#include "mfem.hpp"
#include "mtop_solvers.hpp"

namespace mfem
{

class HomogenizationElast
{
public:
   HomogenizationElast(mfem::ParMesh* mesh_, int vorder=1);

   ~HomogenizationElast();

   /// Set the Linear Solver
   void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1000,
                        int prt_level=1);

   /// Solves the forward problem.
   void FSolve();

   DenseMatrix& HomogenizedTensor();

   /// Returns the displacements.
   mfem::ParGridFunction& GetDisplacements(int case_id);

private:
   mfem::ParMesh* pmesh;

   //solution true vector
   mfem::Vector sol[6];

   //RHS
   mfem::Vector rhs;

   //Linear solver parameters
   double linear_rtol;
   double linear_atol;
   int linear_iter;
   int print_level;

   mfem::HypreBoomerAMG *prec; //preconditioner
   mfem::CGSolver *ls;  //linear solver

   mfem::ParNonlinearForm *nf;
   mfem::ParFiniteElementSpace* vfes;
   mfem::FiniteElementCollection* vfec;

   std::vector<mfem::BasicElasticityCoefficient*> materials;
};


};

#endif
