#ifndef MFEM_ADVECTION
#define MFEM_ADVECTION

#include "../lib/hypsys.hpp"

class Advection : public HyperbolicSystem
{
public:
   double t0 = 0.;
// 	double t;
   double tFinal;
   bool SolutionKnown = true;
   bool WriteErrors = false;

	FiniteElementSpace *fes;
   SparseMatrix K;
   Vector b;

   explicit Advection(FiniteElementSpace *fes_, const int config, const double tEnd,
							 const Vector &bbmin, const Vector &bbmax);
   ~Advection() { };

   virtual void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u);
};

#endif
