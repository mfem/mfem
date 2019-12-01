#ifndef MFEM_ADVECTION
#define MFEM_ADVECTION

#include "../lib/hypsys.hpp"

class Advection
{
public:
   double t0 = 0.;
   double tFinal;
   bool SolutionKnown = true;
   bool WriteErrors = false;

   SparseMatrix K;
   Vector b;

   explicit Advection(const Vector &bbmin, const Vector &bbmax, const int config,
                      const double tEnd);
   virtual ~Advection() { };

   void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u);
   void PostprocessProblem(const GridFunction &u, Array<double> &errors);
};

#endif
