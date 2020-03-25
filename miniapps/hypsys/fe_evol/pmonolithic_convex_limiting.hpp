#ifndef HYPSYS_PARMCL_EVOLUTION
#define HYPSYS_PARMCL_EVOLUTION

#include "monolithic_convex_limiting.hpp"
#include "../lib/pdofs.hpp"
#include "../lib/ptools.hpp"

using namespace std;
using namespace mfem;

class ParMCL_Evolution : public MCL_Evolution
{
public:
   mutable ParGridFunction x_gf_MPI;

   explicit ParMCL_Evolution(ParFiniteElementSpace *pfes_,
                             HyperbolicSystem *hyp_, DofInfo &dofs_);

   virtual ~ParMCL_Evolution() { }

   void Mult(const Vector&x, Vector &y) const override;
   double ConvergenceCheck(double dt, double tol, const Vector &u) const override;
};

#endif
