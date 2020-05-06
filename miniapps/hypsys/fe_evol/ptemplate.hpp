#ifndef HYPSYS_PARTEMPLATE_EVOLUTION
#define HYPSYS_PARTEMPLATE_EVOLUTION

#include "template.hpp"
#include "../lib/plib.hpp"

using namespace std;
using namespace mfem;

class PARTEMPLATE : public TEMPLATE
{
public:
   mutable ParGridFunction x_gf_MPI;

   explicit PARTEMPLATE(ParFiniteElementSpace *pfes_,
                        HyperbolicSystem *hyp_,
                        DofInfo &dofs_);

   virtual ~PARTEMPLATE() { }

   void Mult(const Vector&x, Vector &y) const override;
   double ConvergenceCheck(double dt, double tol, const Vector &u) const override;
};

#endif
