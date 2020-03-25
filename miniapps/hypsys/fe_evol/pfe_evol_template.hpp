#ifndef HYPSYS_PAR_TEMPLATE_EVOLUTION
#define HYPSYS_PAR_TEMPLATE_EVOLUTION

#include "fe_evol_template.hpp"
#include "pdofs.hpp"
#include "ptools.hpp"

using namespace std;
using namespace mfem;

class PAR_TEMPLATE : public TEMPLATE
{
public:
   explicit PAR_TEMPLATE(ParFiniteElementSpace *fes_,
                         HyperbolicSystem *hyp_,
                         DofInfo &dofs_);

   virtual ~PAR_TEMPLATE() { }

   void Mult(const Vector&x, Vector &y) const override;
   double ConvergenceCheck(double dt, double tol, const Vector &u) const override;
};

#endif
