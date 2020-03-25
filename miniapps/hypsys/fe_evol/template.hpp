#ifndef HYPSYS_TEMPLATE_EVOLUTION
#define HYPSYS_TEMPLATE_EVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class TEMPLATE : public FE_Evolution
{
public:
   explicit TEMPLATE(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                     DofInfo &dofs_);

   virtual ~TEMPLATE() { }

   void Mult(const Vector&x, Vector &y) const override;
   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
};

#endif
