#ifndef HYPSYS_MCL_EVOLUTION
#define HYPSYS_MCL_EVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class MCL_Evolution : public FE_Evolution
{
public:
   DenseTensor PrecGrad;
   DenseMatrix MassMatLOR;

   explicit MCL_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                          DofInfo &dofs_);

   virtual ~MCL_Evolution() { }

   void Mult(const Vector&x, Vector &y) const override;
   virtual void ElemEval(const Vector &uElem, Vector &uEval, int k) const override;
   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
};

#endif
