#ifndef HYPSYS_GALERKINEVOLUTION
#define HYPSYS_GALERKINEVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class GalerkinEvolution : public FE_Evolution
{
public:
   explicit GalerkinEvolution(FiniteElementSpace *fes_,
                              HyperbolicSystem *hyp_, DofInfo &dofs_);

   virtual ~GalerkinEvolution() { }

   void Mult(const Vector&x, Vector &y) const override;
   virtual void ElemEval(const Vector &uElem, Vector &uEval, int k) const override;
   virtual void FaceEval(const Vector &x, Vector &y1, Vector &y2,
                         const Vector &xMPI, const Vector &normal,
                         int e, int i, int k) const override;
   virtual void LaxFriedrichs(const Vector &x1, const Vector &x2, const Vector &normal,
                              Vector &y, int e, int k, int i) const override;
   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
};

#endif