#ifndef HYPSYS_MCL_EVOLUTION
#define HYPSYS_MCL_EVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class MCL_Evolution : public FE_Evolution
{
public:
   DenseTensor PrecGradOp, GradProd;
   DenseMatrix MassMatLOR;

   int nscd; // nscd: NumSubcellCrossDofs

   mutable DenseTensor CTilde, CFull;
   mutable Vector C_eij;
   mutable Array<int> eldofs;

   explicit MCL_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                          DofInfo &dofs_);

   virtual ~MCL_Evolution() { }

   void Mult(const Vector&x, Vector &y) const override;
   virtual void ElemEval(const Vector &uElem, Vector &uEval, int k) const override;
   virtual void FaceEval(const Vector &x, Vector &y1, Vector &y2,
                         const Vector &xMPI, const Vector &normal,
                         int e, int i, int k) const override;
   virtual void LaxFriedrichs(const Vector &x1, const Vector &x2, const Vector &normal,
                              Vector &y, int e, int j, int i) const override;
   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
};

#endif
