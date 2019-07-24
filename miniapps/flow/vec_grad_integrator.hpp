#include "mfem.hpp"

namespace mfem
{
class VectorGradientIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix Jadj;
   DenseMatrix elmat_comp;

public:
   VectorGradientIntegrator() { Q = NULL; }
   VectorGradientIntegrator(Coefficient *_q) { Q = _q; }
   VectorGradientIntegrator(Coefficient &q) { Q = &q; }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

} // namespace mfem
