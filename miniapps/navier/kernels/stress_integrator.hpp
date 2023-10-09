#include <mfem.hpp>

namespace mfem
{
class StressIntegrator : public BilinearFormIntegrator
{
public:
   StressIntegrator(Coefficient &q)
      : Q(&q) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   Coefficient *Q = nullptr;
   DenseMatrix dshape, S, pelmat;
   int vdim = -1;
   int dim;
};
}