#include "mfem.hpp"

namespace mfem
{
class VectorConvectionNLFIntegrator : public NonlinearFormIntegrator
{
private:
   Coefficient *Q{};
   DenseMatrix dshape, dshapex, EF, gradEF, ELV, elmat_comp;
   Vector shape;

public:
   VectorConvectionNLFIntegrator(Coefficient &q) : Q(&q) {}
   VectorConvectionNLFIntegrator() = default;

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &trans,
                              const Vector &elfun,
                              Vector &elvect) override;

   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &trans,
                            const Vector &elfun,
                            DenseMatrix &elmat) override;
};
} // namespace mfem
