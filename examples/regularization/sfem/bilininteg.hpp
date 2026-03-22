#include "mfem.hpp"

namespace mfem
{

/** Class for mass matrix assembling $(Q u ⋅ W, v ⋅ W)$ restricted to the
 *  boundary of a domain */
class BoundaryProjectionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient &Q;
   VectorCoefficient *W;

#ifndef MFEM_THREAD_SAFE
   // values of all scalar basis functions for one component of u (which is a
   // vector) at the integration point in the reference space
   Vector shape1;
   Vector w;
#endif

public:
   BoundaryProjectionIntegrator(Coefficient &q) : Q(q), W(NULL) { }

   BoundaryProjectionIntegrator(Coefficient &q, VectorCoefficient &w)
      : Q(q), W(&w) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

} // namespace mfem
