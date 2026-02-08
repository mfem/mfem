#include "mfem.hpp"

namespace mfem
{

/// Class for boundary integration $ L(v) = (g, v \cdot w) $
class BoundaryProjectionLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   Coefficient &Q;
   Vector &W;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient qg
   /// and Vector w
   BoundaryProjectionLFIntegrator(Coefficient &qg, Vector &w,
      int a = 1, int b = 1) : Q(qg), W(w), oa(a), ob(b) { }

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;
   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override;

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for domain integrator $ L(v) := (f, ε(v)) $, where
    $ f = (f_{1x},f_{1y},f_{1z},\dots,f_{nx},f_{ny},f_{nz}) $,
    $ v = (v_1,\dots,v_n) $, and $ ε(v) = (1/2) (\nabla v + \nabla v^T)$. */
class VectorDomainLFStrainIntegrator : public LinearFormIntegrator
{
private:
   Vector shape, Qvec;
   VectorCoefficient &Q;
   DenseMatrix dshape;

public:
   /// Constructs the domain integrator (Q, (1/2) (grad v + grad v^T))
   VectorDomainLFStrainIntegrator(VectorCoefficient &QF) : Q(QF) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override;

   using LinearFormIntegrator::AssembleRHSElementVect;
};

} // namespace mfem
