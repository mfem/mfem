#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <list>

using namespace mfem;
using namespace std;

class DGFaceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *rho;
   VectorCoefficient *u;
   double alpha, beta;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;             ///< Not owned
   const FaceGeometricFactors *geom;  ///< Not owned
   int dim, nf, nq, dofs1D, quad1D;

private:
   Vector shape1, shape2;

public:
   /// Construct integrator with rho = 1.
   DGFaceIntegrator(VectorCoefficient &_u, double a, double b)
   { u = &_u; alpha = a; beta = b;}
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};


/** Class for boundary integration of the linear form:
    (alpha/2) < (u.n) u_D, w > - beta < |u.n| u_D, w >,
    where u_D and u are given scalar and vector coefficients, respectively,
    and w is the scalar test function. */
class BoundaryAdvectIntegrator : public LinearFormIntegrator
{
private:
   Coefficient *uD;
   VectorCoefficient *u;
   double alpha, beta;
   Vector shape;

public:
   BoundaryAdvectIntegrator(Coefficient &_uD, VectorCoefficient &_u,
                          double a, double b)
   { uD = &_uD; u = &_u; alpha = a; beta = b; }
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

