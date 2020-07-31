#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <list>

using namespace mfem;
using namespace std;
/// Class for domain integration L(v) := (f, v)
double CutComputeL2Error( GridFunction &x, FiniteElementSpace *fes,
    Coefficient &exsol, double scale);
class CutDomainLFIntegrator : public DeltaLFIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob, nels;
   double scale;
public:
   /// Constructs a domain integrator with a given Coefficient
   CutDomainLFIntegrator(Coefficient &QF, double scaling, int ne, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is OK
      : DeltaLFIntegrator(QF), Q(QF), scale(scaling), nels(ne), oa(a), ob(b) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void AssembleDeltaElementVect(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// alpha (q . grad v, u)
class AdvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *Q;
   double alpha;
   double scale;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D, nels;

private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif

public:
   AdvectionIntegrator(VectorCoefficient &q, double scaling, int ne, double a = 1.0)
      : Q(&q) { scale = scaling; nels = ne; alpha = a;}
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

class DGFaceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *rho;
   VectorCoefficient *u;
   double alpha, beta, scale;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;             ///< Not owned
   const FaceGeometricFactors *geom;  ///< Not owned
   int dim, nf, nq, dofs1D, quad1D, nels;

private:
   Vector shape1, shape2;

public:
   /// Construct integrator with rho = 1.
   DGFaceIntegrator(VectorCoefficient &_u, double a, double b, double scaling, int ne)
   { u = &_u; alpha = a; beta = b; scale = scaling; nels= ne;}
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
   double alpha, beta, scale;
   int nels;
   Vector shape;

public:
   BoundaryAdvectIntegrator(Coefficient &_uD, VectorCoefficient &_u,
                          double a, double b, int ne, double scaling)
   { uD = &_uD; u = &_u; alpha = a; beta = b; nels= ne; scale =scaling;}
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};