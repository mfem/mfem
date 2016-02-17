// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BILININTEG
#define MFEM_BILININTEG

#include "../config/config.hpp"
#include "nonlininteg.hpp"

namespace mfem
{

/// Abstract base class BilinearFormIntegrator
class BilinearFormIntegrator : public NonlinearFormIntegrator
{
protected:
   const IntegrationRule *IntRule;

   BilinearFormIntegrator(const IntegrationRule *ir = NULL)
   { IntRule = ir; }

public:
   /// Given a particular Finite Element computes the element matrix elmat.
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   /** Compute the local matrix representation of a bilinear form
       a(u,v) defined on different trial (given by u) and test
       (given by v) spaces. The rows in the local matrix correspond
       to the test dofs and the columns -- to the trial dofs. */
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   /** Abstract method used for assembling TraceFaceIntegrators in a
       MixedBilinearForm. */
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   /// Perform the local action of the BilinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat)
   { AssembleElementMatrix(el, Tr, elmat); }

   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u,
                                   const FiniteElement &fluxelem,
                                   Vector &flux, int with_coef = 1) { }

   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL)
   { return 0.0; }

   void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   virtual ~BilinearFormIntegrator() { }
};

class TransposeIntegrator : public BilinearFormIntegrator
{
private:
   int own_bfi;
   BilinearFormIntegrator *bfi;

   DenseMatrix bfi_elmat;

public:
   TransposeIntegrator (BilinearFormIntegrator *_bfi, int _own_bfi = 1)
   { bfi = _bfi; own_bfi = _own_bfi; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   virtual ~TransposeIntegrator() { if (own_bfi) { delete bfi; } }
};

class LumpedIntegrator : public BilinearFormIntegrator
{
private:
   int own_bfi;
   BilinearFormIntegrator *bfi;

public:
   LumpedIntegrator (BilinearFormIntegrator *_bfi, int _own_bfi = 1)
   { bfi = _bfi; own_bfi = _own_bfi; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~LumpedIntegrator() { if (own_bfi) { delete bfi; } }
};

/// Integrator that inverts the matrix assembled by another integrator.
class InverseIntegrator : public BilinearFormIntegrator
{
private:
   int own_integrator;
   BilinearFormIntegrator *integrator;

public:
   InverseIntegrator(BilinearFormIntegrator *integ, int own_integ = 1)
   { integrator = integ; own_integrator = own_integ; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~InverseIntegrator() { if (own_integrator) { delete integrator; } }
};

/// Integrator defining a sum of multiple Integrators.
class SumIntegrator : public BilinearFormIntegrator
{
private:
   int own_integrators;
   DenseMatrix elem_mat;
   Array<BilinearFormIntegrator*> integrators;

public:
   SumIntegrator(int own_integs = 1) { own_integrators = own_integs; }

   void AddIntegrator(BilinearFormIntegrator *integ)
   { integrators.Append(integ); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~SumIntegrator();
};

/** Class for integrating the bilinear form a(u,v) := (Q grad u, grad v)
    where Q can be a scalar or a matrix coefficient. */
class DiffusionIntegrator: public BilinearFormIntegrator
{
private:
   Vector vec, pointflux, shape;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, dshapedxt, invdfdx, mq;
   DenseMatrix te_dshape, te_dshapedxt;
#endif
   Coefficient *Q;
   MatrixCoefficient *MQ;

public:
   /// Construct a diffusion integrator with coefficient Q = 1
   DiffusionIntegrator() { Q = NULL; MQ = NULL; }

   /// Construct a diffusion integrator with a scalar coefficient q
   DiffusionIntegrator (Coefficient &q) : Q(&q) { MQ = NULL; }

   /// Construct a diffusion integrator with a matrix coefficient q
   DiffusionIntegrator (MatrixCoefficient &q) : MQ(&q) { Q = NULL; }

   /** Given a particular Finite Element
       computes the element stiffness matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   /** Given a trial and test Finite Element computes the element stiffness
       matrix elmat. */
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   /// Perform the local action of the BilinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u, const FiniteElement &fluxelem,
                                   Vector &flux, int with_coef = 1);

   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL);
};

/** Class for local mass matrix assembling a(u,v) := (Q u, v) */
class MassIntegrator: public BilinearFormIntegrator
{
private:
   Vector shape, te_shape;
   Coefficient *Q;

public:
   MassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { Q = NULL; }
   /// Construct a mass integrator with coefficient q
   MassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(&q) { }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

class BoundaryMassIntegrator : public MassIntegrator
{
public:
   BoundaryMassIntegrator(Coefficient &q) : MassIntegrator(q) { }
};

/// alpha (q . grad u, v)
class ConvectionIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
   VectorCoefficient &Q;
   double alpha;

public:
   ConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

/// alpha (q . grad u, v) using the "group" FE discretization
class GroupConvectionIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix dshape, adjJ, Q_nodal, grad;
   Vector shape;
   VectorCoefficient &Q;
   double alpha;

public:
   GroupConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

/** Class for integrating the bilinear form a(u,v) := (Q u, v),
    where u=(u1,...,un) and v=(v1,...,vn); ui and vi are defined
    by scalar FE through standard transformation. */
class VectorMassIntegrator: public BilinearFormIntegrator
{
private:
   Vector shape, te_shape, vec;
   DenseMatrix partelmat;
   DenseMatrix mcoeff;
   Coefficient *Q;
   VectorCoefficient *VQ;
   MatrixCoefficient *MQ;

   int Q_order;

public:
   /// Construct an integrator with coefficient 1.0
   VectorMassIntegrator()
   { Q = NULL; VQ = NULL; MQ = NULL; Q_order = 0; }
   /** Construct an integrator with scalar coefficient q.
       If possible, save memory by using a scalar integrator since
       the resulting matrix is block diagonal with the same diagonal
       block repeated. */
   VectorMassIntegrator(Coefficient &q, int qo = 0)
      : Q(&q) { VQ = NULL; MQ = NULL; Q_order = qo; }
   VectorMassIntegrator(Coefficient &q, const IntegrationRule *ir)
      : BilinearFormIntegrator(ir), Q(&q)
   { VQ = NULL; MQ = NULL; Q_order = 0; }
   /// Construct an integrator with diagonal coefficient q
   VectorMassIntegrator(VectorCoefficient &q, int qo = 0)
      : VQ(&q) { Q = NULL; MQ = NULL; Q_order = qo; }
   /// Construct an integrator with matrix coefficient q
   VectorMassIntegrator(MatrixCoefficient &q, int qo = 0)
      : MQ(&q) { Q = NULL; VQ = NULL; Q_order = qo; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};


/** Class for integrating (div u, p) where u is a vector field given
    by VectorFiniteElement through Piola transformation (for RT
    elements); p is scalar function given by FiniteElement through
    standard transformation. Here, u is the trial function and p is
    the test function.
    Note: the element matrix returned by AssembleElementMatrix2
    does NOT depend on the ElementTransformation Trans. */
class VectorFEDivergenceIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;
#ifndef MFEM_THREAD_SAFE
   Vector divshape, shape;
#endif
public:
   VectorFEDivergenceIntegrator() { Q = NULL; }
   VectorFEDivergenceIntegrator(Coefficient &q) { Q = &q; }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};


/// Integrator for (curl u, v) for Nedelec and RT elements
class VectorFECurlIntegrator: public BilinearFormIntegrator
{
private:
   Coefficient *Q;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix curlshapeTrial;
   DenseMatrix vshapeTest;
   DenseMatrix curlshapeTrial_dFT;
#endif
public:
   VectorFECurlIntegrator() { Q = NULL; }
   VectorFECurlIntegrator(Coefficient &q) { Q = &q; }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};


/// Class for integrating (Q D_i(u), v); u and v are scalars
class DerivativeIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient & Q;
   int xi;
   DenseMatrix dshape, dshapedxt, invdfdx;
   Vector shape, dshapedxi;
public:
   DerivativeIntegrator(Coefficient &q, int i) : Q(q), xi(i) { }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat)
   { AssembleElementMatrix2(el,el,Trans,elmat); }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// Integrator for (curl u, curl v) for Nedelec elements
class CurlCurlIntegrator: public BilinearFormIntegrator
{
private:
   Vector vec, pointflux;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix curlshape, curlshape_dFt;
   DenseMatrix vshape, projcurl;
#endif
   Coefficient *Q;

public:
   CurlCurlIntegrator() { Q = NULL; }
   /// Construct a bilinear form integrator for Nedelec elements
   CurlCurlIntegrator(Coefficient &q) : Q(&q) { }

   /* Given a particular Finite Element, compute the
      element curl-curl matrix elmat */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u, const FiniteElement &fluxelem,
                                   Vector &flux, int with_coef);

   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL);
};

/** Integrator for (curl u, curl v) for FE spaces defined by 'dim' copies of a
    scalar FE space. */
class VectorCurlCurlIntegrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape_hat, dshape, curlshape, Jadj, grad_hat, grad;
#endif
   Coefficient *Q;

public:
   VectorCurlCurlIntegrator() { Q = NULL; }

   VectorCurlCurlIntegrator(Coefficient &q) : Q(&q) { }

   /// Assemble an element matrix
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   /// Compute element energy: (1/2) (curl u, curl u)_E
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);
};

/// Integrator for (Q u, v) for VectorFiniteElements
class VectorFEMassIntegrator: public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   VectorCoefficient *VQ;
   MatrixCoefficient *MQ;
   void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
   { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
   Vector shape;
   Vector D;
   DenseMatrix K;
   DenseMatrix test_vshape;
   DenseMatrix trial_vshape;
#endif

public:
   VectorFEMassIntegrator() { Init(NULL, NULL, NULL); }
   VectorFEMassIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
   VectorFEMassIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
   VectorFEMassIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
   VectorFEMassIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
   VectorFEMassIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
   VectorFEMassIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/** Integrator for (Q div u, p) where u=(v1,...,vn) and all
    vi are in the same scalar FE space; p is also in
    a (different) scalar FE space.  */
class VectorDivergenceIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;

   Vector shape;
   Vector divshape;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix Jadj;

public:
   VectorDivergenceIntegrator() { Q = NULL; }
   VectorDivergenceIntegrator(Coefficient *_q) { Q = _q; }
   VectorDivergenceIntegrator(Coefficient &q) { Q = &q; }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// (Q div u, div v) for RT elements
class DivDivIntegrator: public BilinearFormIntegrator
{
private:
   Coefficient *Q;

#ifndef MFEM_THREAD_SAFE
   Vector divshape;
#endif

public:
   DivDivIntegrator() { Q = NULL; }
   DivDivIntegrator(Coefficient &q) : Q(&q) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
};

/** Integrator for (Q grad u, grad v) = sum_i (Q grad u_i, grad v_i)
    for FE spaces defined by 'dim' copies of a scalar FE space. */
class VectorDiffusionIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;

   DenseMatrix Jinv;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix pelmat;

public:
   VectorDiffusionIntegrator() { Q = NULL; }
   VectorDiffusionIntegrator(Coefficient &q) { Q = &q; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
};

/** Integrator for the linear elasticity form:
    a(u,v) = (lambda div(u), div(v)) + (2 mu e(u), e(v)),
    where e(v) = (1/2) (grad(v) + grad(v)^T).
    This is a 'Vector' integrator, i.e. defined for FE spaces
    using multiple copies of a scalar FE space. */
class ElasticityIntegrator : public BilinearFormIntegrator
{
private:
   double q_lambda, q_mu;
   Coefficient *lambda, *mu;

#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, Jinv, gshape, pelmat;
   Vector divshape;
#endif

public:
   ElasticityIntegrator(Coefficient &l, Coefficient &m)
   { lambda = &l; mu = &m; }
   /** With this constructor lambda = q_l * m and mu = q_m * m;
       if dim * q_l + 2 * q_m = 0 then trace(sigma) = 0. */
   ElasticityIntegrator(Coefficient &m, double q_l, double q_m)
   { lambda = NULL; mu = &m; q_lambda = q_l; q_mu = q_m; }

   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

/** Integrator for the DG form:
    alpha < rho_u (u.n) {v},[w] > + beta < rho_u |u.n| [v],[w] >,
    where v and w are the trial and test variables, respectively, and rho/u are
    given scalar/vector coefficients. The vector coefficient, u, is assumed to
    be continuous across the faces and when given the scalar coefficient, rho,
    is assumed to be discontinuous. The integrator uses the upwind value of rho,
    rho_u, which is value from the side into which the vector coefficient, u,
    points. */
class DGTraceIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *rho;
   VectorCoefficient *u;
   double alpha, beta;

   Vector shape1, shape2;

public:
   /// Construct integrator with rho = 1.
   DGTraceIntegrator(VectorCoefficient &_u, double a, double b)
   { rho = NULL; u = &_u; alpha = a; beta = b; }

   DGTraceIntegrator(Coefficient &_rho, VectorCoefficient &_u,
                     double a, double b)
   { rho = &_rho; u = &_u; alpha = a; beta = b; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Integrator for the DG form:

    - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >
    + kappa < {h^{-1} Q} [u], [v] >,

    where Q is a scalar or matrix diffusion coefficient and u, v are the trial
    and test spaces, respectively. The parameters sigma and kappa determine the
    DG method to be used (when this integrator is added to the "broken"
    DiffusionIntegrator):
    * sigma = -1, kappa >= kappa0: symm. interior penalty (IP or SIPG) method,
    * sigma = +1, kappa > 0: non-symmetric interior penalty (NIPG) method,
    * sigma = +1, kappa = 0: the method of Baumann and Oden. */
class DGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
   double sigma, kappa;

   // these are not thread-safe!
   Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
   DGDiffusionIntegrator(const double s, const double k)
      : Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
   DGDiffusionIntegrator(Coefficient &q, const double s, const double k)
      : Q(&q), MQ(NULL), sigma(s), kappa(k) { }
   DGDiffusionIntegrator(MatrixCoefficient &q, const double s, const double k)
      : Q(NULL), MQ(&q), sigma(s), kappa(k) { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Integrator for the DPG form: < v, [w] > over all faces (the interface) where
    the trial variable v is defined on the interface and the test variable w is
    defined inside the elements, generally in a DG space. */
class TraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, shape1, shape2;

public:
   TraceJumpIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Integrator for the form: < v, [w.n] > over all faces (the interface) where
    the trial variable v is defined on the interface and the test variable w is
    in an H(div)-conforming space. */
class NormalTraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, normal, shape1_n, shape2_n;
   DenseMatrix shape1, shape2;

public:
   NormalTraceJumpIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Abstract class to serve as a base for local interpolators to be used in
    the DiscreteLinearOperator class. */
class DiscreteInterpolator : public BilinearFormIntegrator { };


/** Class for constructing the gradient as a DiscreteLinearOperator from an
    H1-conforming space to an H(curl)-conforming space. The range space can
    be vector L2 space as well. */
class GradientInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &h1_fe,
                                       const FiniteElement &nd_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { nd_fe.ProjectGrad(h1_fe, Trans, elmat); }
};


/** Class for constructing the identity map as a DiscreteLinearOperator. This
    is the discrete embedding matrix when the domain space is a subspace of
    the range space. Otherwise, a dof projection matrix is constructed. */
class IdentityInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { ran_fe.Project(dom_fe, Trans, elmat); }
};


/** Class for constructing the (local) discrete curl matrix which can be used
    as an integrator in a DiscreteLinearOperator object to assemble the global
    discrete curl matrix. */
class CurlInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { ran_fe.ProjectCurl(dom_fe, Trans, elmat); }
};


/** Class for constructing the (local) discrete divergence matrix which can
    be used as an integrator in a DiscreteLinearOperator object to assemble
    the global discrete divergence matrix.

    Note: Since the dofs in the L2_FECollection are nodal values, the local
    discrete divergence matrix (with an RT-type domain space) will depend on
    the transformation. On the other hand, the local matrix returned by
    VectorFEDivergenceIntegrator is independent of the transformation. */
class DivergenceInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { ran_fe.ProjectDiv(dom_fe, Trans, elmat); }
};


/** A trace face interpolator class for interpolating the normal component of
    the domain space, e.g. vector H1, into the range space, e.g. the trace of
    RT which uses FiniteElement::INTEGRAL map type. */
class NormalInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

}

#endif
