// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LININTEG
#define MFEM_LININTEG

#include "../config/config.hpp"
#include "coefficient.hpp"
#include "bilininteg.hpp"
#include <random>

namespace mfem
{

/// Abstract base class LinearFormIntegrator
class LinearFormIntegrator
{
protected:
   const IntegrationRule *IntRule;

   LinearFormIntegrator(const IntegrationRule *ir = NULL) { IntRule = ir; }

public:

   /// Method probing for assembly on device
   virtual bool SupportsDevice() { return false; }

   /// Method defining assembly on device
   virtual void AssembleDevice(const FiniteElementSpace &fes,
                               const Array<int> &markers,
                               Vector &b);

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) = 0;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   virtual void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }
   const IntegrationRule* GetIntRule() { return IntRule; }

   virtual ~LinearFormIntegrator() { }
};


/// Abstract class for integrators that support delta coefficients
class DeltaLFIntegrator : public LinearFormIntegrator
{
protected:
   DeltaCoefficient *delta;
   VectorDeltaCoefficient *vec_delta;

   /** @brief This constructor should be used by derived classes that use a
       scalar DeltaCoefficient. */
   DeltaLFIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : LinearFormIntegrator(ir),
        delta(dynamic_cast<DeltaCoefficient*>(&q)),
        vec_delta(NULL) { }

   /** @brief This constructor should be used by derived classes that use a
       VectorDeltaCoefficient. */
   DeltaLFIntegrator(VectorCoefficient &vq,
                     const IntegrationRule *ir = NULL)
      : LinearFormIntegrator(ir),
        delta(NULL),
        vec_delta(dynamic_cast<VectorDeltaCoefficient*>(&vq)) { }

public:
   /// Returns true if the derived class instance uses a delta coefficient.
   bool IsDelta() const { return (delta || vec_delta); }

   /// Returns the center of the delta coefficient.
   void GetDeltaCenter(Vector &center)
   {
      if (delta) { delta->GetDeltaCenter(center); return; }
      if (vec_delta) { vec_delta->GetDeltaCenter(center); return; }
      center.SetSize(0);
   }

   /** @brief Assemble the delta coefficient at the IntegrationPoint set in
       @a Trans which is assumed to map to the delta coefficient center.

       @note This method should be called for one mesh element only, including
       in parallel, even when the center of the delta coefficient is shared by
       multiple elements. */
   virtual void AssembleDeltaElementVect(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         Vector &elvect) = 0;
};


/// Class for domain integration L(v) := (f, v)
class DomainLFIntegrator : public DeltaLFIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   DomainLFIntegrator(Coefficient &QF, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is OK
      : DeltaLFIntegrator(QF), Q(QF), oa(a), ob(b) { }

   /// Constructs a domain integrator with a given Coefficient
   DomainLFIntegrator(Coefficient &QF, const IntegrationRule *ir)
      : DeltaLFIntegrator(QF, ir), Q(QF), oa(1), ob(1) { }

   virtual bool SupportsDevice() { return true; }

   /// Method defining assembly on device
   virtual void AssembleDevice(const FiniteElementSpace &fes,
                               const Array<int> &markers,
                               Vector &b);

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

/// Class for domain integrator L(v) := (f, grad v)
class DomainLFGradIntegrator : public DeltaLFIntegrator
{
private:
   Vector shape, Qvec;
   VectorCoefficient &Q;
   DenseMatrix dshape;

public:
   /// Constructs the domain integrator (Q, grad v)
   DomainLFGradIntegrator(VectorCoefficient &QF)
      : DeltaLFIntegrator(QF), Q(QF) { }

   virtual bool SupportsDevice() { return true; }

   /// Method defining assembly on device
   virtual void AssembleDevice(const FiniteElementSpace &fes,
                               const Array<int> &markers,
                               Vector &b);

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


/// Class for boundary integration L(v) := (g, v)
class BoundaryLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob;
public:
   /** @brief Constructs a boundary integrator with a given Coefficient @a QG.
       Integration order will be @a a * basis_order + @a b. */
   BoundaryLFIntegrator(Coefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) { }

   virtual bool SupportsDevice() { return true; }

   /// Method defining assembly on device
   virtual void AssembleDevice(const FiniteElementSpace &fes,
                               const Array<int> &markers,
                               Vector &b);

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration \f$ L(v) = (g \cdot n, v) \f$
class BoundaryNormalLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   VectorCoefficient &Q;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   BoundaryNormalLFIntegrator(VectorCoefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) { }

   virtual bool SupportsDevice() { return true; }

   /// Method defining assembly on device
   virtual void AssembleDevice(const FiniteElementSpace &fes,
                               const Array<int> &markers,
                               Vector &b);

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration \f$ L(v) = (g \cdot \tau, v) \f$ in 2D
class BoundaryTangentialLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   VectorCoefficient &Q;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   BoundaryTangentialLFIntegrator(VectorCoefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for domain integration of L(v) := (f, v), where
    f=(f1,...,fn) and v=(v1,...,vn). */
class VectorDomainLFIntegrator : public DeltaLFIntegrator
{
private:
   Vector shape, Qvec;
   VectorCoefficient &Q;

public:
   /// Constructs a domain integrator with a given VectorCoefficient
   VectorDomainLFIntegrator(VectorCoefficient &QF)
      : DeltaLFIntegrator(QF), Q(QF) { }

   virtual bool SupportsDevice() { return true; }

   /// Method defining assembly on device
   virtual void AssembleDevice(const FiniteElementSpace &fes,
                               const Array<int> &markers,
                               Vector &b);

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

/** Class for domain integrator L(v) := (f, grad v), where
    f=(f1x,f1y,f1z,...,fnx,fny,fnz) and v=(v1,...,vn). */
class VectorDomainLFGradIntegrator : public DeltaLFIntegrator
{
private:
   Vector shape, Qvec;
   VectorCoefficient &Q;
   DenseMatrix dshape;

public:
   /// Constructs the domain integrator (Q, grad v)
   VectorDomainLFGradIntegrator(VectorCoefficient &QF)
      : DeltaLFIntegrator(QF), Q(QF) { }

   virtual bool SupportsDevice() override { return true; }

   /// Method defining assembly on device
   virtual void AssembleDevice(const FiniteElementSpace &fes,
                               const Array<int> &markers,
                               Vector &b) override;

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override;

   virtual void AssembleDeltaElementVect(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         Vector &elvect) override;

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for boundary integration of L(v) := (g, v), where
    f=(f1,...,fn) and v=(v1,...,vn). */
class VectorBoundaryLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape, vec;
   VectorCoefficient &Q;

public:
   /// Constructs a boundary integrator with a given VectorCoefficient QG
   VectorBoundaryLFIntegrator(VectorCoefficient &QG) : Q(QG) { }

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   // For DG spaces
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// \f$ (f, v)_{\Omega} \f$ for VectorFiniteElements (Nedelec, Raviart-Thomas)
class VectorFEDomainLFIntegrator : public DeltaLFIntegrator
{
private:
   VectorCoefficient &QF;
   DenseMatrix vshape;
   Vector vec;

public:
   VectorFEDomainLFIntegrator(VectorCoefficient &F)
      : DeltaLFIntegrator(F), QF(F) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void AssembleDeltaElementVect(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// \f$ (Q, curl v)_{\Omega} \f$ for Nedelec Elements)
class VectorFEDomainLFCurlIntegrator : public DeltaLFIntegrator
{
private:
   VectorCoefficient *QF=nullptr;
   DenseMatrix curlshape;
   Vector vec;

public:
   /// Constructs the domain integrator (Q, curl v)
   VectorFEDomainLFCurlIntegrator(VectorCoefficient &F)
      : DeltaLFIntegrator(F), QF(&F) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void AssembleDeltaElementVect(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// \f$ (Q, div v)_{\Omega} \f$ for RT Elements)
class VectorFEDomainLFDivIntegrator : public DeltaLFIntegrator
{
private:
   Vector divshape;
   Coefficient &Q;
public:
   /// Constructs the domain integrator (Q, div v)
   VectorFEDomainLFDivIntegrator(Coefficient &QF)
      : DeltaLFIntegrator(QF), Q(QF) { }

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

/** \f$ (f, v \cdot n)_{\partial\Omega} \f$ for vector test function
    v=(v1,...,vn) where all vi are in the same scalar FE space and f is a
    scalar function. */
class VectorBoundaryFluxLFIntegrator : public LinearFormIntegrator
{
private:
   double Sign;
   Coefficient *F;
   Vector shape, nor;

public:
   VectorBoundaryFluxLFIntegrator(Coefficient &f, double s = 1.0,
                                  const IntegrationRule *ir = NULL)
      : LinearFormIntegrator(ir), Sign(s), F(&f) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for boundary integration of (f, v.n) for scalar coefficient f and
    RT vector test function v. This integrator works with RT spaces defined
    using the RT_FECollection class. */
class VectorFEBoundaryFluxLFIntegrator : public LinearFormIntegrator
{
private:
   Coefficient *F;
   Vector shape;
   int oa, ob; // these control the quadrature order, see DomainLFIntegrator

public:
   VectorFEBoundaryFluxLFIntegrator(int a = 1, int b = -1)
      : F(NULL), oa(a), ob(b) { }
   VectorFEBoundaryFluxLFIntegrator(Coefficient &f, int a = 2, int b = 0)
      : F(&f), oa(a), ob(b) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration \f$ L(v) = (n \times f, v) \f$
class VectorFEBoundaryTangentLFIntegrator : public LinearFormIntegrator
{
private:
   VectorCoefficient &f;
   int oa, ob;

public:
   VectorFEBoundaryTangentLFIntegrator(VectorCoefficient &QG,
                                       int a = 2, int b = 0)
      : f(QG), oa(a), ob(b) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** Class for boundary integration of the linear form:
    (alpha/2) < (u.n) f, w > - beta < |u.n| f, w >,
    where f and u are given scalar and vector coefficients, respectively,
    and w is the scalar test function. */
class BoundaryFlowIntegrator : public LinearFormIntegrator
{
private:
   Coefficient *f;
   VectorCoefficient *u;
   double alpha, beta;

   Vector shape;

public:
   BoundaryFlowIntegrator(Coefficient &f_, VectorCoefficient &u_,
                          double a)
   { f = &f_; u = &u_; alpha = a; beta = 0.5*a; }

   BoundaryFlowIntegrator(Coefficient &f_, VectorCoefficient &u_,
                          double a, double b)
   { f = &f_; u = &u_; alpha = a; beta = b; }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** Boundary linear integrator for imposing non-zero Dirichlet boundary
    conditions, to be used in conjunction with DGDiffusionIntegrator.
    Specifically, given the Dirichlet data u_D, the linear form assembles the
    following integrals on the boundary:

    sigma < u_D, (Q grad(v)).n > + kappa < {h^{-1} Q} u_D, v >,

    where Q is a scalar or matrix diffusion coefficient and v is the test
    function. The parameters sigma and kappa should be the same as the ones
    used in the DGDiffusionIntegrator. */
class DGDirichletLFIntegrator : public LinearFormIntegrator
{
protected:
   Coefficient *uD, *Q;
   MatrixCoefficient *MQ;
   double sigma, kappa;

   // these are not thread-safe!
   Vector shape, dshape_dn, nor, nh, ni;
   DenseMatrix dshape, mq, adjJ;

public:
   DGDirichletLFIntegrator(Coefficient &u, const double s, const double k)
      : uD(&u), Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
   DGDirichletLFIntegrator(Coefficient &u, Coefficient &q,
                           const double s, const double k)
      : uD(&u), Q(&q), MQ(NULL), sigma(s), kappa(k) { }
   DGDirichletLFIntegrator(Coefficient &u, MatrixCoefficient &q,
                           const double s, const double k)
      : uD(&u), Q(NULL), MQ(&q), sigma(s), kappa(k) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** Boundary linear form integrator for imposing non-zero Dirichlet boundary
    conditions, in a DG elasticity formulation. Specifically, the linear form is
    given by

    alpha < u_D, (lambda div(v) I + mu (grad(v) + grad(v)^T)) . n > +
      + kappa < h^{-1} (lambda + 2 mu) u_D, v >,

    where u_D is the given Dirichlet data. The parameters alpha, kappa, lambda
    and mu, should match the parameters with the same names used in the bilinear
    form integrator, DGElasticityIntegrator. */
class DGElasticityDirichletLFIntegrator : public LinearFormIntegrator
{
protected:
   VectorCoefficient &uD;
   Coefficient *lambda, *mu;
   double alpha, kappa;

#ifndef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix adjJ;
   DenseMatrix dshape_ps;
   Vector nor;
   Vector dshape_dn;
   Vector dshape_du;
   Vector u_dir;
#endif

public:
   DGElasticityDirichletLFIntegrator(VectorCoefficient &uD_,
                                     Coefficient &lambda_, Coefficient &mu_,
                                     double alpha_, double kappa_)
      : uD(uD_), lambda(&lambda_), mu(&mu_), alpha(alpha_), kappa(kappa_) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** Class for spatial white Gaussian noise integration.

    The target problem is the linear SPDE a(u,v) = F(v) with F(v) := <Ẇ,v>,
    where Ẇ is spatial white Gaussian noise. When the Galerkin method is used to
    discretize this problem into a linear system of equations Ax = b, the RHS is
    a Gaussian random vector b~N(0,M) whose covariance matrix is the same as the
    mass matrix M_ij = (v_i,v_j). This property can be ensured if b = H w, where
    HHᵀ = M and each component w_i~N(0,1).

    There is much flexibility in how we may wish to define H. In this PR, we
    define H = Pᵀ diag(L_e), where P is the local-to-global dof assembly matrix
    and diag(L_e) is a block-diagonal matrix with L_e L_eᵀ = M_e, where M_e is
    the element mass matrix for element e. A straightforward computation shows
    that HHᵀ = Pᵀ diag(M_e) P = M, as necessary. */
class WhiteGaussianNoiseDomainLFIntegrator : public LinearFormIntegrator
{
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
   MassIntegrator massinteg;
   Array<DenseMatrix *> L;

   // Define random generator with Gaussian distribution
   std::default_random_engine generator;
   std::normal_distribution<double> dist;

   bool save_factors = false;
public:

#ifdef MFEM_USE_MPI
   /** @brief Sets the @a seed_ of the random number generator. A fixed seed
       allows for a reproducible sequence of white noise vectors. */
   WhiteGaussianNoiseDomainLFIntegrator(int seed_ = 0)
      : LinearFormIntegrator(), comm(MPI_COMM_NULL)
   {
      if (seed_ > 0) { SetSeed(seed_); }
   }

   /** @brief Sets the MPI communicator @a comm_ and the @a seed_ of the random
       number generator. A fixed seed allows for a reproducible sequence of
       white noise vectors. */
   WhiteGaussianNoiseDomainLFIntegrator(MPI_Comm comm_, int seed_)
      : LinearFormIntegrator(), comm(comm_)
   {
      int myid;
      MPI_Comm_rank(comm, &myid);

      int seed = (seed_ > 0) ? seed_ + myid : time(0) + myid;
      SetSeed(seed);
   }
#else
   /** @brief Sets the @a seed_ of the random number generator. A fixed seed
       allows for a reproducible sequence of white noise vectors. */
   WhiteGaussianNoiseDomainLFIntegrator(int seed_ = 0)
      : LinearFormIntegrator()
   {
      if (seed_ > 0) { SetSeed(seed_); }
   }
#endif
   /// @brief Sets/resets the @a seed of the random number generator.
   void SetSeed(int seed)
   {
      generator.seed(seed);
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   /** @brief Saves the lower triangular matrices in the element-wise Cholesky
       decomposition. The parameter @a NE should be the number of elements in
       the mesh. */
   void SaveFactors(int NE)
   {
      save_factors = true;
      ResetFactors(NE);
   }

   /** @brief Resets the array of saved lower triangular Cholesky decomposition
       matrices. The parameter @a NE should be the number of elements in the
       mesh. */
   void ResetFactors(int NE)
   {
      for (int i = 0; i<L.Size(); i++)
      {
         delete L[i];
      }
      L.DeleteAll();

      L.SetSize(NE);
      for (int i = 0; i<NE; i++)
      {
         L[i] = nullptr;
      }
   }

   ~WhiteGaussianNoiseDomainLFIntegrator()
   {
      for (int i = 0; i<L.Size(); i++)
      {
         delete L[i];
      }
      L.DeleteAll();
   }
};


/** Class for domain integration of L(v) := (f, v), where
    f=(f1,...,fn) and v=(v1,...,vn). that makes use of
    VectorQuadratureFunctionCoefficient*/
class VectorQuadratureLFIntegrator : public LinearFormIntegrator
{
private:
   VectorQuadratureFunctionCoefficient &vqfc;

public:
   VectorQuadratureLFIntegrator(VectorQuadratureFunctionCoefficient &vqfc,
                                const IntegrationRule *ir)
      : LinearFormIntegrator(ir), vqfc(vqfc)
   {
      if (ir)
      {
         MFEM_WARNING("Integration rule not used in this class. "
                      "The QuadratureFunction integration rules are used instead");
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &fe,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void SetIntRule(const IntegrationRule *ir)
   {
      MFEM_WARNING("Integration rule not used in this class. "
                   "The QuadratureFunction integration rules are used instead");
   }
};


/** Class for domain integration L(v) := (f, v) that makes use
    of QuadratureFunctionCoefficient. */
class QuadratureLFIntegrator : public LinearFormIntegrator
{
private:
   QuadratureFunctionCoefficient &qfc;

public:
   QuadratureLFIntegrator(QuadratureFunctionCoefficient &qfc,
                          const IntegrationRule *ir)
      : LinearFormIntegrator(ir), qfc(qfc)
   {
      if (ir)
      {
         MFEM_WARNING("Integration rule not used in this class. "
                      "The QuadratureFunction integration rules are used instead");
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &fe,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void SetIntRule(const IntegrationRule *ir)
   {
      MFEM_WARNING("Integration rule not used in this class. "
                   "The QuadratureFunction integration rules are used instead");
   }
};

}


#endif
