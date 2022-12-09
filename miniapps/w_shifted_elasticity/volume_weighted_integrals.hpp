// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_VOLUME_WEIGHTED_INTEGRALS
#define MFEM_VOLUME_WEIGHTED_INTEGRALS

#include "mfem.hpp"

using namespace std;
using namespace mfem;

/// BilinearFormIntegrator for the high-order extension of shifted boundary
/// method.
/// A(u, w) = -<2*mu*epsilon(u) n, w>
///           -<(p*I) n, w>
///           -<u, sigma(w,q) n> // transpose of the above two terms
///           +<alpha h^{-1} u , w >
namespace mfem
{
  
  class WeightedStressForceIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    Coefficient *mu;
    Coefficient *kappa;
    
public:
    WeightedStressForceIntegrator(ParGridFunction &alphaF, Coefficient &mu_, Coefficient &kappa_) : alpha(&alphaF), mu(&mu_), kappa(&kappa_) {}
    virtual void AssembleElementMatrix(const FiniteElement &el,
				       ElementTransformation &Trans,
				       DenseMatrix &elmat);
    
    const IntegrationRule &GetRule(const FiniteElement &trial_fe,
				   const FiniteElement &test_fe,
				   ElementTransformation &Trans);
    
  };
  
  class WeightedVectorForceIntegrator : public LinearFormIntegrator
{
private:
  ParGridFunction *alpha;
  VectorCoefficient *Q;
  
public:
   /// Constructs the domain integrator (Q * f, div v)
  WeightedVectorForceIntegrator(ParGridFunction &alphaF, VectorCoefficient &QF) : alpha(&alphaF), Q(&QF) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};


}

#endif // NITSCHE_SOLVER
