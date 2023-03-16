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

#ifndef MFEM_WEIGHTED_NITSCHE_SOLVER
#define MFEM_WEIGHTED_NITSCHE_SOLVER

#include "mfem.hpp"
#include "marking.hpp"

using namespace std;
using namespace mfem;

/// A(u, v) = <alpha * sigma(v) n, u>
///           -<alpha * sigma(u) n, v>
///           +(3.0 * Kappa * penPar / h) <alpha * u.n, v.n>
///           +(2.0 * Mu * penPar / h) <alpha * u (I - n x n), v>

///  l(v)   = <alpha * sigma(v) n, uD>
///           +(3.0 * Kappa * penPar / h) <alpha * v.n, uD.n>  
///           +(2.0 * Mu * penPar / h) <alpha * v (I - n x n), uD>
///           +<alpha *v ,tN>
///  alpha = 1.0;       
///  for 0 < alpha < 1 we apply the shifted operators (see shifted_weighted_solver.cpp)

namespace mfem
{
  // A(u,v) = <sigma(v) n, u>
  class WeightedStressBoundaryForceIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    Coefficient *mu;
    Coefficient *kappa;
    
  public:
    WeightedStressBoundaryForceIntegrator(ParGridFunction &alphaF, Coefficient &mu_, Coefficient &kappa_)  : alpha(&alphaF), mu(&mu_), kappa(&kappa_) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };
 // A(u,v) = -<sigma(u) n, v>
  class WeightedStressBoundaryForceTransposeIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    Coefficient *mu;
    Coefficient *kappa;

  public:
    WeightedStressBoundaryForceTransposeIntegrator(ParGridFunction &alphaF, Coefficient &mu_, Coefficient &kappa_)  : alpha(&alphaF), mu(&mu_), kappa(&kappa_) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  // A(u,v) = (3.0 * Kappa * penPar / h) <u.n, v.n>
  class WeightedNormalDisplacementPenaltyIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    double penaltyParameter;
    Coefficient *kappa;

  public:
    WeightedNormalDisplacementPenaltyIntegrator(ParGridFunction &alphaF, double penParameter, Coefficient &kappa_) : alpha(&alphaF), penaltyParameter(penParameter), kappa(&kappa_) { }
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  // A(u,v) = (2.0 * Mu * penPar / h) <u (I - n x n), v> 
  class WeightedTangentialDisplacementPenaltyIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    double penaltyParameter;
    Coefficient *mu;
    
  public:
    WeightedTangentialDisplacementPenaltyIntegrator(ParGridFunction &alphaF, double penParameter, Coefficient &mu_) : alpha(&alphaF), penaltyParameter(penParameter), mu(&mu_) { }
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  // l(v) = <sigma(v) n, uD> 
  class WeightedStressNitscheBCForceIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    Coefficient *mu;
    Coefficient *kappa;
    VectorCoefficient *uD;
    
  public:
    WeightedStressNitscheBCForceIntegrator(ParGridFunction &alphaF, Coefficient &mu_, Coefficient &kappa_, VectorCoefficient &uD_)  : alpha(&alphaF), mu(&mu_), kappa(&kappa_), uD(&uD_) {}
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };

  // l(v) = (3.0 * Kappa * penPar / h) <v.n, uD.n>  
  class WeightedNormalDisplacementBCPenaltyIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    double penaltyParameter;
    Coefficient *kappa;
    VectorCoefficient *uD;
    
  public:
    WeightedNormalDisplacementBCPenaltyIntegrator(ParGridFunction &alphaF, double penParameter, Coefficient &kappa_, VectorCoefficient &uD_) : alpha(&alphaF), penaltyParameter(penParameter), kappa(&kappa_), uD(&uD_) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };

  // l(v) = (2.0 * Mu * penPar / h) <v (I - n x n), uD> 
  class WeightedTangentialDisplacementBCPenaltyIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    double penaltyParameter;
    Coefficient *mu;
    VectorCoefficient *uD;
    
  public:
    WeightedTangentialDisplacementBCPenaltyIntegrator(ParGridFunction &alphaF, double penParameter, Coefficient &mu_, VectorCoefficient &uD_) : alpha(&alphaF), penaltyParameter(penParameter), mu(&mu_), uD(&uD_) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };

  // l(v) =  <v ,tN>  
  class WeightedTractionBCIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    MatrixFunctionCoefficient *tN;
    
  public:
    WeightedTractionBCIntegrator(ParGridFunction &alphaF, MatrixFunctionCoefficient &tN_) : alpha(&alphaF), tN(&tN_) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };

}

#endif // NITSCHE_SOLVER
