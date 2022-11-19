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

  class WeightedStrainBoundaryForceIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    Coefficient *mu;

  public:
    WeightedStrainBoundaryForceIntegrator(ParGridFunction &alphaF, Coefficient &mu_)  : alpha(&alphaF), mu(&mu_) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  class WeightedStrainBoundaryForceTransposeIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    Coefficient *mu;
    
  public:
    WeightedStrainBoundaryForceTransposeIntegrator(ParGridFunction &alphaF, Coefficient &mu_)  : alpha(&alphaF), mu(&mu_) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  class WeightedPressureBoundaryForceIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;

  public:
    WeightedPressureBoundaryForceIntegrator(ParGridFunction &alphaF)  : alpha(&alphaF) { }
    virtual void AssembleFaceMatrix(const FiniteElement &trial_fe,
				    const FiniteElement &test_fe1,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };
  
  class WeightedPressureBoundaryForceTransposeIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    
  public:
    WeightedPressureBoundaryForceTransposeIntegrator(ParGridFunction &alphaF) : alpha(&alphaF) { }
    virtual void AssembleFaceMatrix(const FiniteElement &trial_fe,
				    const FiniteElement &test_fe1,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  // Performs full assembly for the normal velocity mass matrix operator.
  class WeightedVelocityPenaltyIntegrator : public BilinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    double penaltyParameter;
    Coefficient *mu;
  public:
    WeightedVelocityPenaltyIntegrator(ParGridFunction &alphaF, double penParameter, Coefficient &mu_) : alpha(&alphaF), penaltyParameter(penParameter), mu(&mu_) { }
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };
  class WeightedStrainNitscheBCForceIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    Coefficient *mu;
    VectorCoefficient *uD;
  public:
    WeightedStrainNitscheBCForceIntegrator(ParGridFunction &alphaF, Coefficient &mu_, VectorCoefficient &uD_)  : alpha(&alphaF), mu(&mu_), uD(&uD_) {}
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };

  class WeightedPressureNitscheBCForceIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    VectorCoefficient *uD;
  public:
    WeightedPressureNitscheBCForceIntegrator(ParGridFunction &alphaF, VectorCoefficient &uD_) : alpha(&alphaF), uD(&uD_) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };
  
  // Performs full assembly for the normal velocity mass matrix operator.
  class WeightedVelocityBCPenaltyIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    double penaltyParameter;
    Coefficient *mu;
    VectorCoefficient *uD;
  public:
    WeightedVelocityBCPenaltyIntegrator(ParGridFunction &alphaF, double penParameter, Coefficient &mu_, VectorCoefficient &uD_) : alpha(&alphaF), penaltyParameter(penParameter), mu(&mu_), uD(&uD_) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };
  
  // Performs full assembly for the normal velocity mass matrix operator.
  class WeightedTractionBCIntegrator : public LinearFormIntegrator
  {
  private:
    ParGridFunction *alpha;
    VectorCoefficient *tN;
  public:
    WeightedTractionBCIntegrator(ParGridFunction &alphaF, VectorCoefficient &tN_) : alpha(&alphaF), tN(&tN_) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect);
  };

}

#endif // NITSCHE_SOLVER
