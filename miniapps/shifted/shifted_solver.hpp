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

#ifndef MFEM_SHIFTED_SOLVER
#define MFEM_SHIFTED_SOLVER

#include "mfem.hpp"
#include "AnalyticalSurface.hpp"

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

  double factorial(int nTerms);
  
class ShiftedVectorFunctionCoefficient : public VectorCoefficient
{
protected:
   std::function<void(const Vector &, Vector &)> Function;

public:
   ShiftedVectorFunctionCoefficient(int dim,
                                    std::function<void(const Vector &, Vector &)> F)
      : VectorCoefficient(dim), Function(std::move(F)) { }

   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector D(vdim);
      D = 0.;
      return (this)->Eval(V, T, ip, D);
   }

   /// Evaluate the coefficient at @a ip + @a D.
   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip,
             const Vector &D);
};

  class ShiftedStrainBoundaryForceIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    Coefficient *mu;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
    
  public:
    ShiftedStrainBoundaryForceIntegrator(const ParMesh *pmesh, Coefficient &mu_, AnalyticalSurface *analyticalSurface, int nTerms)  : pmesh(pmesh), mu(&mu_), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  class ShiftedStrainBoundaryForceTransposeIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    Coefficient *mu;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    
  public:
    ShiftedStrainBoundaryForceTransposeIntegrator(const ParMesh *pmesh, Coefficient &mu_, AnalyticalSurface *analyticalSurface) : pmesh(pmesh), mu(&mu_), analyticalSurface(analyticalSurface), par_shared_face_count(0) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  class ShiftedPressureBoundaryForceIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
  
  public:
    ShiftedPressureBoundaryForceIntegrator(const ParMesh *pmesh, AnalyticalSurface *analyticalSurface, int nTerms) : pmesh(pmesh), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms) { }
    virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
				    const FiniteElement &trial_fe2,
				    const FiniteElement &test_fe1,
				    const FiniteElement &test_fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };
  
  class ShiftedPressureBoundaryForceTransposeIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    
  public:
    ShiftedPressureBoundaryForceTransposeIntegrator(const ParMesh *pmesh, AnalyticalSurface *analyticalSurface) : pmesh(pmesh), analyticalSurface(analyticalSurface), par_shared_face_count(0) { }
    virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
				    const FiniteElement &trial_fe2,
				    const FiniteElement &test_fe1,
				    const FiniteElement &test_fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  // Performs full assembly for the normal velocity mass matrix operator.
  class ShiftedVelocityPenaltyIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    double alpha;
    Coefficient *mu;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
    
  public:
    ShiftedVelocityPenaltyIntegrator(const ParMesh *pmesh, double penParameter, Coefficient &mu_, AnalyticalSurface *analyticalSurface, int nTerms) : pmesh(pmesh), alpha(penParameter), mu(&mu_), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms) { }
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };
  class ShiftedStrainNitscheBCForceIntegrator : public LinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    Coefficient *mu;
    ShiftedVectorFunctionCoefficient *uD;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    
  public:
    ShiftedStrainNitscheBCForceIntegrator(const ParMesh *pmesh, Coefficient &mu_, ShiftedVectorFunctionCoefficient &uD_, AnalyticalSurface *analyticalSurface) : pmesh(pmesh), mu(&mu_), uD(&uD_), analyticalSurface(analyticalSurface), par_shared_face_count(0) {}
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					const FiniteElement &el2,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect) {}
  
  };

  class ShiftedPressureNitscheBCForceIntegrator : public LinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    ShiftedVectorFunctionCoefficient *uD;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    
  public:
    ShiftedPressureNitscheBCForceIntegrator(const ParMesh *pmesh, ShiftedVectorFunctionCoefficient &uD_, AnalyticalSurface *analyticalSurface) : pmesh(pmesh), uD(&uD_), analyticalSurface(analyticalSurface), par_shared_face_count(0) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					const FiniteElement &el2,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect) {}
  
  };
  
  // Performs full assembly for the normal velocity mass matrix operator.
  class ShiftedVelocityBCPenaltyIntegrator : public LinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    double alpha;
    Coefficient *mu;
    ShiftedVectorFunctionCoefficient *uD;
    AnalyticalSurface *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
  public:
    ShiftedVelocityBCPenaltyIntegrator(const ParMesh *pmesh, double penParameter, Coefficient &mu_, ShiftedVectorFunctionCoefficient &uD_, AnalyticalSurface *analyticalSurface, int nTerms) : pmesh(pmesh), alpha(penParameter), mu(&mu_), uD(&uD_), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms) { }
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					const FiniteElement &el2,
					FaceElementTransformations &Tr,
					Vector &elvect);
    virtual void AssembleRHSElementVect(const FiniteElement &el,
					ElementTransformation &Tr,
					Vector &elvect) {}
  };

}

#endif // NITSCHE_SOLVER
