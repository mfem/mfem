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

#ifndef MFEM_SHIFTED_WEIGHTED_SOLVER
#define MFEM_SHIFTED_WEIGHTED_SOLVER

#include "mfem.hpp"
#include "AnalyticalGeometricShape.hpp"
#include "marking.hpp"

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

  class WeightedShiftedStressBoundaryForceIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    ParGridFunction *alpha;
    Coefficient *mu;
    Coefficient *kappa;
    VectorCoefficient *vD;
    VectorCoefficient *vN;
    ShiftedFaceMarker *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
    bool include_cut;
    
  public:
    WeightedShiftedStressBoundaryForceIntegrator(const ParMesh *pmesh, ParGridFunction &alphaF, Coefficient &mu_, Coefficient &kappa_, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec, ShiftedFaceMarker *analyticalSurface, int nTerms, bool includeCut = 0)  : pmesh(pmesh), alpha(&alphaF), mu(&mu_), kappa(&kappa_), vD(dist_vec), vN(normal_vec), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms), include_cut(includeCut) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

  class WeightedShiftedStressBoundaryForceTransposeIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    ParGridFunction *alpha;
    Coefficient *mu;
    Coefficient *kappa;
    ShiftedFaceMarker *analyticalSurface;
    int par_shared_face_count;
    bool include_cut;
  public:
    WeightedShiftedStressBoundaryForceTransposeIntegrator(const ParMesh *pmesh, ParGridFunction &alphaF, Coefficient &mu_, Coefficient &kappa_, ShiftedFaceMarker *analyticalSurface, bool includeCut = 0) : pmesh(pmesh), alpha(&alphaF), mu(&mu_), kappa(&kappa_), analyticalSurface(analyticalSurface), par_shared_face_count(0), include_cut(includeCut) {}
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };
  class WeightedShiftedStressNitscheBCForceIntegrator : public LinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    ParGridFunction *alpha;
    ShiftedVectorFunctionCoefficient *uD;
    VectorCoefficient *vD;
    VectorCoefficient *vN;
    ShiftedFaceMarker *analyticalSurface;
    int par_shared_face_count;
    bool include_cut;
  public:
    WeightedShiftedStressNitscheBCForceIntegrator(const ParMesh *pmesh, ParGridFunction &alphaF, ShiftedVectorFunctionCoefficient &uD_, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec, ShiftedFaceMarker *analyticalSurface, bool includeCut = 0) : pmesh(pmesh), alpha(&alphaF), uD(&uD_), vD(dist_vec), vN(normal_vec), analyticalSurface(analyticalSurface), par_shared_face_count(0), include_cut(includeCut) {}
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
