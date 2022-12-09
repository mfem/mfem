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

#ifndef MFEM_GHOST_PENALTY
#define MFEM_GHOST_PENALTY

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

  // Performs full assembly for the normal velocity mass matrix operator.
  class GhostStressPenaltyIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    ParGridFunction *alpha;
    double penaltyParameter;
    Coefficient *mu;
    Coefficient *kappa;
    ShiftedFaceMarker *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
  public:
    GhostStressPenaltyIntegrator(const ParMesh *pmesh, Coefficient &mu_, Coefficient &kappa_, ParGridFunction &alphaF, double penParameter, ShiftedFaceMarker *analyticalSurface, int nTerms) : pmesh(pmesh), mu(&mu_), kappa(&kappa_), alpha(&alphaF), penaltyParameter(penParameter), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms) { }
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };
  // Performs full assembly for the normal velocity mass matrix operator.
  class GhostStressFullGradPenaltyIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;
    double penaltyParameter;
    Coefficient *mu;
    Coefficient *kappa;
    ShiftedFaceMarker *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
  public:
    GhostStressFullGradPenaltyIntegrator(const ParMesh *pmesh, Coefficient &mu_, Coefficient &kappa_, double penParameter, ShiftedFaceMarker *analyticalSurface, int nTerms) : pmesh(pmesh), mu(&mu_), kappa(&kappa_), penaltyParameter(penParameter), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms) { }
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

}

#endif // NITSCHE_SOLVER
