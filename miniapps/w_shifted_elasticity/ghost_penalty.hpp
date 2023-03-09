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
#include "marking.hpp"

using namespace std;
using namespace mfem;

/// Ghost Penalty
/// A(u,v) = \sum_{1}^{nTerms} (penPar/i!) h^{2*i-1) < [[ sigma(u),(i-1) ]], [[sigma(v),(i-1)]] >
/// where [[sigma_{0}]] = sigma_{ij} n_tilda_{j} 
///       [[sigma_{1}]] = sigma_{ij,k} n_tilda_{k}
///       ....

namespace mfem
{
  void AddOneToBinaryArray(Array<int> & binary, int size, int dim);
  // Performs full assembly for the normal velocity mass matrix operator.
  class GhostStressFullGradPenaltyIntegrator : public BilinearFormIntegrator
  {
  private:
    ParMesh *pmesh;
    double penaltyParameter;
    Coefficient *mu;
    Coefficient *kappa;
    ShiftedFaceMarker *analyticalSurface;
    int par_shared_face_count;
    int nTerms;
    double dupPenaltyParameter;
  public:
    GhostStressFullGradPenaltyIntegrator(ParMesh *pmesh, Coefficient &mu_, Coefficient &kappa_, double penParameter, ShiftedFaceMarker *analyticalSurface, int nTerms) : pmesh(pmesh), mu(&mu_), kappa(&kappa_), penaltyParameter(penParameter), analyticalSurface(analyticalSurface), par_shared_face_count(0), nTerms(nTerms), dupPenaltyParameter(penParameter) { }
    virtual void AssembleFaceMatrix(const FiniteElement &fe,
				    const FiniteElement &fe2,
				    FaceElementTransformations &Tr,
				    DenseMatrix &elmat);
  };

}

#endif // NITSCHE_SOLVER
