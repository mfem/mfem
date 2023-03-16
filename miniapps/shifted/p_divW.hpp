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

#ifndef MFEM_PDIVU_SOLVER
#define MFEM_PDIVU_SOLVER

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

  class PDivWForceIntegrator : public BilinearFormIntegrator
{
public:
  PDivWForceIntegrator() {}
  virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
				      const FiniteElement &test_fe,
				      ElementTransformation &Trans,
				      DenseMatrix &elmat);
  const IntegrationRule &GetRule(const FiniteElement &trial_fe,
				 const FiniteElement &test_fe,
				 ElementTransformation &Trans);
  
  };
}

#endif // NITSCHE_SOLVER
