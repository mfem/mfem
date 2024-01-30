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

#ifndef MFEM_PARAMETRIZED_TMOP_INTEGRATOR
#define MFEM_PARAMETRIZED_TMOP_INTEGRATOR

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include "AnalyticalSurface.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{
class ParametrizedTMOP_Integrator : public TMOP_Integrator
{
protected:
   AnalyticalSurface *analyticalSurface;
public:
   ParametrizedTMOP_Integrator(TMOP_QualityMetric *m, TargetConstructor *tc,
                               TMOP_QualityMetric *hm, AnalyticalSurface *analyticalSurface) : TMOP_Integrator(m, tc, hm), analyticalSurface(analyticalSurface)
{
}
// For interior nodes
// \f$ (dmu/dT)_{jp} Dsh_{Bs} Winv_{sp} x_{Bj} \f$
// x_{Bj} is the solution vector
// Upper case letter for boundary nodes
// \f$ (dmu/dT)_{op} Dsh_{As} Winv_{sp} (dx_{Ao}/dt(Bj}) t_{Bj} \f$
// where t_{Bj} is the parametrized solution vector. 
virtual void AssembleElementVectorExact(const FiniteElement &el,
                                        ElementTransformation &T,
                                        const Vector &elfun, Vector &elvect);
// For interior nodes
// \f$ d((dmu/dT)_{jp} Dsh_{Bs} Winv_{sp})/dx_{Dr} \f$
// For boundary nodes
// \f$ d((dmu/dT)_{jp} Dsh_{Bs} Winv_{sp})/dt_{Eg} dx_{Eg}/dt{Dr} dx_{Ao}/dt{Bj} \f$
//                                        +
// \f$ (dmu/dT)_{op} Dsh_{As} Winv_{sp} d^{2}x_{Ao}/dt{Bj}dt_{Dr} \f$
virtual void AssembleElementGradExact(const FiniteElement &el,
                                      ElementTransformation &T,
                                      const Vector &elfun, DenseMatrix &elmat);
};
}

#endif
