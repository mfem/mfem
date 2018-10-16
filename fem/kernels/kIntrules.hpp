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
#ifndef MFEM_FEM_KERNELS_INTRULES
#define MFEM_FEM_KERNELS_INTRULES

MFEM_NAMESPACE

// *****************************************************************************
class IntegrationPoint;

void kIntRulesInit(const size_t, IntegrationPoint*);
void kIntRulesPointIni(IntegrationPoint*);
void kIntRulesLinear1DIni(IntegrationPoint*);
void kIntRulesLinear2DIni(IntegrationPoint*);
void kIntRulesLinear3DIni(IntegrationPoint*);
void kIntRulesBiLinear2DIni(IntegrationPoint*);
void kIntRulesTriLinear3DIni(IntegrationPoint*);

void kIPSetX(const IntegrationPoint*,const double, const size_t =0);
double kIPGetX(const IntegrationPoint*, const size_t =0);
double kIPGetY(const IntegrationPoint*, const size_t =0);
void kIPSetXY(const IntegrationPoint*,const double*,const int,const double*, const int,const size_t =0);
void kIPSetW(const IntegrationPoint*,const double, const size_t =0);

void kIPPts(const IntegrationPoint*,const size_t, double*);

void kCalcChebyshev(const int, const double, double*);
   
MFEM_NAMESPACE_END

#endif // MFEM_FEM_KERNELS_INTRULES
