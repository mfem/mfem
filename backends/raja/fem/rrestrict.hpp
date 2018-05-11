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
#ifndef LAGHOS_RAJA_RESTRICT_OP
#define LAGHOS_RAJA_RESTRICT_OP

namespace mfem
{

// ***************************************************************************
// * RajaRestrictionOperator
// ***************************************************************************
class RajaRestrictionOperator : public RajaOperator
{
protected:
   int entries;
   const RajaArray<int> *indices;
public:
   RajaRestrictionOperator(const int h, const int w,
                           const RajaArray<int> *idx):
      RajaOperator(h,w),
      entries(idx->size()>>1),
      indices(idx) {}
   void Mult(const RajaVector& x, RajaVector& y) const ;
};

} // mfem

#endif // LAGHOS_RAJA_RESTRICT_OP
