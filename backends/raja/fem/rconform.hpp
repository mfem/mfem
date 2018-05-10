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
#ifndef LAGHOS_RAJA_CONFORM_PROLONGATION_OP
#define LAGHOS_RAJA_CONFORM_PROLONGATION_OP

namespace mfem {
  
  // ***************************************************************************
  // * RajaConformingProlongationOperator
  //  **************************************************************************
  class RajaConformingProlongationOperator : public RajaOperator{
  protected:
    Array<int> external_ldofs;
    //RajaArray<int> d_non_empty_m;
    RajaArray<int> d_external_ldofs;
    RajaCommD *gc;
    int kMaxTh;
  public:
    RajaConformingProlongationOperator(ParFiniteElementSpace &);
    ~RajaConformingProlongationOperator();
    void d_Mult(const RajaVector &x, RajaVector &y) const;
    void d_MultTranspose(const RajaVector &x, RajaVector &y) const;  
    void h_Mult(const Vector &x, Vector &y) const;
    void h_MultTranspose(const Vector &x, Vector &y) const;
  };
  
} // mfem

#endif // LAGHOS_RAJA_CONFORM_PROLONGATION_OP
