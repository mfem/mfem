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
#ifndef MFEM_BACKENDS_KERNELS_CONFORM_PROLONGATION_OP
#define MFEM_BACKENDS_KERNELS_CONFORM_PROLONGATION_OP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

#ifdef MFEM_USE_MPI

// ***************************************************************************
// * kConformingProlongationOperator
//  **************************************************************************
class kConformingProlongationOperator : public kernels::Operator
{
protected:
   mfem::Array<int> external_ldofs;
   kernels::array<int> d_external_ldofs;
   kCommD *gc;
   int kMaxTh;
public:
   kConformingProlongationOperator(Layout&, Layout&, mfem::ParFiniteElementSpace&);
   ~kConformingProlongationOperator();
   void d_Mult(const kernels::Vector &x, kernels::Vector &y) const;
   void d_MultTranspose(const kernels::Vector &x, kernels::Vector &y) const;

   virtual void Mult_(const kernels::Vector &x, kernels::Vector &y) const;
   virtual void MultTranspose_(const kernels::Vector &x, kernels::Vector &y) const;

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
   virtual void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const;
};

#endif

} // namespace mfem::kernels

} // mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_CONFORM_PROLONGATION_OP
