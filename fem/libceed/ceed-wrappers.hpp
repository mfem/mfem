// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CEED_WRAPPERS_HPP
#define MFEM_CEED_WRAPPERS_HPP

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED
#include <ceed.h>
#include "ceedsolvers-interpolation.h"
#include "../../linalg/operator.hpp"

namespace mfem
{

class MFEMCeedOperator : public Operator
{
public:
   MFEMCeedOperator(CeedOperator oper, const Array<int> &ess_tdofs_, const Operator *P_);
   MFEMCeedOperator(CeedOperator oper, const Operator *P_);
   ~MFEMCeedOperator();
   void Mult(const Vector& x, Vector& y) const;
   CeedOperator GetCeedOperator() const;
   const Array<int> &GetEssentialTrueDofs() const;
   const Operator *GetProlongation() const;
private:
   Array<int> ess_tdofs;
   const Operator *P;
   class UnconstrainedMFEMCeedOperator *unconstrained_op;
   ConstrainedOperator *constrained_op;
};

/**
   Wrap CeedInterpolation object in an mfem::Operator
*/
class MFEMCeedInterpolation : public mfem::Operator
{
public:
   MFEMCeedInterpolation(
      Ceed ceed, CeedBasis basisctof,
      CeedElemRestriction erestrictu_coarse,
      CeedElemRestriction erestrictu_fine);

   ~MFEMCeedInterpolation();

   virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;

   virtual void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const;

   using Operator::SetupRAP;
private:
   int Initialize(Ceed ceed, CeedBasis basisctof,
                  CeedElemRestriction erestrictu_coarse,
                  CeedElemRestriction erestrictu_fine);

   CeedBasis basisctof_;
   CeedVector u_, v_;

   CeedInterpolation ceed_interp_;

   bool owns_basis_;
};

}

#endif

#endif
