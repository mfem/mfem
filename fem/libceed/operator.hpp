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

#ifndef MFEM_LIBCEED_OPERATOR
#define MFEM_LIBCEED_OPERATOR

#include "../../config/config.hpp"

#ifdef MFEM_USE_CEED
#include "ceed.hpp"
#include <ceed.h>
#include "../../linalg/operator.hpp"
#include "util.h"

namespace mfem
{

/** A base class to represent a CeedOperator as an MFEM Operator. */
class MFEMCeedOperator : public Operator
{
protected:
   CeedOperator oper;
   CeedVector u, v;

   /// The base class owns and destroys u, v but expects its
   /// derived classes to manage oper
   MFEMCeedOperator() : oper(nullptr), u(nullptr), v(nullptr) { }

public:
   void Mult(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y) const;
   void GetDiagonal(Vector &diag) const;
   virtual ~MFEMCeedOperator()
   {
      CeedVectorDestroy(&u);
      CeedVectorDestroy(&v);
   }
   CeedOperator GetCeedOperator() const { return oper; }
};

class UnconstrainedMFEMCeedOperator : public MFEMCeedOperator
{
public:
   UnconstrainedMFEMCeedOperator(CeedOperator ceed_op)
   {
      oper = ceed_op;
      CeedElemRestriction er;
      CeedOperatorGetActiveElemRestriction(oper, &er);
      int s;
      CeedElemRestrictionGetLVectorSize(er, &s);
      height = width = s;
      CeedVectorCreate(internal::ceed, height, &v);
      CeedVectorCreate(internal::ceed, width, &u);
   }

   Operator * SetupRAP(const Operator *Pi, const Operator *Po)
   {
      return Operator::SetupRAP(Pi, Po);
   }
};

/** Wraps a CeedOperator in an mfem::Operator, with essential boundary
    conditions. */
class ConstrainedMFEMCeedOperator : public Operator
{
public:
   ConstrainedMFEMCeedOperator(CeedOperator oper, const Array<int> &ess_tdofs_,
                               const Operator *P_);
   ConstrainedMFEMCeedOperator(CeedOperator oper, const Operator *P_);
   ~ConstrainedMFEMCeedOperator();
   void Mult(const Vector& x, Vector& y) const;
   CeedOperator GetCeedOperator() const;
   const Array<int> &GetEssentialTrueDofs() const;
   const Operator *GetProlongation() const;
private:
   Array<int> ess_tdofs;
   const Operator *P;
   UnconstrainedMFEMCeedOperator *unconstrained_op;
   ConstrainedOperator *constrained_op;
};

/** @brief Multigrid interpolation operator in Ceed framework

    Interpolation/restriction has two components, an element-wise
    interpolation and then a scaling to correct multiplicity
    on shared ldofs. This encapsulates those two in one object
    using the MFEM Operator interface. */
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
   int Finalize();

   CeedBasis basisctof_;
   CeedVector u_, v_;

   bool owns_basis_;

   CeedQFunction qf_restrict, qf_prolong;
   CeedOperator op_interp, op_restrict;
   CeedVector fine_multiplicity_r;
   CeedVector fine_work;
};

/** The different evaluation modes available for PA and MF CeedIntegrator. */
enum class EvalMode { None, Interp, Grad, InterpAndGrad };

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_LIBCEED_OPERATOR
