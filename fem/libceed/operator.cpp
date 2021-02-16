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

#include "operator.hpp"

#include "../../config/config.hpp"
#include "../../linalg/vector.hpp"
#include "../fespace.hpp"
#include "../../general/forall.hpp"

#ifdef MFEM_USE_CEED

namespace mfem
{

void MFEMCeedOperator::Mult(const Vector &x, Vector &y) const
{
#ifdef MFEM_USE_CEED
   int ierr;
   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      x_ptr = x.Read();
      y_ptr = y.Write();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = y.HostWrite();
      mem = CEED_MEM_HOST;
   }
   ierr = CeedVectorSetArray(u, mem, CEED_USE_POINTER,
                             const_cast<CeedScalar*>(x_ptr)); PCeedChk(ierr);
   ierr = CeedVectorSetArray(v, mem, CEED_USE_POINTER, y_ptr); PCeedChk(ierr);

   ierr = CeedOperatorApply(oper, u, v, CEED_REQUEST_IMMEDIATE);
   PCeedChk(ierr);

   ierr = CeedVectorTakeArray(u, mem, const_cast<CeedScalar**>(&x_ptr));
   PCeedChk(ierr);
   ierr = CeedVectorTakeArray(v, mem, &y_ptr); PCeedChk(ierr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void MFEMCeedOperator::AddMult(const Vector &x, Vector &y) const
{
#ifdef MFEM_USE_CEED
   int ierr;
   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      x_ptr = x.Read();
      y_ptr = y.ReadWrite();
   }
   else
   {
      x_ptr = x.HostRead();
      y_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   ierr = CeedVectorSetArray(u, mem, CEED_USE_POINTER,
                             const_cast<CeedScalar*>(x_ptr)); PCeedChk(ierr);
   ierr = CeedVectorSetArray(v, mem, CEED_USE_POINTER, y_ptr); PCeedChk(ierr);

   ierr = CeedOperatorApplyAdd(oper, u, v, CEED_REQUEST_IMMEDIATE);
   PCeedChk(ierr);

   ierr = CeedVectorTakeArray(u, mem, const_cast<CeedScalar**>(&x_ptr));
   PCeedChk(ierr);
   ierr = CeedVectorTakeArray(v, mem, &y_ptr); PCeedChk(ierr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void MFEMCeedOperator::GetDiagonal(Vector &diag) const
{
#ifdef MFEM_USE_CEED
   int ierr;
   CeedScalar *d_ptr;
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      d_ptr = diag.ReadWrite();
   }
   else
   {
      d_ptr = diag.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   ierr = CeedVectorSetArray(v, mem, CEED_USE_POINTER, d_ptr); PCeedChk(ierr);

   ierr = CeedOperatorLinearAssembleAddDiagonal(oper, v, CEED_REQUEST_IMMEDIATE);
   PCeedChk(ierr);

   ierr = CeedVectorTakeArray(v, mem, &d_ptr); PCeedChk(ierr);
#else
   MFEM_ABORT("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

ConstrainedMFEMCeedOperator::ConstrainedMFEMCeedOperator(
   CeedOperator oper,
   const Array<int> &ess_tdofs_,
   const Operator *P_)
   : ess_tdofs(ess_tdofs_), P(P_)
{
   unconstrained_op = new UnconstrainedMFEMCeedOperator(oper);
   Operator *rap = unconstrained_op->SetupRAP(P, P);
   height = width = rap->Height();
   bool own_rap = (rap != unconstrained_op);
   constrained_op = new ConstrainedOperator(rap, ess_tdofs, own_rap);
}

ConstrainedMFEMCeedOperator::ConstrainedMFEMCeedOperator(CeedOperator oper,
                                                         const Operator *P_)
   : ConstrainedMFEMCeedOperator(oper, Array<int>(), P_)
{ }

ConstrainedMFEMCeedOperator::~ConstrainedMFEMCeedOperator()
{
   delete constrained_op;
   delete unconstrained_op;
}

void ConstrainedMFEMCeedOperator::Mult(const Vector& x, Vector& y) const
{
   constrained_op->Mult(x, y);
}

CeedOperator ConstrainedMFEMCeedOperator::GetCeedOperator() const
{
   return unconstrained_op->GetCeedOperator();
}

const Array<int> &ConstrainedMFEMCeedOperator::GetEssentialTrueDofs() const
{
   return ess_tdofs;
}

const Operator *ConstrainedMFEMCeedOperator::GetProlongation() const
{
   return P;
}

int MFEMCeedInterpolation::Initialize(
   Ceed ceed, CeedBasis basisctof,
   CeedElemRestriction erestrictu_coarse, CeedElemRestriction erestrictu_fine)
{
   int ierr = 0;

   int height, width;
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_coarse, &width);
   CeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_fine, &height);
   CeedChk(ierr);

   // interpolation qfunction
   const int bp3_ncompu = 1;
   CeedQFunction l_qf_restrict, l_qf_prolong;
   ierr = CeedQFunctionCreateIdentity(ceed, bp3_ncompu, CEED_EVAL_NONE,
                                      CEED_EVAL_INTERP, &l_qf_restrict); CeedChk(ierr);
   ierr = CeedQFunctionCreateIdentity(ceed, bp3_ncompu, CEED_EVAL_INTERP,
                                      CEED_EVAL_NONE, &l_qf_prolong); CeedChk(ierr);

   qf_restrict = l_qf_restrict;
   qf_prolong = l_qf_prolong;

   CeedVector c_fine_multiplicity;
   ierr = CeedVectorCreate(ceed, height, &c_fine_multiplicity); CeedChk(ierr);
   ierr = CeedVectorSetValue(c_fine_multiplicity, 0.0); CeedChk(ierr);

   // Create the restriction operator
   // Restriction - Fine to coarse
   ierr = CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE,
                             CEED_QFUNCTION_NONE, &op_restrict); CeedChk(ierr);
   ierr = CeedOperatorSetField(op_restrict, "input", erestrictu_fine,
                               CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE); CeedChk(ierr);
   ierr = CeedOperatorSetField(op_restrict, "output", erestrictu_coarse,
                               basisctof, CEED_VECTOR_ACTIVE); CeedChk(ierr);

   // Interpolation - Coarse to fine
   // Create the prolongation operator
   ierr =  CeedOperatorCreate(ceed, qf_prolong, CEED_QFUNCTION_NONE,
                              CEED_QFUNCTION_NONE, &op_interp); CeedChk(ierr);
   ierr =  CeedOperatorSetField(op_interp, "input", erestrictu_coarse,
                                basisctof, CEED_VECTOR_ACTIVE); CeedChk(ierr);
   ierr = CeedOperatorSetField(op_interp, "output", erestrictu_fine,
                               CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE); CeedChk(ierr);

   ierr = CeedElemRestrictionGetMultiplicity(erestrictu_fine,
                                             c_fine_multiplicity); CeedChk(ierr);
   ierr = CeedVectorCreate(ceed, height, &fine_multiplicity_r); CeedChk(ierr);

   CeedScalar* fine_r_data;
   const CeedScalar* fine_data;
   ierr = CeedVectorGetArray(fine_multiplicity_r, CEED_MEM_HOST,
                             &fine_r_data); CeedChk(ierr);
   ierr = CeedVectorGetArrayRead(c_fine_multiplicity, CEED_MEM_HOST,
                                 &fine_data); CeedChk(ierr);
   for (int i = 0; i < height; ++i)
   {
      fine_r_data[i] = 1.0 / fine_data[i];
   }

   ierr = CeedVectorRestoreArray(fine_multiplicity_r, &fine_r_data); CeedChk(ierr);
   ierr = CeedVectorRestoreArrayRead(c_fine_multiplicity, &fine_data);
   CeedChk(ierr);
   ierr = CeedVectorDestroy(&c_fine_multiplicity); CeedChk(ierr);

   ierr = CeedVectorCreate(ceed, height, &fine_work); CeedChk(ierr);

   ierr = CeedVectorCreate(ceed, height, &v_); CeedChk(ierr);
   ierr = CeedVectorCreate(ceed, width, &u_); CeedChk(ierr);

   return 0;
}

int MFEMCeedInterpolation::Finalize()
{
   int ierr;

   ierr = CeedQFunctionDestroy(&qf_restrict); CeedChk(ierr);
   ierr = CeedQFunctionDestroy(&qf_prolong); CeedChk(ierr);
   ierr = CeedOperatorDestroy(&op_interp); CeedChk(ierr);
   ierr = CeedOperatorDestroy(&op_restrict); CeedChk(ierr);
   ierr = CeedVectorDestroy(&fine_multiplicity_r); CeedChk(ierr);
   ierr = CeedVectorDestroy(&fine_work); CeedChk(ierr);

   return 0;
}

MFEMCeedInterpolation::MFEMCeedInterpolation(
   Ceed ceed, CeedBasis basisctof,
   CeedElemRestriction erestrictu_coarse,
   CeedElemRestriction erestrictu_fine)
{
   int ierr;
   int lo_nldofs, ho_nldofs;
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_coarse, &lo_nldofs);
   PCeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(erestrictu_fine,
                                            &ho_nldofs); PCeedChk(ierr);
   height = ho_nldofs;
   width = lo_nldofs;
   owns_basis_ = false;
   Initialize(ceed, basisctof, erestrictu_coarse, erestrictu_fine);
}

MFEMCeedInterpolation::~MFEMCeedInterpolation()
{
   int ierr;
   ierr = CeedVectorDestroy(&v_); PCeedChk(ierr);
   ierr = CeedVectorDestroy(&u_); PCeedChk(ierr);
   if (owns_basis_)
   {
      ierr = CeedBasisDestroy(&basisctof_); PCeedChk(ierr);
   }
   Finalize();
}

void MFEMCeedInterpolation::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   int ierr = 0;
   const CeedScalar *in_ptr;
   CeedScalar *out_ptr;
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      in_ptr = x.Read();
      out_ptr = y.ReadWrite();
   }
   else
   {
      in_ptr = x.HostRead();
      out_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   ierr = CeedVectorSetArray(u_, mem, CEED_USE_POINTER,
                             const_cast<CeedScalar*>(in_ptr)); PCeedChk(ierr);
   ierr = CeedVectorSetArray(v_, mem, CEED_USE_POINTER,
                             out_ptr); PCeedChk(ierr);

   ierr = CeedOperatorApply(op_interp, u_, v_,
                            CEED_REQUEST_IMMEDIATE); PCeedChk(ierr);
   ierr = CeedVectorPointwiseMult(v_, fine_multiplicity_r); PCeedChk(ierr);

   ierr = CeedVectorTakeArray(u_, mem, const_cast<CeedScalar**>(&in_ptr));
   PCeedChk(ierr);
   ierr = CeedVectorTakeArray(v_, mem, &out_ptr); PCeedChk(ierr);
}

void MFEMCeedInterpolation::MultTranspose(const mfem::Vector& x,
                                          mfem::Vector& y) const
{
   int ierr = 0;
   CeedMemType mem;
   ierr = CeedGetPreferredMemType(internal::ceed, &mem); PCeedChk(ierr);
   const CeedScalar *in_ptr;
   CeedScalar *out_ptr;
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      in_ptr = x.Read();
      out_ptr = y.ReadWrite();
   }
   else
   {
      in_ptr = x.HostRead();
      out_ptr = y.HostReadWrite();
      mem = CEED_MEM_HOST;
   }
   ierr = CeedVectorSetArray(v_, mem, CEED_USE_POINTER,
                             const_cast<CeedScalar*>(in_ptr)); PCeedChk(ierr);
   ierr = CeedVectorSetArray(u_, mem, CEED_USE_POINTER,
                             out_ptr); PCeedChk(ierr);

   int length;
   ierr = CeedVectorGetLength(v_, &length); PCeedChk(ierr);

   const CeedScalar *multiplicitydata;
   CeedScalar *workdata;
   ierr = CeedVectorGetArrayRead(fine_multiplicity_r, mem,
                                 &multiplicitydata); PCeedChk(ierr);
   ierr = CeedVectorGetArray(fine_work, mem, &workdata); PCeedChk(ierr);
   MFEM_FORALL(i, length,
   {workdata[i] = in_ptr[i] * multiplicitydata[i];});
   ierr = CeedVectorRestoreArrayRead(fine_multiplicity_r,
                                      &multiplicitydata);
   ierr = CeedVectorRestoreArray(fine_work, &workdata); PCeedChk(ierr);

   ierr = CeedOperatorApply(op_restrict, fine_work, u_,
                            CEED_REQUEST_IMMEDIATE); PCeedChk(ierr);

   ierr = CeedVectorTakeArray(v_, mem, const_cast<CeedScalar**>(&in_ptr));
   PCeedChk(ierr);
   ierr = CeedVectorTakeArray(u_, mem, &out_ptr); PCeedChk(ierr);
}

} // namespace mfem

#endif
