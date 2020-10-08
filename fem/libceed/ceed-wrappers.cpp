#include "ceed-wrappers.hpp"

#ifdef MFEM_USE_CEED
#include <ceed-backend.h>
#include "ceed.hpp"
#include "ceedsolvers-utility.h"

namespace mfem
{

class UnconstrainedMFEMCeedOperator : public Operator
{
public:
   UnconstrainedMFEMCeedOperator(CeedOperator oper);
   ~UnconstrainedMFEMCeedOperator();

   virtual void Mult(const Vector& x, Vector& y) const;
   using Operator::SetupRAP;
private:
   CeedOperator oper_;
   CeedVector u_, v_;
};

UnconstrainedMFEMCeedOperator::UnconstrainedMFEMCeedOperator(CeedOperator oper)
 : oper_(oper)
{
   int ierr = 0;
   Ceed ceed;
   ierr += CeedOperatorGetCeed(oper, &ceed);
   CeedElemRestriction er;
   ierr += CeedOperatorGetActiveElemRestriction(oper, &er);
   int s;
   ierr += CeedElemRestrictionGetLVectorSize(er, &s);
   height = width = s;
   ierr += CeedVectorCreate(ceed, height, &v_);
   ierr += CeedVectorCreate(ceed, width, &u_);
   MFEM_ASSERT(ierr == 0, "CEED error");
}

UnconstrainedMFEMCeedOperator::~UnconstrainedMFEMCeedOperator()
{
   int ierr = 0;
   ierr += CeedVectorDestroy(&v_);
   ierr += CeedVectorDestroy(&u_);
   MFEM_ASSERT(ierr == 0, "CEED error");
}

void UnconstrainedMFEMCeedOperator::Mult(const Vector& x, Vector& y) const
{
   int ierr = 0;

   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if (Device::Allows(Backend::CUDA) && mem == CEED_MEM_DEVICE)
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

   ierr += CeedVectorSetArray(u_, mem, CEED_USE_POINTER, const_cast<CeedScalar*>(x_ptr));
   ierr += CeedVectorSetArray(v_, mem, CEED_USE_POINTER, y_ptr);

   ierr += CeedOperatorApply(oper_, u_, v_, CEED_REQUEST_IMMEDIATE);

   ierr += CeedVectorTakeArray(u_, mem, const_cast<CeedScalar**>(&x_ptr));
   ierr += CeedVectorTakeArray(v_, mem, &y_ptr);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

MFEMCeedOperator::MFEMCeedOperator(
   CeedOperator oper,
   const Array<int> &ess_tdofs,
   const Operator *P
)
{
   unconstrained_op_ = new UnconstrainedMFEMCeedOperator(oper);
   Operator *rap = unconstrained_op_->SetupRAP(P, P);
   height = width = rap->Height();
   bool own_rap = rap != unconstrained_op_;
   constrained_op_ = new ConstrainedOperator(rap, ess_tdofs, own_rap);
}

MFEMCeedOperator::MFEMCeedOperator(CeedOperator oper, const Operator *P)
 : MFEMCeedOperator(oper, Array<int>(), P)
{ }

MFEMCeedOperator::~MFEMCeedOperator()
{
   delete constrained_op_;
   delete unconstrained_op_;
}

void MFEMCeedOperator::Mult(const Vector& x, Vector& y) const
{
   constrained_op_->Mult(x, y);
}

int MFEMCeedInterpolation::Initialize(
  Ceed ceed, CeedBasis basisctof,
  CeedElemRestriction erestrictu_coarse, CeedElemRestriction erestrictu_fine)
{
   int ierr = 0;

   ierr = CeedInterpolationCreate(ceed, basisctof, erestrictu_coarse,
                                  erestrictu_fine, &ceed_interp_); CeedChk(ierr);

   ierr = CeedVectorCreate(ceed, height, &v_); CeedChk(ierr);
   ierr = CeedVectorCreate(ceed, width, &u_); CeedChk(ierr);

   return 0;
}

MFEMCeedInterpolation::MFEMCeedInterpolation(
   Ceed ceed, CeedBasis basisctof,
   CeedElemRestriction erestrictu_coarse,
   CeedElemRestriction erestrictu_fine)
{
   int lo_nldofs, ho_nldofs;
   CeedElemRestrictionGetLVectorSize(erestrictu_coarse, &lo_nldofs);
   CeedElemRestrictionGetLVectorSize(erestrictu_fine, &ho_nldofs);
   height = ho_nldofs;
   width = lo_nldofs;
   owns_basis_ = false;
   Initialize(ceed, basisctof, erestrictu_coarse, erestrictu_fine);
}

MFEMCeedInterpolation::~MFEMCeedInterpolation()
{
   CeedVectorDestroy(&v_);
   CeedVectorDestroy(&u_);
   if (owns_basis_)
   {
      CeedBasisDestroy(&basisctof_);
   }
   CeedInterpolationDestroy(&ceed_interp_);
}

void MFEMCeedInterpolation::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   int ierr = 0;

   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::CUDA) && mem==CEED_MEM_DEVICE )
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

   ierr += CeedVectorSetArray(u_, mem, CEED_USE_POINTER, const_cast<CeedScalar*>(x_ptr));
   ierr += CeedVectorSetArray(v_, mem, CEED_USE_POINTER, y_ptr);

   ierr += CeedInterpolationInterpolate(ceed_interp_, u_, v_);

   ierr += CeedVectorTakeArray(u_, mem, const_cast<CeedScalar**>(&x_ptr));
   ierr += CeedVectorTakeArray(v_, mem, &y_ptr);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

void MFEMCeedInterpolation::MultTranspose(const mfem::Vector& x,
                                          mfem::Vector& y) const
{
   int ierr = 0;

   const CeedScalar *x_ptr;
   CeedScalar *y_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if ( Device::Allows(Backend::CUDA) && mem==CEED_MEM_DEVICE )
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

   ierr += CeedVectorSetArray(v_, mem, CEED_USE_POINTER, const_cast<CeedScalar*>(x_ptr));
   ierr += CeedVectorSetArray(u_, mem, CEED_USE_POINTER, y_ptr);

   ierr += CeedInterpolationRestrict(ceed_interp_, v_, u_);

   ierr += CeedVectorTakeArray(v_, mem, const_cast<CeedScalar**>(&x_ptr));
   ierr += CeedVectorTakeArray(u_, mem, &y_ptr);

   MFEM_ASSERT(ierr == 0, "CEED error");
}

}

#endif
