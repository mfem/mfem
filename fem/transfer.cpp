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

#include "transfer.hpp"
#include "../general/forall.hpp"

namespace mfem
{

TransferOperator::TransferOperator(const FiniteElementSpace& lFESpace_,
                                   const FiniteElementSpace& hFESpace_)
   : Operator(hFESpace_.GetVSize(), lFESpace_.GetVSize())
{
   if (lFESpace_.FEColl() == hFESpace_.FEColl())
   {
      OperatorPtr P(Operator::ANY_TYPE);
      hFESpace_.GetTransferOperator(lFESpace_, P);
      P.SetOperatorOwner(false);
      opr = P.Ptr();
   }
   else if (lFESpace_.GetMesh()->GetNE() > 0
            && hFESpace_.GetMesh()->GetNE() > 0
            && dynamic_cast<const TensorBasisElement*>(lFESpace_.GetFE(0))
            && dynamic_cast<const TensorBasisElement*>(hFESpace_.GetFE(0)))
   {
      opr = new TensorProductPRefinementTransferOperator(lFESpace_, hFESpace_);
   }
   else
   {
      opr = new PRefinementTransferOperator(lFESpace_, hFESpace_);
   }
}

TransferOperator::~TransferOperator() { delete opr; }

void TransferOperator::Mult(const Vector& x, Vector& y) const
{
   opr->Mult(x, y);
}

void TransferOperator::MultTranspose(const Vector& x, Vector& y) const
{
   opr->MultTranspose(x, y);
}

PRefinementTransferOperator::PRefinementTransferOperator(
   const FiniteElementSpace& lFESpace_, const FiniteElementSpace& hFESpace_)
   : Operator(hFESpace_.GetVSize(), lFESpace_.GetVSize()), lFESpace(lFESpace_),
     hFESpace(hFESpace_)
{
}

PRefinementTransferOperator::~PRefinementTransferOperator() {}

void PRefinementTransferOperator::Mult(const Vector& x, Vector& y) const
{
   Mesh* mesh = hFESpace.GetMesh();
   Array<int> l_dofs, h_dofs, l_vdofs, h_vdofs;
   DenseMatrix loc_prol;
   Vector subY, subX;

   Geometry::Type cached_geom = Geometry::INVALID;
   const FiniteElement* h_fe = NULL;
   const FiniteElement* l_fe = NULL;
   IsoparametricTransformation T;

   int vdim = lFESpace.GetVDim();

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      hFESpace.GetElementDofs(i, h_dofs);
      lFESpace.GetElementDofs(i, l_dofs);

      const Geometry::Type geom = mesh->GetElementBaseGeometry(i);
      if (geom != cached_geom)
      {
         h_fe = hFESpace.GetFE(i);
         l_fe = lFESpace.GetFE(i);
         T.SetIdentityTransformation(h_fe->GetGeomType());
         h_fe->GetTransferMatrix(*l_fe, T, loc_prol);
         subY.SetSize(loc_prol.Height());
         cached_geom = geom;
      }

      for (int vd = 0; vd < vdim; vd++)
      {
         l_dofs.Copy(l_vdofs);
         lFESpace.DofsToVDofs(vd, l_vdofs);
         h_dofs.Copy(h_vdofs);
         hFESpace.DofsToVDofs(vd, h_vdofs);
         x.GetSubVector(l_vdofs, subX);
         loc_prol.Mult(subX, subY);
         y.SetSubVector(h_vdofs, subY);
      }
   }
}

void PRefinementTransferOperator::MultTranspose(const Vector& x,
                                                Vector& y) const
{
   y = 0.0;

   Mesh* mesh = hFESpace.GetMesh();
   Array<int> l_dofs, h_dofs, l_vdofs, h_vdofs;
   DenseMatrix loc_prol;
   Vector subY, subX;

   Array<char> processed(hFESpace.GetVSize());
   processed = 0;

   Geometry::Type cached_geom = Geometry::INVALID;
   const FiniteElement* h_fe = NULL;
   const FiniteElement* l_fe = NULL;
   IsoparametricTransformation T;

   int vdim = lFESpace.GetVDim();

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      hFESpace.GetElementDofs(i, h_dofs);
      lFESpace.GetElementDofs(i, l_dofs);

      const Geometry::Type geom = mesh->GetElementBaseGeometry(i);
      if (geom != cached_geom)
      {
         h_fe = hFESpace.GetFE(i);
         l_fe = lFESpace.GetFE(i);
         T.SetIdentityTransformation(h_fe->GetGeomType());
         h_fe->GetTransferMatrix(*l_fe, T, loc_prol);
         loc_prol.Transpose();
         subY.SetSize(loc_prol.Height());
         cached_geom = geom;
      }

      for (int vd = 0; vd < vdim; vd++)
      {
         l_dofs.Copy(l_vdofs);
         lFESpace.DofsToVDofs(vd, l_vdofs);
         h_dofs.Copy(h_vdofs);
         hFESpace.DofsToVDofs(vd, h_vdofs);

         x.GetSubVector(h_vdofs, subX);
         for (int p = 0; p < h_dofs.Size(); ++p)
         {
            if (processed[lFESpace.DecodeDof(h_dofs[p])])
            {
               subX[p] = 0.0;
            }
         }

         loc_prol.Mult(subX, subY);
         y.AddElementVector(l_vdofs, subY);
      }

      for (int p = 0; p < h_dofs.Size(); ++p)
      {
         processed[lFESpace.DecodeDof(h_dofs[p])] = 1;
      }
   }
}

TensorProductPRefinementTransferOperator::
TensorProductPRefinementTransferOperator(
   const FiniteElementSpace& lFESpace_,
   const FiniteElementSpace& hFESpace_)
   : Operator(hFESpace_.GetVSize(), lFESpace_.GetVSize()), lFESpace(lFESpace_),
     hFESpace(hFESpace_)
{
   // Assuming the same element type
   Mesh* mesh = lFESpace.GetMesh();
   dim = mesh->Dimension();
   if (mesh->GetNE() == 0)
   {
      return;
   }
   const FiniteElement& el = *lFESpace.GetFE(0);

   const TensorBasisElement* ltel =
      dynamic_cast<const TensorBasisElement*>(&el);
   MFEM_VERIFY(ltel, "Low order FE space must be tensor product space");

   const TensorBasisElement* htel =
      dynamic_cast<const TensorBasisElement*>(hFESpace.GetFE(0));
   MFEM_VERIFY(htel, "High order FE space must be tensor product space");
   const Array<int>& hdofmap = htel->GetDofMap();

   const IntegrationRule& ir = hFESpace.GetFE(0)->GetNodes();
   IntegrationRule irLex = ir;

   // The quadrature points, or equivalently, the dofs of the high order space
   // must be sorted in lexicographical order
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      irLex.IntPoint(i) = ir.IntPoint(hdofmap[i]);
   }

   NE = lFESpace.GetNE();
   const DofToQuad& maps = el.GetDofToQuad(irLex, DofToQuad::TENSOR);

   D1D = maps.ndof;
   Q1D = maps.nqpt;
   B = maps.B;
   Bt = maps.Bt;

   elem_restrict_lex_l =
      lFESpace.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   MFEM_VERIFY(elem_restrict_lex_l,
               "Low order ElementRestriction not available");

   elem_restrict_lex_h =
      hFESpace.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   MFEM_VERIFY(elem_restrict_lex_h,
               "High order ElementRestriction not available");

   localL.SetSize(elem_restrict_lex_l->Height(), Device::GetMemoryType());
   localH.SetSize(elem_restrict_lex_h->Height(), Device::GetMemoryType());
   localL.UseDevice(true);
   localH.UseDevice(true);

   MFEM_VERIFY(dynamic_cast<const ElementRestriction*>(elem_restrict_lex_h),
               "High order element restriction is of unsupported type");

   mask.SetSize(localH.Size(), Device::GetMemoryType());
   static_cast<const ElementRestriction*>(elem_restrict_lex_h)
   ->BooleanMask(mask);
   mask.UseDevice(true);
}

namespace TransferKernels
{
void Prolongation2D(const int NE, const int D1D, const int Q1D,
                    const Vector& localL, Vector& localH,
                    const Array<double>& B, const Vector& mask)
{
   auto x_ = Reshape(localL.Read(), D1D, D1D, NE);
   auto y_ = Reshape(localH.ReadWrite(), Q1D, Q1D, NE);
   auto B_ = Reshape(B.Read(), Q1D, D1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, NE);

   localH = 0.0;

   MFEM_FORALL(e, NE,
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x_(dx, dy, e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B_(qx, dx) * s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double d2q = B_(qy, dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               y_(qx, qy, e) += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            y_(qx, qy, e) *= m_(qx, qy, e);
         }
      }
   });
}

void Prolongation3D(const int NE, const int D1D, const int Q1D,
                    const Vector& localL, Vector& localH,
                    const Array<double>& B, const Vector& mask)
{
   auto x_ = Reshape(localL.Read(), D1D, D1D, D1D, NE);
   auto y_ = Reshape(localH.ReadWrite(), Q1D, Q1D, Q1D, NE);
   auto B_ = Reshape(B.Read(), Q1D, D1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, Q1D, NE);

   localH = 0.0;

   MFEM_FORALL(e, NE,
   {
      for (int dz = 0; dz < D1D; ++dz)
      {
         double sol_xy[MAX_Q1D][MAX_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double sol_x[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] = 0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x_(dx, dy, dz, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B_(qx, dx) * s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = B_(qy, dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz = B_(qz, dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  y_(qx, qy, qz, e) += wz * sol_xy[qy][qx];
               }
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               y_(qx, qy, qz, e) *= m_(qx, qy, qz, e);
            }
         }
      }
   });
}

void Restriction2D(const int NE, const int D1D, const int Q1D,
                   const Vector& localH, Vector& localL,
                   const Array<double>& Bt, const Vector& mask)
{
   auto x_ = Reshape(localH.Read(), Q1D, Q1D, NE);
   auto y_ = Reshape(localL.ReadWrite(), D1D, D1D, NE);
   auto Bt_ = Reshape(Bt.Read(), D1D, Q1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, NE);

   localL = 0.0;

   MFEM_FORALL(e, NE,
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[MAX_D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double s = m_(qx, qy, e) * x_(qx, qy, e);
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt_(dx, qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double q2d = Bt_(dy, qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y_(dx, dy, e) += q2d * sol_x[dx];
            }
         }
      }
   });
}
void Restriction3D(const int NE, const int D1D, const int Q1D,
                   const Vector& localH, Vector& localL,
                   const Array<double>& Bt, const Vector& mask)
{
   auto x_ = Reshape(localH.Read(), Q1D, Q1D, Q1D, NE);
   auto y_ = Reshape(localL.ReadWrite(), D1D, D1D, D1D, NE);
   auto Bt_ = Reshape(Bt.Read(), D1D, Q1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, Q1D, NE);

   localL = 0.0;

   MFEM_FORALL(e, NE,
   {
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double sol_xy[MAX_D1D][MAX_D1D];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double sol_x[MAX_D1D];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double s = m_(qx, qy, qz, e) * x_(qx, qy, qz, e);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += Bt_(dx, qx) * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy = Bt_(dy, qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz = Bt_(dz, qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y_(dx, dy, dz, e) += wz * sol_xy[dy][dx];
               }
            }
         }
      }
   });
}
} // namespace TransferKernels

TensorProductPRefinementTransferOperator::
~TensorProductPRefinementTransferOperator()
{
}

void TensorProductPRefinementTransferOperator::Mult(const Vector& x,
                                                    Vector& y) const
{
   if (lFESpace.GetMesh()->GetNE() == 0)
   {
      return;
   }

   elem_restrict_lex_l->Mult(x, localL);
   if (dim == 2)
   {
      TransferKernels::Prolongation2D(NE, D1D, Q1D, localL, localH, B, mask);
   }
   else if (dim == 3)
   {
      TransferKernels::Prolongation3D(NE, D1D, Q1D, localL, localH, B, mask);
   }
   else
   {
      MFEM_ABORT("TensorProductPRefinementTransferOperator::Mult not "
                 "implemented for dim = "
                 << dim);
   }
   elem_restrict_lex_h->MultTranspose(localH, y);
}

void TensorProductPRefinementTransferOperator::MultTranspose(const Vector& x,
                                                             Vector& y) const
{
   if (lFESpace.GetMesh()->GetNE() == 0)
   {
      return;
   }

   elem_restrict_lex_h->Mult(x, localH);
   if (dim == 2)
   {
      TransferKernels::Restriction2D(NE, D1D, Q1D, localH, localL, Bt, mask);
   }
   else if (dim == 3)
   {
      TransferKernels::Restriction3D(NE, D1D, Q1D, localH, localL, Bt, mask);
   }
   else
   {
      MFEM_ABORT("TensorProductPRefinementTransferOperator::MultTranspose not "
                 "implemented for dim = "
                 << dim);
   }
   elem_restrict_lex_l->MultTranspose(localL, y);
}

#ifdef MFEM_USE_MPI
TrueTransferOperator::TrueTransferOperator(const
                                           ParFiniteElementSpace& lFESpace_,
                                           const ParFiniteElementSpace& hFESpace_)
   : lFESpace(lFESpace_), hFESpace(hFESpace_)
{
   localTransferOperator = new TransferOperator(lFESpace_, hFESpace_);

   tmpL.SetSize(lFESpace_.GetVSize());
   tmpH.SetSize(hFESpace_.GetVSize());

   hFESpace.GetRestrictionMatrix()->BuildTranspose();
}

TrueTransferOperator::~TrueTransferOperator()
{
   delete localTransferOperator;
}

void TrueTransferOperator::Mult(const Vector& x, Vector& y) const
{
   lFESpace.GetProlongationMatrix()->Mult(x, tmpL);
   localTransferOperator->Mult(tmpL, tmpH);
   hFESpace.GetRestrictionMatrix()->Mult(tmpH, y);
}

void TrueTransferOperator::MultTranspose(const Vector& x, Vector& y) const
{
   hFESpace.GetRestrictionMatrix()->MultTranspose(x, tmpH);
   localTransferOperator->MultTranspose(tmpH, tmpL);
   lFESpace.GetProlongationMatrix()->MultTranspose(tmpL, y);
}
#endif

} // namespace mfem
