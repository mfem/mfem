// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "change_basis.hpp"
#include "qinterp/dispatch.hpp"

namespace mfem
{

ChangeOfBasis::ChangeOfBasis(FiniteElementSpace &fes_, int dest_btype)
   : Operator(fes_.GetVSize()),
     fes(fes_)
{
   const FiniteElement *fe = fes.GetFE(0);
   auto *tbe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tbe != nullptr, "Must be a tensor element.");
   const int source_btype = tbe->GetBasisType();

   const int order = fes.GetElementOrder(0);
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(Geometry::SEGMENT, 2*order - 1);

   auto compute_vandermonde = [&](int btype, DenseMatrix &V)
   {
      if (btype < BasisType::NumBasisTypes)
      {
         Poly_1D::Basis &basis = poly1d.GetBasis(order, btype);
         for (int i = 0; i < ir.Size(); ++i)
         {
            Vector col;
            V.GetColumnReference(i, col);
            basis.Eval(ir[i].x, col);
         }
         V.Transpose();
      }
      else if (btype == LEGENDRE)
      {
         for (int i = 0; i < ir.Size(); ++i)
         {
            Vector col;
            V.GetColumnReference(i, col);
            Poly_1D::CalcLegendre(order, ir[i].x, col.HostWrite());
         }
         V.Transpose();
      }
      else if (btype == INTEGRATED_LEGENDRE)
      {
         MFEM_ABORT("Not yet implemented.");
      }
      else
      {
         MFEM_ABORT("");
      }
   };

   DenseMatrix V1(order + 1, order + 1);
   compute_vandermonde(source_btype, V1);

   DenseMatrix V2(order + 1, order + 1);
   compute_vandermonde(dest_btype, V2);

   DenseMatrixInverse V2_inv(V2);

   T1D.SetSize(order + 1, order + 1);
   V2_inv.Mult(V1, T1D);

   {
      T1D_inv.SetSize(order + 1, order + 1);
      DenseMatrix A = T1D;
      DenseMatrixInverse A_inv(A);
      A_inv.GetInverseMatrix(T1D_inv);
   }

   dof2quad.FE = fe;
   dof2quad.mode = DofToQuad::TENSOR;
   dof2quad.ndof = order + 1;
   dof2quad.nqpt = order + 1;
   dof2quad.B.SetSize((order + 1)*(order + 1));
}

void ChangeOfBasis::Mult_(const DenseMatrix &B1D, const Vector &x,
                          Vector &y) const
{
   using namespace internal::quadrature_interpolator;

   const auto ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *restr_op = fes.GetElementRestriction(ordering);
   auto restr = dynamic_cast<const ElementRestrictionOperator*>(restr_op);
   MFEM_VERIFY(restr != nullptr, "Unsupported element restriction type.");

   x_e.SetSize(restr->Height());
   y_e.SetSize(restr->Height());

   restr->Mult(x, x_e);

   dof2quad.B.GetMemory().CopyFrom(B1D.GetMemory(), dof2quad.B.Size());

   if (fes.GetOrdering() == Ordering::byNODES)
   {
      TensorValues<QVectorLayout::byNODES>(fes.GetNE(), 1, dof2quad, x_e, y_e);
   }
   else
   {
      TensorValues<QVectorLayout::byVDIM>(fes.GetNE(), 1, dof2quad, x_e, y_e);
   }

   restr->MultLeftInverse(y_e, y);
}

void ChangeOfBasis::Mult(const Vector &x, Vector &y) const
{
   Mult_(T1D, x, y);
}

void ChangeOfBasis::MultInverse(const Vector &x, Vector &y) const
{
   Mult_(T1D_inv, x, y);
}

} // namespace mfem
