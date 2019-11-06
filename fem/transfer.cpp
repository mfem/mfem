// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

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
   else if (hFESpace_.GetMesh()->GetNE() > 0 &&
            dynamic_cast<const TensorBasisElement*>(hFESpace_.GetFE(0)))
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
            if (processed[DecodeDof(h_dofs[p])])
            {
               subX[p] = 0.0;
            }
         }

         loc_prol.Mult(subX, subY);
         y.AddElementVector(l_vdofs, subY);
      }

      for (int p = 0; p < h_dofs.Size(); ++p)
      {
         processed[DecodeDof(h_dofs[p])] = 1;
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
   ldofmap = ltel->GetDofMap();

   const TensorBasisElement* htel =
       dynamic_cast<const TensorBasisElement*>(hFESpace.GetFE(0));
   MFEM_VERIFY(htel, "High order FE space must be tensor product space");
   hdofmap = htel->GetDofMap();

   const IntegrationRule& ir = hFESpace.GetFE(0)->GetNodes();
   irLex = ir;

   // The quadrature points, or equivalently, the dofs of the high order space
   // must be sorted in lexicographical order
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      irLex.IntPoint(i) = ir.IntPoint(hdofmap[i]);
   }

   NE = lFESpace.GetNE();
   maps = &el.GetDofToQuad(irLex, DofToQuad::TENSOR);

   D1D = maps->ndof;
   Q1D = maps->nqpt;
   B = maps->B;
   Bt = maps->Bt;

   elem_restrict_lex_l = lFESpace.GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC);
   elem_restrict_lex_h = hFESpace.GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC);

   localL.SetSize(elem_restrict_lex_l->Height(), Device::GetMemoryType());
   localH.SetSize(elem_restrict_lex_h->Height(), Device::GetMemoryType());
   localL.UseDevice(true);
   localH.UseDevice(true);

   Vector mask_host(localH.Size());
   ((ElementRestriction*)elem_restrict_lex_h)->BooleanMask(mask_host);

   mask.SetSize(localH.Size(), Device::GetMemoryType());
   mask = mask_host;
   mask.UseDevice(true);
}

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

   if (dim == 2)
   {
      Mult2D(x, y);
   }
   else if (dim == 3)
   {
      Mult3D(x, y);
   }
   else
   {
      MFEM_ABORT("TensorProductPRefinementTransferOperator::Mult not "
                 "implemented for dim = "
                 << dim);
   }
}

void TensorProductPRefinementTransferOperator::Mult2D(const Vector& x,
                                                      Vector& y) const
{   
   elem_restrict_lex_l->Mult(x, localL);

   auto x_ = Reshape(localL.Read(), D1D, D1D, NE);
   auto y_ = Reshape(localH.ReadWrite(), Q1D, Q1D, NE);
   auto B_ = Reshape(B.Read(), Q1D, D1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, NE);

   localH = 0.0;

   MFEM_FORALL(e, NE,
   {
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[Q1D];
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
            y_(qx, qy, e) = m_(qx, qy, e) * y_(qx, qy, e);
         }
      }
   });

   elem_restrict_lex_h->MultTranspose(localH, y);
}

void TensorProductPRefinementTransferOperator::Mult3D(const Vector& x,
                                                      Vector& y) const
{
   Array<int> l_dofs, h_dofs, l_vdofs, h_vdofs;
   Vector subY, subY_lex, subX, subX_lex;
   subX.SetSize(D1D * D1D * D1D);
   subX_lex.SetSize(subX.Size());
   subY.SetSize(Q1D * Q1D * Q1D);
   subY_lex.SetSize(subY.Size());

   double* sol_x = new double[Q1D];
   double* sol_xy_ = new double[Q1D * Q1D];
   auto sol_xy = Reshape(sol_xy_, Q1D, Q1D);

   for (int e = 0; e < NE; ++e)
   {
      // Extract dofs in lexicographical order
      lFESpace.GetElementDofs(e, l_dofs);
      hFESpace.GetElementDofs(e, h_dofs);

      const int vdim = lFESpace.GetVDim();
      for (int vd = 0; vd < vdim; ++vd)
      {
         l_dofs.Copy(l_vdofs);
         lFESpace.DofsToVDofs(vd, l_vdofs);
         x.GetSubVector(l_vdofs, subX);

         for (int i = 0; i < subX.Size(); ++i)
         {
            subX_lex[i] = subX[ldofmap[i]];
         }

         // Apply doftoQuad map using sum factorization
         auto subX_lex_ = Reshape(subX_lex.Read(), D1D, D1D, D1D);
         auto subY_lex_ = Reshape(subY_lex.Write(), Q1D, Q1D, Q1D);
         auto B_ = Reshape(B.Read(), Q1D, D1D);

         subY_lex = 0.0;

         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy(qx, qy) = 0.0;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] = 0;
               }
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double s = subX_lex_(dx, dy, dz);
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
                     sol_xy(qx, qy) += wy * sol_x[qx];
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
                     subY_lex_(qx, qy, qz) += wz * sol_xy(qx, qy);
                  }
               }
            }
         }

         for (int i = 0; i < subY.Size(); ++i)
         {
            subY[hdofmap[i]] = subY_lex[i];
         }

         // Set subvectors
         h_dofs.Copy(h_vdofs);
         hFESpace.DofsToVDofs(vd, h_vdofs);
         y.SetSubVector(h_vdofs, subY);
      }
   }

   delete[] sol_xy_;
   delete[] sol_x;
}

void TensorProductPRefinementTransferOperator::MultTranspose(const Vector& x,
                                                             Vector& y) const
{
   if (lFESpace.GetMesh()->GetNE() == 0)
   {
      return;
   }

   y = 0.0;

   if (dim == 2)
   {
      MultTranspose2D(x, y);
   }
   else if (dim == 3)
   {
      MultTranspose3D(x, y);
   }
   else
   {
      MFEM_ABORT("TensorProductPRefinementTransferOperator::Mult not "
                 "implemented for dim = "
                 << dim);
   }
}

void TensorProductPRefinementTransferOperator::MultTranspose2D(const Vector& x,
                                                               Vector& y) const
{
   elem_restrict_lex_h->Mult(x, localH);

   auto x_ = Reshape(localH.Read(), Q1D, Q1D, NE);
   auto y_ = Reshape(localL.ReadWrite(), D1D, D1D, NE);
   auto Bt_ = Reshape(Bt.Read(), D1D, Q1D);
   auto m_ = Reshape(mask.Read(), Q1D, Q1D, NE);

   localL = 0.0;

   MFEM_FORALL(e, NE,
   {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double s = (m_(qx, qy, e) == 0.0) ? 0.0 : x_(qx, qy, e);
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

   elem_restrict_lex_l->MultTranspose(localL, y);
}

void TensorProductPRefinementTransferOperator::MultTranspose3D(const Vector& x,
                                                               Vector& y) const
{
   Array<int> l_dofs, h_dofs, l_vdofs, h_vdofs;
   Vector subY, subY_lex, subX, subX_lex;
   subX.SetSize(Q1D * Q1D * Q1D);
   subX_lex.SetSize(subX.Size());
   subY.SetSize(D1D * D1D * D1D);
   subY_lex.SetSize(subY.Size());

   Array<char> processed(hFESpace.GetVSize());
   processed = 0;

   double* sol_x = new double[D1D];
   double* sol_xy_ = new double[D1D * D1D];
   auto sol_xy = Reshape(sol_xy_, D1D, D1D);

   for (int e = 0; e < NE; ++e)
   {
      hFESpace.GetElementDofs(e, h_dofs);
      lFESpace.GetElementDofs(e, l_dofs);

      const int vdim = lFESpace.GetVDim();
      for (int vd = 0; vd < vdim; ++vd)
      {
         // Extract dofs in lexicographical order
         h_dofs.Copy(h_vdofs);
         hFESpace.DofsToVDofs(vd, h_vdofs);
         x.GetSubVector(h_vdofs, subX);

         for (int p = 0; p < h_dofs.Size(); ++p)
         {
            if (processed[h_dofs[p]])
            {
               subX[p] = 0.0;
            }
         }

         for (int i = 0; i < subX.Size(); ++i)
         {
            subX_lex[i] = subX[hdofmap[i]];
         }

         // Apply doftoQuad map using sum factorization
         auto subX_lex_ = Reshape(subX_lex.Read(), Q1D, Q1D, Q1D);
         auto subY_lex_ = Reshape(subY_lex.Write(), D1D, D1D, D1D);
         auto Bt_ = Reshape(Bt.Read(), D1D, Q1D);

         subY_lex = 0.0;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy(dx, dy) = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double s = subX_lex_(qx, qy, qz);
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
                     sol_xy(dx, dy) += wy * sol_x[dx];
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
                     subY_lex_(dx, dy, dz) += wz * sol_xy(dx, dy);
                  }
               }
            }
         }

         for (int i = 0; i < subY.Size(); ++i)
         {
            subY[ldofmap[i]] = subY_lex[i];
         }

         l_dofs.Copy(l_vdofs);
         lFESpace.DofsToVDofs(vd, l_vdofs);
         y.AddElementVector(l_vdofs, subY);
      }

      for (int p = 0; p < h_dofs.Size(); ++p)
      {
         processed[h_dofs[p]] = 1;
      }
   }

   delete[] sol_xy_;
   delete[] sol_x;
}

TrueTransferOperator::TrueTransferOperator(const FiniteElementSpace& lFESpace_,
                                           const FiniteElementSpace& hFESpace_)
{
   localTransferOperator = new TransferOperator(lFESpace_, hFESpace_);

   opr = new TripleProductOperator(
       hFESpace_.GetRestrictionMatrix(), localTransferOperator,
       lFESpace_.GetProlongationMatrix(), false, false, false);
}

TrueTransferOperator::~TrueTransferOperator()
{
   delete opr;
   delete localTransferOperator;
}

void TrueTransferOperator::Mult(const Vector& x, Vector& y) const
{
   opr->Mult(x, y);
}

void TrueTransferOperator::MultTranspose(const Vector& x, Vector& y) const
{
   opr->MultTranspose(x, y);
}

} // namespace mfem