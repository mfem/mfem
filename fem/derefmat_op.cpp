// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "derefmat_op.hpp"
#include "fes_kernels.hpp"

/// \cond DO_NOT_DOCUMENT
namespace mfem
{

namespace
{

Geometry::Type PointMatrixGeometry(const DenseMatrixStack &pmats, int matrix)
{
   const int dim = pmats.SizeI(matrix);
   const int nverts = pmats.SizeJ(matrix);
   for (int geom = Geometry::DimStart[dim];
        geom < Geometry::DimStart[dim + 1]; geom++)
   {
      if (Geometry::NumVerts[geom] == nverts)
      {
         return Geometry::Type(geom);
      }
   }
   MFEM_ABORT("invalid point matrix dimensions");
   return Geometry::INVALID;
}

void GetMixedNodalRestriction(const FiniteElement &coarse_fe,
                              const FiniteElement &fine_fe,
                              Geometry::Type fine_geom,
                              const DenseMatrix &point_matrix,
                              DenseMatrix &restriction)
{
   MFEM_VERIFY(dynamic_cast<const NodalFiniteElement *>(&coarse_fe) &&
               dynamic_cast<const NodalFiniteElement *>(&fine_fe),
               "mixed-child derefinement currently requires nodal elements.");

   const int dim = coarse_fe.GetDim();
   restriction.SetSize(coarse_fe.GetDof(), fine_fe.GetDof());
   IsoparametricTransformation trans;
   trans.SetIdentityTransformation(fine_geom);
   trans.SetPointMat(point_matrix);
   trans.SetIntPoint(&Geometries.GetCenter(fine_geom));

   IntegrationPoint origin;
   origin.Set3(0.0, 0.0, 0.0);
   Vector parent_origin(dim);
   trans.Transform(origin, parent_origin);

   Vector fine_shape(fine_fe.GetDof());
   const IntegrationRule &coarse_nodes = coarse_fe.GetNodes();
   for (int i = 0; i < coarse_fe.GetDof(); i++)
   {
      real_t parent_coords[3], fine_coords[3];
      coarse_nodes.IntPoint(i).Get(parent_coords, dim);
      Vector offset(parent_coords, dim), fine_point(fine_coords, dim);
      offset -= parent_origin;
      trans.InverseJacobian().Mult(offset, fine_point);
      IntegrationPoint ip;
      ip.Set(fine_coords, dim);
      constexpr real_t eps = MFEM_PRECISION_SWITCH(1e-14, 1e-6f);
      if (Geometries.CheckPoint(fine_geom, ip, eps))
      {
         fine_fe.CalcShape(ip, fine_shape);
         restriction.SetRow(i, fine_shape);
      }
      else
      {
         restriction.SetRow(i, infinity());
      }
   }
   restriction.Threshold(1e-12);
}

} // namespace

namespace internal
{
template <Ordering::Type Order, bool Atomic>
static void DerefMultKernelImpl(const DerefineMatrixOp &op, const Vector &x,
                                Vector &y)
{
   DerefineMatrixOpMultFunctor<Order, Atomic> func;
   func.xptr = x.Read();
   y.UseDevice();
   y = 0.;
   func.yptr = y.ReadWrite();
   func.bsptr = op.block_storage.Read();
   func.boptr = op.block_offsets.Read();
   func.brptr = op.block_row_idcs_offsets.Read();
   func.bcptr = op.block_col_idcs_offsets.Read();
   func.rptr = op.row_idcs.Read();
   func.cptr = op.col_idcs.Read();
   func.vdims = op.fespace->GetVDim();
   func.nblocks = op.block_offsets.Size();
   func.width = op.Width() / func.vdims;
   func.height = op.Height() / func.vdims;
   func.Run(op.max_rows);
}

} // namespace internal

DerefineMatrixOp::DerefineMatrixOp(FiniteElementSpace &fespace_, int old_ndofs,
                                   const Table *old_elem_dof,
                                   const Table *old_elem_fos)
   : Operator(fespace_.GetVSize(), old_ndofs * fespace_.GetVDim()),
     fespace(&fespace_)
{
   static Kernels kernels;
   constexpr int max_team_size = 256;
   /// TODO: Implement DofTransformation support

   MFEM_VERIFY(fespace->Nonconforming(),
               "Not implemented for conforming meshes.");
   MFEM_VERIFY(old_ndofs, "Missing previous (finer) space.");
   MFEM_VERIFY(fespace->GetNDofs() <= old_ndofs,
               "Previous space is not finer.");

   const CoarseFineTransformations &dtrans =
      fespace->GetMesh()->ncmesh->GetDerefinementTransforms();

   MFEM_ASSERT(dtrans.embeddings.Size() == old_elem_dof->Size(), "");

   const bool is_dg = fespace->FEColl()->GetContType()
                      == FiniteElementCollection::DISCONTINUOUS;
   DenseMatrix localRVO; // for variable-order only

   DenseTensor localR[Geometry::NumGeom];
   const auto get_mixed_restriction = [&](int k, DenseMatrix &restriction)
   {
      const Embedding &emb = dtrans.embeddings[k];
      const Geometry::Type parent_geom =
         fespace->GetMesh()->GetElementBaseGeometry(emb.parent);
      const DenseMatrixStack &pmats = dtrans.point_matrices[parent_geom];
      const Geometry::Type child_geom = PointMatrixGeometry(pmats, emb.matrix);
      const FiniteElement *parent_fe =
         fespace->FEColl()->FiniteElementForGeometry(parent_geom);
      const FiniteElement *child_fe =
         fespace->FEColl()->FiniteElementForGeometry(child_geom);
      GetMixedNodalRestriction(*parent_fe, *child_fe, child_geom,
                               pmats(emb.matrix), restriction);
   };
   bool mixed_children = false;
   for (int k = 0; k < dtrans.embeddings.Size(); k++)
   {
      const Embedding &emb = dtrans.embeddings[k];
      const Geometry::Type parent_geom =
         fespace->GetMesh()->GetElementBaseGeometry(emb.parent);
      const DenseMatrixStack &pmats = dtrans.point_matrices[parent_geom];
      mixed_children |= PointMatrixGeometry(pmats, emb.matrix) != parent_geom;
   }
   int total_rows = 0;
   int total_cols = 0;
   block_offsets.SetSize(dtrans.embeddings.Size());
   block_offsets.HostWrite();
   if (mixed_children)
   {
      MFEM_VERIFY(!fespace->IsVariableOrder(),
                  "mixed-child derefinement does not support variable order.");
      MFEM_VERIFY(!is_dg,
                  "mixed-child derefinement of discontinuous fields requires "
                  "a conservative restriction.");
      int total_size = 0;
      DenseMatrix restriction;
      for (int k = 0; k < dtrans.embeddings.Size(); ++k)
      {
         get_mixed_restriction(k, restriction);
         block_offsets[k] = total_size;
         total_rows += restriction.Height();
         total_cols += restriction.Width();
         total_size += restriction.Height() * restriction.Width();
      }
      block_storage.SetSize(total_size);
   }
   else if (fespace->IsVariableOrder())
   {
      // TODO: any potential for some compression here?
      // determine storage size and offsets
      block_offsets[0] = 0;
      int total_size = 0;
      for (int k = 0; k < dtrans.embeddings.Size(); ++k)
      {
         const Embedding &emb = dtrans.embeddings[k];
         const FiniteElement *fe = fespace->GetFE(emb.parent);
         const int ldof = fe->GetDof();
         if (k + 1 < dtrans.embeddings.Size())
         {
            block_offsets[k + 1] = block_offsets[k] + ldof * ldof;
         }
         total_rows += ldof;
         total_cols += ldof;
         total_size += ldof * ldof;
      }
      block_storage.SetSize(total_size);
   }
   else
   {
      // compression scheme:
      // block_offsets is the start of each block, potentially repeated
      // only need to store localR for used shapes
      Mesh::GeometryList elem_geoms(*fespace->GetMesh());

      int geom_offsets[Geometry::NumGeom];
      {
         int size = 0;
         for (int i = 0; i < elem_geoms.Size(); ++i)
         {
            fespace->GetLocalDerefinementMatrices(elem_geoms[i],
                                                  localR[elem_geoms[i]]);
            geom_offsets[elem_geoms[i]] = size;
            size += localR[elem_geoms[i]].TotalSize();
         }
         block_storage.SetSize(size);
         // copy blocks into block_storage
         auto bs_ptr = block_storage.HostWrite();
         for (int i = 0; i < elem_geoms.Size(); ++i)
         {
            std::copy(localR[elem_geoms[i]].Data(),
                      localR[elem_geoms[i]].Data()
                      + localR[elem_geoms[i]].TotalSize(),
                      bs_ptr);
            bs_ptr += localR[elem_geoms[i]].TotalSize();
         }
      }
      for (int k = 0; k < dtrans.embeddings.Size(); ++k)
      {
         const Embedding &emb = dtrans.embeddings[k];
         Geometry::Type geom =
            fespace->GetMesh()->GetElementBaseGeometry(emb.parent);

         auto size = localR[geom].SizeI() * localR[geom].SizeJ();
         total_rows += localR[geom].SizeI();
         total_cols += localR[geom].SizeJ();
         // set block offsets and sizes
         block_offsets[k] = geom_offsets[geom] + size * emb.matrix;
      }
   }
   row_idcs.SetSize(total_rows);
   row_idcs.HostWrite();
   col_idcs.SetSize(total_cols);
   col_idcs.HostWrite();
   block_row_idcs_offsets.SetSize(dtrans.embeddings.Size() + 1);
   block_row_idcs_offsets.HostWrite();
   block_col_idcs_offsets.SetSize(dtrans.embeddings.Size() + 1);
   block_col_idcs_offsets.HostWrite();
   block_row_idcs_offsets[0] = 0;
   block_col_idcs_offsets[0] = 0;

   // compute index information
   Array<int> dofs, old_dofs;
   max_rows = 1;

   {
      Array<int> mark(fespace->GetNDofs());
      mark = 0;
      auto bs_ptr = block_storage.HostWrite();
      int ridx = 0;
      int cidx = 0;
      int num_marked = 0;
      for (int k = 0; k < dtrans.embeddings.Size(); k++)
      {
         const Embedding &emb = dtrans.embeddings[k];
         Geometry::Type geom =
            fespace->GetMesh()->GetElementBaseGeometry(emb.parent);

         DenseMatrix mixed_restriction;
         if (mixed_children)
         {
            get_mixed_restriction(k, mixed_restriction);
            std::copy(mixed_restriction.Data(),
                      mixed_restriction.Data() + mixed_restriction.Height()
                      * mixed_restriction.Width(), bs_ptr + block_offsets[k]);
         }
         else if (fespace->IsVariableOrder())
         {
            const FiniteElement *fe = fespace->GetFE(emb.parent);
            const DenseMatrixStack &pmats = dtrans.point_matrices[geom];
            const int ldof = fe->GetDof();

            IsoparametricTransformation isotr;
            isotr.SetIdentityTransformation(geom);

            localRVO.SetSize(ldof, ldof);
            isotr.SetPointMat(pmats(emb.matrix));
            // Local restriction is size ldofxldof assuming that the parent
            // and child are of same polynomial order.
            fe->GetLocalRestriction(isotr, localRVO);
            // copy block
            auto size = localRVO.Height() * localRVO.Width();
            std::copy(localRVO.Data(), localRVO.Data() + size, bs_ptr);
            bs_ptr += size;
         }
         DenseMatrix &lR = mixed_children ? mixed_restriction :
                            (fespace->IsVariableOrder() ? localRVO :
                             localR[geom](emb.matrix));
         block_row_idcs_offsets[k + 1] =
            block_row_idcs_offsets[k] + lR.Height();
         block_col_idcs_offsets[k + 1] = block_col_idcs_offsets[k] + lR.Width();
         max_rows = std::max(lR.Height(), max_rows);
         // index information
         fespace->elem_dof->GetRow(emb.parent, dofs);
         old_elem_dof->GetRow(k, old_dofs);
         MFEM_VERIFY(old_dofs.Size() == lR.Width() &&
                     dofs.Size() == lR.Height(),
                     "incompatible parent and child DoFs.");
         for (int i = 0; i < lR.Height(); ++i, ++ridx)
         {
            if (!std::isfinite(lR(i, 0)))
            {
               row_idcs[ridx] = INT_MAX;
               continue;
            }
            int r = dofs[i];
            int m = (r >= 0) ? r : (-1 - r);
            if (is_dg || !mark[m])
            {
               row_idcs[ridx] = r;
               mark[m] = 1;
               ++num_marked;
            }
            else
            {
               row_idcs[ridx] = INT_MAX;
            }
         }
         for (int i = 0; i < lR.Width(); ++i, ++cidx)
         {
            col_idcs[cidx] = old_dofs[i];
         }
      }
      if (!is_dg && !fespace->IsVariableOrder())
      {
         MFEM_VERIFY(num_marked * fespace->GetVDim() == Height(),
                     "internal error: not all rows were set.");
      }
   }
   // if not using GPU, set max_rows/max_cols to zero
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      max_rows = std::min(max_rows, max_team_size);
   }
   else
   {
      max_rows = 1;
   }
}

void DerefineMatrixOp::Mult(const Vector &x, Vector &y) const
{
   const bool is_dg = fespace->FEColl()->GetContType()
                      == FiniteElementCollection::DISCONTINUOUS;
   // DG needs atomic summation
   MultKernel::Run(fespace->GetOrdering(), is_dg, *this, x, y);
}

DerefineMatrixOp::Kernels::Kernels()
{
   MultKernel::Specialization<Ordering::byNODES, false>::Add();
   MultKernel::Specialization<Ordering::byVDIM, false>::Add();
   MultKernel::Specialization<Ordering::byNODES, true>::Add();
   MultKernel::Specialization<Ordering::byVDIM, true>::Add();
}

template <Ordering::Type Order, bool Atomic>
DerefineMatrixOp::MultKernelType DerefineMatrixOp::MultKernel::Kernel()
{
   return internal::DerefMultKernelImpl<Order, Atomic>;
}

DerefineMatrixOp::MultKernelType
DerefineMatrixOp::MultKernel::Fallback(Ordering::Type, bool)
{
   MFEM_ABORT("invalid MultKernel parameters");
}
} // namespace mfem
/// \endcond DO_NOT_DOCUMENT
