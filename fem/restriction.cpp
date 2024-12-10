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

#include "restriction.hpp"
#include "normal_deriv_restriction.hpp"
#include "gridfunc.hpp"
#include "fespace.hpp"
#include "pgridfunc.hpp"
#include "qspace.hpp"
#include "fe/face_map_utils.hpp"
#include "../general/forall.hpp"

#include <climits>

namespace mfem
{

ElementRestriction::ElementRestriction(const FiniteElementSpace &f,
                                       ElementDofOrdering e_ordering)
   : fes(f),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
     nedofs(ne*dof),
     offsets(ndofs+1),
     indices(ne*dof),
     gather_map(ne*dof)
{
   // Assuming all finite elements are the same.
   height = vdim*ne*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   const int *dof_map = NULL;
   if (dof_reorder && ne > 0)
   {
      for (int e = 0; e < ne; ++e)
      {
         const FiniteElement *fe = fes.GetFE(e);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
      const FiniteElement *fe = fes.GetFE(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
      dof_map = fe_dof_map.GetData();
   }
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* element_map = e2dTable.GetJ();
   // We will be keeping a count of how many local nodes point to its global dof
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int sgid = element_map[dof*e + d];  // signed
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point to it
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int sdid = dof_reorder ? dof_map[d] : 0;  // signed
         const int did = (!dof_reorder)?d:(sdid >= 0 ? sdid : -1-sdid);
         const int sgid = element_map[dof*e + did];  // signed
         const int gid = (sgid >= 0) ? sgid : -1-sgid;
         const int lid = dof*e + d;
         const bool plus = (sgid >= 0 && sdid >= 0) || (sgid < 0 && sdid < 0);
         gather_map[lid] = plus ? gid : -1-gid;
         indices[offsets[gid]++] = plus ? lid : -1-lid;
      }
   }
   // We shifted the offsets vector by 1 by using it as a counter.
   // Now we shift it back.
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void ElementRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, ne);
   auto d_gather_map = gather_map.Read();
   mfem::forall(dof*ne, [=] MFEM_HOST_DEVICE (int i)
   {
      const int gid = d_gather_map[i];
      const bool plus = gid >= 0;
      const int j = plus ? gid : -1-gid;
      for (int c = 0; c < vd; ++c)
      {
         const real_t dof_value = d_x(t?c:j, t?j:c);
         d_y(i % nd, c, i / nd) = plus ? dof_value : -dof_value;
      }
   });
}

void ElementRestriction::MultUnsigned(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, ne);
   auto d_gather_map = gather_map.Read();

   mfem::forall(dof*ne, [=] MFEM_HOST_DEVICE (int i)
   {
      const int gid = d_gather_map[i];
      const int j = gid >= 0 ? gid : -1-gid;
      for (int c = 0; c < vd; ++c)
      {
         d_y(i % nd, c, i / nd) = d_x(t?c:j, t?j:c);
      }
   });
}

template <bool ADD>
void ElementRestriction::TAddMultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(ADD ? y.ReadWrite() : y.Write(), t?vd:ndofs, t?ndofs:vd);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         real_t dof_value = 0;
         for (int j = offset; j < next_offset; ++j)
         {
            const int idx_j = (d_indices[j] >= 0) ? d_indices[j] : -1 - d_indices[j];
            dof_value += ((d_indices[j] >= 0) ? d_x(idx_j % nd, c, idx_j / nd) :
                          -d_x(idx_j % nd, c, idx_j / nd));
         }
         if (ADD) { d_y(t?c:i,t?i:c) += dof_value; }
         else { d_y(t?c:i,t?i:c) = dof_value; }
      }
   });
}

void ElementRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   constexpr bool ADD = false;
   TAddMultTranspose<ADD>(x, y);
}

void ElementRestriction::AddMultTranspose(const Vector& x, Vector& y,
                                          const real_t a) const
{
   MFEM_VERIFY(a == 1.0, "General coefficient case is not yet supported!");
   constexpr bool ADD = true;
   TAddMultTranspose<ADD>(x, y);
}

void ElementRestriction::MultTransposeUnsigned(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         real_t dof_value = 0;
         for (int j = offset; j < next_offset; ++j)
         {
            const int idx_j = (d_indices[j] >= 0) ? d_indices[j] : -1 - d_indices[j];
            dof_value += d_x(idx_j % nd, c, idx_j / nd);
         }
         d_y(t?c:i,t?i:c) = dof_value;
      }
   });
}

void ElementRestriction::MultLeftInverse(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         real_t dof_value = 0;
         const int j = next_offset - 1;
         const int idx_j = (d_indices[j] >= 0) ? d_indices[j] : -1 - d_indices[j];
         dof_value = (d_indices[j] >= 0) ? d_x(idx_j % nd, c, idx_j / nd) :
                     -d_x(idx_j % nd, c, idx_j / nd);
         d_y(t?c:i,t?i:c) = dof_value;
      }
   });
}

void ElementRestriction::BooleanMask(Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;

   Array<char> processed(vd * ndofs);
   processed = 0;

   auto d_offsets = offsets.HostRead();
   auto d_indices = indices.HostRead();
   auto d_x = Reshape(processed.HostReadWrite(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.HostWrite(), nd, vd, ne);
   for (int i = 0; i < ndofs; ++i)
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i+1];
      for (int c = 0; c < vd; ++c)
      {
         for (int j = offset; j < next_offset; ++j)
         {
            const int idx_j = d_indices[j];
            if (d_x(t?c:i,t?i:c))
            {
               d_y(idx_j % nd, c, idx_j / nd) = 0.0;
            }
            else
            {
               d_y(idx_j % nd, c, idx_j / nd) = 1.0;
               d_x(t?c:i,t?i:c) = 1;
            }
         }
      }
   }
}

void ElementRestriction::FillSparseMatrix(const Vector &mat_ea,
                                          SparseMatrix &mat) const
{
   mat.GetMemoryI().New(mat.Height()+1, mat.GetMemoryI().GetMemoryType());
   const int nnz = FillI(mat);
   mat.GetMemoryJ().New(nnz, mat.GetMemoryJ().GetMemoryType());
   mat.GetMemoryData().New(nnz, mat.GetMemoryData().GetMemoryType());
   FillJAndData(mat_ea, mat);
}

static MFEM_HOST_DEVICE int GetMinElt(const int *my_elts, const int nbElts,
                                      const int *nbr_elts, const int nbrNbElts)
{
   // Find the minimal element index found in both my_elts[] and nbr_elts[]
   int min_el = INT_MAX;
   for (int i = 0; i < nbElts; i++)
   {
      const int e_i = my_elts[i];
      if (e_i >= min_el) { continue; }
      for (int j = 0; j < nbrNbElts; j++)
      {
         if (e_i==nbr_elts[j])
         {
            min_el = e_i; // we already know e_i < min_el
            break;
         }
      }
   }
   return min_el;
}

/** Returns the index where a non-zero entry should be added and increment the
    number of non-zeros for the row i_L. */
static MFEM_HOST_DEVICE int GetAndIncrementNnzIndex(const int i_L, int* I)
{
   int ind = AtomicAdd(I[i_L],1);
   return ind;
}

int ElementRestriction::FillI(SparseMatrix &mat) const
{
   static constexpr int Max = MaxNbNbr;
   const int all_dofs = ndofs;
   const int vd = vdim;
   const int elt_dofs = dof;
   auto I = mat.ReadWriteI();
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gather_map = gather_map.Read();
   mfem::forall(vd*all_dofs+1, [=] MFEM_HOST_DEVICE (int i_L)
   {
      I[i_L] = 0;
   });
   mfem::forall(ne*elt_dofs, [=] MFEM_HOST_DEVICE (int l_dof)
   {
      const int e = l_dof/elt_dofs;
      const int i = l_dof%elt_dofs;

      int i_elts[Max];
      const int i_gm = e*elt_dofs + i;
      const int i_L = d_gather_map[i_gm];
      const int i_offset = d_offsets[i_L];
      const int i_next_offset = d_offsets[i_L+1];
      const int i_nbElts = i_next_offset - i_offset;
      MFEM_ASSERT_KERNEL(
         i_nbElts <= Max,
         "The connectivity of this mesh is beyond the max, increase the "
         "MaxNbNbr variable to comply with your mesh.");
      for (int e_i = 0; e_i < i_nbElts; ++e_i)
      {
         const int i_E = d_indices[i_offset+e_i];
         i_elts[e_i] = i_E/elt_dofs;
      }
      for (int j = 0; j < elt_dofs; j++)
      {
         const int j_gm = e*elt_dofs + j;
         const int j_L = d_gather_map[j_gm];
         const int j_offset = d_offsets[j_L];
         const int j_next_offset = d_offsets[j_L+1];
         const int j_nbElts = j_next_offset - j_offset;
         MFEM_ASSERT_KERNEL(
            j_nbElts <= Max,
            "The connectivity of this mesh is beyond the max, increase the "
            "MaxNbNbr variable to comply with your mesh.");
         if (i_nbElts == 1 || j_nbElts == 1) // no assembly required
         {
            GetAndIncrementNnzIndex(i_L, I);
         }
         else // assembly required
         {
            int j_elts[Max];
            for (int e_j = 0; e_j < j_nbElts; ++e_j)
            {
               const int j_E = d_indices[j_offset+e_j];
               const int elt = j_E/elt_dofs;
               j_elts[e_j] = elt;
            }
            int min_e = GetMinElt(i_elts, i_nbElts, j_elts, j_nbElts);
            if (e == min_e) // add the nnz only once
            {
               GetAndIncrementNnzIndex(i_L, I);
            }
         }
      }
   });
   // We need to sum the entries of I, we do it on CPU as it is very sequential.
   auto h_I = mat.HostReadWriteI();
   const int nTdofs = vd*all_dofs;
   int sum = 0;
   for (int i = 0; i < nTdofs; i++)
   {
      const int nnz = h_I[i];
      h_I[i] = sum;
      sum+=nnz;
   }
   h_I[nTdofs] = sum;
   // We return the number of nnz
   return h_I[nTdofs];
}

void ElementRestriction::FillJAndData(const Vector &ea_data,
                                      SparseMatrix &mat) const
{
   static constexpr int Max = MaxNbNbr;
   const int all_dofs = ndofs;
   const int vd = vdim;
   const int elt_dofs = dof;
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gather_map = gather_map.Read();
   auto mat_ea = Reshape(ea_data.Read(), elt_dofs, elt_dofs, ne);
   mfem::forall(ne*elt_dofs, [=] MFEM_HOST_DEVICE (int l_dof)
   {
      const int e = l_dof/elt_dofs;
      const int i = l_dof%elt_dofs;

      int i_elts[Max];
      int i_B[Max];
      const int i_gm = e*elt_dofs + i;
      const int i_L = d_gather_map[i_gm];
      const int i_offset = d_offsets[i_L];
      const int i_next_offset = d_offsets[i_L+1];
      const int i_nbElts = i_next_offset - i_offset;
      MFEM_ASSERT_KERNEL(
         i_nbElts <= Max,
         "The connectivity of this mesh is beyond the max, increase the "
         "MaxNbNbr variable to comply with your mesh.");
      for (int e_i = 0; e_i < i_nbElts; ++e_i)
      {
         const int i_E = d_indices[i_offset+e_i];
         i_elts[e_i] = i_E/elt_dofs;
         i_B[e_i]    = i_E%elt_dofs;
      }
      for (int j = 0; j < elt_dofs; j++)
      {
         const int j_gm = e*elt_dofs + j;
         const int j_L = d_gather_map[j_gm];
         const int j_offset = d_offsets[j_L];
         const int j_next_offset = d_offsets[j_L+1];
         const int j_nbElts = j_next_offset - j_offset;
         if (i_nbElts == 1 || j_nbElts == 1) // no assembly required
         {
            const int nnz = GetAndIncrementNnzIndex(i_L, I);
            J[nnz] = j_L;
            Data[nnz] = mat_ea(j,i,e);
         }
         else // assembly required
         {
            int j_elts[Max];
            int j_B[Max];
            for (int e_j = 0; e_j < j_nbElts; ++e_j)
            {
               const int j_E = d_indices[j_offset+e_j];
               const int elt = j_E/elt_dofs;
               j_elts[e_j] = elt;
               j_B[e_j]    = j_E%elt_dofs;
            }
            int min_e = GetMinElt(i_elts, i_nbElts, j_elts, j_nbElts);
            if (e == min_e) // add the nnz only once
            {
               real_t val = 0.0;
               for (int k = 0; k < i_nbElts; k++)
               {
                  const int e_i = i_elts[k];
                  const int i_Bloc = i_B[k];
                  for (int l = 0; l < j_nbElts; l++)
                  {
                     const int e_j = j_elts[l];
                     const int j_Bloc = j_B[l];
                     if (e_i == e_j)
                     {
                        val += mat_ea(j_Bloc, i_Bloc, e_i);
                     }
                  }
               }
               const int nnz = GetAndIncrementNnzIndex(i_L, I);
               J[nnz] = j_L;
               Data[nnz] = val;
            }
         }
      }
   });
   // We need to shift again the entries of I, we do it on CPU as it is very
   // sequential.
   auto h_I = mat.HostReadWriteI();
   const int size = vd*all_dofs;
   for (int i = 0; i < size; i++)
   {
      h_I[size-i] = h_I[size-(i+1)];
   }
   h_I[0] = 0;
}

L2ElementRestriction::L2ElementRestriction(const FiniteElementSpace &fes)
   : ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndof(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
     ndofs(fes.GetNDofs())
{
   height = vdim*ne*ndof;
   width = vdim*ne*ndof;
}

void L2ElementRestriction::Mult(const Vector &x, Vector &y) const
{
   const int nd = ndof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, ne);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int idx = i;
      const int dof = idx % nd;
      const int e = idx / nd;
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, e) = d_x(t?c:idx, t?idx:c);
      }
   });
}

template <bool ADD>
void L2ElementRestriction::TAddMultTranspose(const Vector &x, Vector &y) const
{
   const int nd = ndof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(ADD ? y.ReadWrite() : y.Write(), t?vd:ndofs, t?ndofs:vd);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int idx = i;
      const int dof = idx % nd;
      const int e = idx / nd;
      for (int c = 0; c < vd; ++c)
      {
         if (ADD) { d_y(t?c:idx,t?idx:c) += d_x(dof, c, e); }
         else { d_y(t?c:idx,t?idx:c) = d_x(dof, c, e); }
      }
   });
}

void L2ElementRestriction::MultTranspose(const Vector &x, Vector &y) const
{
   constexpr bool ADD = false;
   TAddMultTranspose<ADD>(x, y);
}

void L2ElementRestriction::AddMultTranspose(const Vector &x, Vector &y,
                                            const real_t a) const
{
   MFEM_VERIFY(a == 1.0, "General coefficient case is not yet supported!");
   constexpr bool ADD = true;
   TAddMultTranspose<ADD>(x, y);
}

void L2ElementRestriction::FillI(SparseMatrix &mat) const
{
   const int elem_dofs = ndof;
   const int vd = vdim;
   auto I = mat.WriteI();
   const int isize = mat.Height() + 1;
   const int interior_dofs = ne*elem_dofs*vd;
   mfem::forall(isize, [=] MFEM_HOST_DEVICE (int dof)
   {
      I[dof] = dof<interior_dofs ? elem_dofs : 0;
   });
}

static MFEM_HOST_DEVICE int AddNnz(const int iE, int *I, const int dofs)
{
   int val = AtomicAdd(I[iE],dofs);
   return val;
}

void L2ElementRestriction::FillJAndData(const Vector &ea_data,
                                        SparseMatrix &mat) const
{
   const int elem_dofs = ndof;
   const int vd = vdim;
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   auto mat_ea = Reshape(ea_data.Read(), elem_dofs, elem_dofs, ne);
   mfem::forall(ne*elem_dofs*vd, [=] MFEM_HOST_DEVICE (int iE)
   {
      const int offset = AddNnz(iE,I,elem_dofs);
      const int e = iE/elem_dofs;
      const int i = iE%elem_dofs;
      for (int j = 0; j < elem_dofs; j++)
      {
         J[offset+j] = e*elem_dofs+j;
         Data[offset+j] = mat_ea(j,i,e);
      }
   });
}

ConformingFaceRestriction::ConformingFaceRestriction(
   const FiniteElementSpace &fes,
   const ElementDofOrdering f_ordering,
   const FaceType type,
   bool build)
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     face_dofs(nf > 0 ? fes.GetFaceElement(0)->GetDof() : 0),
     elem_dofs(fes.GetFE(0)->GetDof()),
     nfdofs(nf*face_dofs),
     ndofs(fes.GetNDofs()),
     scatter_indices(nf*face_dofs),
     gather_offsets(ndofs+1),
     gather_indices(nf*face_dofs),
     face_map(face_dofs)
{
   height = vdim*nf*face_dofs;
   width = fes.GetVSize();
   if (nf==0) { return; }

   CheckFESpace(f_ordering);

   // Get the mapping from lexicographic DOF ordering to native ordering.
   const TensorBasisElement* el =
      dynamic_cast<const TensorBasisElement*>(fes.GetFE(0));
   const Array<int> &dof_map_ = el->GetDofMap();
   if (dof_map_.Size() > 0)
   {
      vol_dof_map.MakeRef(dof_map_);
   }
   else
   {
      // For certain types of elements dof_map_ is empty. In this case, that
      // means the element is already ordered lexicographically, so the
      // permutation is the identity.
      vol_dof_map.SetSize(elem_dofs);
      for (int i = 0; i < elem_dofs; ++i) { vol_dof_map[i] = i; }
   }

   if (!build) { return; }
   ComputeScatterIndicesAndOffsets(f_ordering, type);
   ComputeGatherIndices(f_ordering,type);
}

ConformingFaceRestriction::ConformingFaceRestriction(
   const FiniteElementSpace &fes,
   const ElementDofOrdering f_ordering,
   const FaceType type)
   : ConformingFaceRestriction(fes, f_ordering, type, true)
{ }

void ConformingFaceRestriction::Mult(const Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_indices = scatter_indices.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
   mfem::forall(nfdofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int s_idx = d_indices[i];
      const int sgn = (s_idx >= 0) ? 1 : -1;
      const int idx = (s_idx >= 0) ? s_idx : -1 - s_idx;
      const int dof = i % nface_dofs;
      const int face = i / nface_dofs;
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, face) = sgn*d_x(t?c:idx, t?idx:c);
      }
   });
}

static void ConformingFaceRestriction_AddMultTranspose(
   const int ndofs,
   const int face_dofs,
   const int nf,
   const int vdim,
   const bool by_vdim,
   const Array<int> &gather_offsets,
   const Array<int> &gather_indices,
   const Vector &x,
   Vector &y,
   bool use_signs,
   const real_t a)
{
   MFEM_VERIFY(a == 1.0, "General coefficient case is not yet supported!");
   if (nf==0) { return; }
   // Assumes all elements have the same number of dofs
   auto d_offsets = gather_offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x.Read(), face_dofs, vdim, nf);
   auto d_y = Reshape(y.ReadWrite(), by_vdim?vdim:ndofs, by_vdim?ndofs:vdim);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vdim; ++c)
      {
         real_t dof_value = 0;
         for (int j = offset; j < next_offset; ++j)
         {
            const int s_idx_j = d_indices[j];
            const real_t sgn = (s_idx_j >= 0 || !use_signs) ? 1.0 : -1.0;
            const int idx_j = (s_idx_j >= 0) ? s_idx_j : -1 - s_idx_j;
            dof_value += sgn*d_x(idx_j % face_dofs, c, idx_j / face_dofs);
         }
         d_y(by_vdim?c:i,by_vdim?i:c) += dof_value;
      }
   });
}

void ConformingFaceRestriction::AddMultTranspose(
   const Vector& x, Vector& y, const real_t a) const
{
   ConformingFaceRestriction_AddMultTranspose(
      ndofs, face_dofs, nf, vdim, byvdim, gather_offsets, gather_indices, x, y,
      true, a);
}

void ConformingFaceRestriction::AddMultTransposeUnsigned(
   const Vector& x, Vector& y, const real_t a) const
{
   ConformingFaceRestriction_AddMultTranspose(
      ndofs, face_dofs, nf, vdim, byvdim, gather_offsets, gather_indices, x, y,
      false, a);
}

void ConformingFaceRestriction::CheckFESpace(const ElementDofOrdering
                                             f_ordering)
{
#ifdef MFEM_USE_MPI

   // If the underlying finite element space is parallel, ensure the face
   // neighbor information is generated.
   if (const ParFiniteElementSpace *pfes
       = dynamic_cast<const ParFiniteElementSpace*>(&fes))
   {
      pfes->GetParMesh()->ExchangeFaceNbrData();
   }

#endif

#ifdef MFEM_DEBUG
   const FiniteElement *fe0 = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe0);
   MFEM_VERIFY(tfe != NULL,
               "ConformingFaceRestriction only supports TensorBasisElements");
   MFEM_VERIFY(tfe->GetBasisType()==BasisType::GaussLobatto ||
               tfe->GetBasisType()==BasisType::Positive,
               "ConformingFaceRestriction only supports Gauss-Lobatto and Bernstein bases");

   // Assuming all finite elements are using Gauss-Lobatto.
   const bool dof_reorder = (f_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe = fes.GetFaceElement(f);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
#endif
}

void ConformingFaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering f_ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter indices and offsets
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if ( face.IsOfFaceType(type) )
      {
         SetFaceDofsScatterIndices(face, f_ind, f_ordering);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      gather_offsets[i] += gather_offsets[i - 1];
   }
}

void ConformingFaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering f_ordering,
   const FaceType type)
{
   Mesh &mesh = *fes.GetMesh();

   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if ( face.IsOfFaceType(type) )
      {
         SetFaceDofsGatherIndices(face, f_ind, f_ordering);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Reset offsets to their initial value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

static inline int absdof(int i) { return i < 0 ? -1-i : i; }

void ConformingFaceRestriction::SetFaceDofsScatterIndices(
   const Mesh::FaceInformation &face,
   const int face_index,
   const ElementDofOrdering f_ordering)
{
   MFEM_ASSERT(!(face.IsNonconformingCoarse()),
               "This method should not be used on nonconforming coarse faces.");
   MFEM_ASSERT(face.element[0].orientation==0,
               "FaceRestriction used on degenerated mesh.");
   MFEM_VERIFY(f_ordering == ElementDofOrdering::LEXICOGRAPHIC,
               "NATIVE ordering is not supported yet");

   fes.GetFE(0)->GetFaceMap(face.element[0].local_face_id, face_map);

   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int elem_index = face.element[0].index;

   for (int face_dof = 0; face_dof < face_dofs; ++face_dof)
   {
      const int lex_volume_dof = face_map[face_dof];
      const int s_volume_dof = AsConst(vol_dof_map)[lex_volume_dof]; // signed
      const int volume_dof = absdof(s_volume_dof);
      const int s_global_dof = elem_map[elem_index*elem_dofs + volume_dof];
      const int global_dof = absdof(s_global_dof);
      const int restriction_dof = face_dofs*face_index + face_dof;
      scatter_indices[restriction_dof] = s_global_dof;
      ++gather_offsets[global_dof + 1];
   }
}

void ConformingFaceRestriction::SetFaceDofsGatherIndices(
   const Mesh::FaceInformation &face,
   const int face_index,
   const ElementDofOrdering f_ordering)
{
   MFEM_ASSERT(!(face.IsNonconformingCoarse()),
               "This method should not be used on nonconforming coarse faces.");
   MFEM_VERIFY(f_ordering == ElementDofOrdering::LEXICOGRAPHIC,
               "NATIVE ordering is not supported yet");

   fes.GetFE(0)->GetFaceMap(face.element[0].local_face_id, face_map);

   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int elem_index = face.element[0].index;

   for (int face_dof = 0; face_dof < face_dofs; ++face_dof)
   {
      const int lex_volume_dof = face_map[face_dof];
      const int s_volume_dof = AsConst(vol_dof_map)[lex_volume_dof];
      const int volume_dof = absdof(s_volume_dof);
      const int s_global_dof = elem_map[elem_index*elem_dofs + volume_dof];
      const int sgn = (s_global_dof >= 0) ? 1 : -1;
      const int global_dof = absdof(s_global_dof);
      const int restriction_dof = face_dofs*face_index + face_dof;
      const int s_restriction_dof = (sgn >= 0) ? restriction_dof : -1 -
                                    restriction_dof;
      gather_indices[gather_offsets[global_dof]++] = s_restriction_dof;
   }
}

// Permute dofs or quads on a face for e2 to match with the ordering of e1
int PermuteFaceL2(const int dim, const int face_id1,
                  const int face_id2, const int orientation,
                  const int size1d, const int index)
{
   switch (dim)
   {
      case 1:
         return 0;
      case 2:
         return internal::PermuteFace2D(face_id1, face_id2, orientation, size1d, index);
      case 3:
         return internal::PermuteFace3D(face_id1, face_id2, orientation, size1d, index);
      default:
         MFEM_ABORT("Unsupported dimension.");
         return 0;
   }
}

L2FaceRestriction::L2FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering f_ordering,
                                     const FaceType type,
                                     const L2FaceValues m,
                                     bool build)
   : fes(fes),
     ordering(f_ordering),
     nf(fes.GetNFbyType(type)),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     face_dofs(nf > 0 ?
               fes.GetTraceElement(0, fes.GetMesh()->GetFaceGeometry(0))->GetDof()
               : 0),
     elem_dofs(fes.GetFE(0)->GetDof()),
     nfdofs(nf*face_dofs),
     ndofs(fes.GetNDofs()),
     type(type),
     m(m),
     scatter_indices1(nf*face_dofs),
     scatter_indices2(m==L2FaceValues::DoubleValued?nf*face_dofs:0),
     gather_offsets(ndofs+1),
     gather_indices((m==L2FaceValues::DoubleValued? 2 : 1)*nf*face_dofs),
     face_map(face_dofs)
{
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*face_dofs;
   width = fes.GetVSize();
   if (!build) { return; }

   CheckFESpace();
   ComputeScatterIndicesAndOffsets();
   ComputeGatherIndices();
}

L2FaceRestriction::L2FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering f_ordering,
                                     const FaceType type,
                                     const L2FaceValues m)
   : L2FaceRestriction(fes, f_ordering, type, m, true)
{ }

void L2FaceRestriction::SingleValuedConformingMult(const Vector& x,
                                                   Vector& y) const
{
   if (nf == 0) { return; }
   MFEM_ASSERT(
      m == L2FaceValues::SingleValued,
      "This method should be called when m == L2FaceValues::SingleValued.");
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_indices1 = scatter_indices1.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
   mfem::forall(nfdofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof = i % nface_dofs;
      const int face = i / nface_dofs;
      const int idx1 = d_indices1[i];
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, face) = d_x(t?c:idx1, t?idx1:c);
      }
   });
}

void L2FaceRestriction::DoubleValuedConformingMult(const Vector& x,
                                                   Vector& y) const
{
   MFEM_ASSERT(
      m == L2FaceValues::DoubleValued,
      "This method should be called when m == L2FaceValues::DoubleValued.");
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nface_dofs, vd, 2, nf);
   mfem::forall(nfdofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof = i % nface_dofs;
      const int face = i / nface_dofs;
      const int idx1 = d_indices1[i];
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, 0, face) = d_x(t?c:idx1, t?idx1:c);
      }
      const int idx2 = d_indices2[i];
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, 1, face) = idx2==-1 ? 0.0 : d_x(t?c:idx2, t?idx2:c);
      }
   });
}

void L2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   if (m==L2FaceValues::DoubleValued)
   {
      DoubleValuedConformingMult(x, y);
   }
   else
   {
      SingleValuedConformingMult(x, y);
   }
}

void L2FaceRestriction::SingleValuedConformingAddMultTranspose(
   const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = gather_offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x.Read(), nface_dofs, vd, nf);
   auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         real_t dof_value = 0;
         for (int j = offset; j < next_offset; ++j)
         {
            int idx_j = d_indices[j];
            dof_value +=  d_x(idx_j % nface_dofs, c, idx_j / nface_dofs);
         }
         d_y(t?c:i,t?i:c) += dof_value;
      }
   });
}

void L2FaceRestriction::DoubleValuedConformingAddMultTranspose(
   const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   const int dofs = nfdofs;
   auto d_offsets = gather_offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x.Read(), nface_dofs, vd, 2, nf);
   auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
   mfem::forall(ndofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         real_t dof_value = 0;
         for (int j = offset; j < next_offset; ++j)
         {
            int idx_j = d_indices[j];
            bool isE1 = idx_j < dofs;
            idx_j = isE1 ? idx_j : idx_j - dofs;
            dof_value +=  isE1 ?
                          d_x(idx_j % nface_dofs, c, 0, idx_j / nface_dofs)
                          :d_x(idx_j % nface_dofs, c, 1, idx_j / nface_dofs);
         }
         d_y(t?c:i,t?i:c) += dof_value;
      }
   });
}

void L2FaceRestriction::AddMultTranspose(const Vector& x, Vector& y,
                                         const real_t a) const
{
   MFEM_VERIFY(a == 1.0, "General coefficient case is not yet supported!");
   if (nf==0) { return; }
   if (m == L2FaceValues::DoubleValued)
   {
      DoubleValuedConformingAddMultTranspose(x, y);
   }
   else
   {
      SingleValuedConformingAddMultTranspose(x, y);
   }
}

void L2FaceRestriction::FillI(SparseMatrix &mat,
                              const bool keep_nbr_block) const
{
   const int nface_dofs = face_dofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int iE1 = d_indices1[fdof];
      const int iE2 = d_indices2[fdof];
      AddNnz(iE1,I,nface_dofs);
      AddNnz(iE2,I,nface_dofs);
   });
}

void L2FaceRestriction::FillJAndData(const Vector &fea_data,
                                     SparseMatrix &mat,
                                     const bool keep_nbr_block) const
{
   const int nface_dofs = face_dofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   auto mat_fea = Reshape(fea_data.Read(), nface_dofs, nface_dofs, 2, nf);
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int f  = fdof/nface_dofs;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      const int iE2 = d_indices2[f*nface_dofs+iF];
      const int offset1 = AddNnz(iE1,I,nface_dofs);
      const int offset2 = AddNnz(iE2,I,nface_dofs);
      for (int jF = 0; jF < nface_dofs; jF++)
      {
         const int jE1 = d_indices1[f*nface_dofs+jF];
         const int jE2 = d_indices2[f*nface_dofs+jF];
         J[offset2+jF] = jE1;
         J[offset1+jF] = jE2;
         Data[offset2+jF] = mat_fea(jF,iF,0,f);
         Data[offset1+jF] = mat_fea(jF,iF,1,f);
      }
   });
}

void L2FaceRestriction::AddFaceMatricesToElementMatrices(const Vector &fea_data,
                                                         Vector &ea_data) const
{
   const int nface_dofs = face_dofs;
   const int nelem_dofs = elem_dofs;
   const int NE = ne;
   if (m==L2FaceValues::DoubleValued)
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto mat_fea = Reshape(fea_data.Read(), nface_dofs, nface_dofs, 2, nf);
      auto mat_ea = Reshape(ea_data.ReadWrite(), nelem_dofs, nelem_dofs, ne);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         const int e1 = d_indices1[f*nface_dofs]/nelem_dofs;
         const int e2 = d_indices2[f*nface_dofs]/nelem_dofs;
         for (int j = 0; j < nface_dofs; j++)
         {
            const int jB1 = d_indices1[f*nface_dofs+j]%nelem_dofs;
            for (int i = 0; i < nface_dofs; i++)
            {
               const int iB1 = d_indices1[f*nface_dofs+i]%nelem_dofs;
               AtomicAdd(mat_ea(iB1,jB1,e1), mat_fea(i,j,0,f));
            }
         }
         if (e2 < NE)
         {
            for (int j = 0; j < nface_dofs; j++)
            {
               const int jB2 = d_indices2[f*nface_dofs+j]%nelem_dofs;
               for (int i = 0; i < nface_dofs; i++)
               {
                  const int iB2 = d_indices2[f*nface_dofs+i]%nelem_dofs;
                  AtomicAdd(mat_ea(iB2,jB2,e2), mat_fea(i,j,1,f));
               }
            }
         }
      });
   }
   else
   {
      auto d_indices = scatter_indices1.Read();
      auto mat_fea = Reshape(fea_data.Read(), nface_dofs, nface_dofs, nf);
      auto mat_ea = Reshape(ea_data.ReadWrite(), nelem_dofs, nelem_dofs, ne);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         const int e = d_indices[f*nface_dofs]/nelem_dofs;
         for (int j = 0; j < nface_dofs; j++)
         {
            const int jE = d_indices[f*nface_dofs+j]%nelem_dofs;
            for (int i = 0; i < nface_dofs; i++)
            {
               const int iE = d_indices[f*nface_dofs+i]%nelem_dofs;
               AtomicAdd(mat_ea(iE,jE,e), mat_fea(i,j,f));
            }
         }
      });
   }
}

void L2FaceRestriction::CheckFESpace()
{
#ifdef MFEM_USE_MPI

   // If the underlying finite element space is parallel, ensure the face
   // neighbor information is generated.
   if (const ParFiniteElementSpace *pfes
       = dynamic_cast<const ParFiniteElementSpace*>(&fes))
   {
      pfes->GetParMesh()->ExchangeFaceNbrData();
   }

#endif

#ifdef MFEM_DEBUG
   // If fespace == L2
   const FiniteElement *fe0 = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe0);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "L2FaceRestriction.");
   if (nf==0) { return; }
   const bool dof_reorder = (ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            fes.GetTraceElement(f, fes.GetMesh()->GetFaceGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
#endif
}

void L2FaceRestriction::ComputeScatterIndicesAndOffsets()
{
   Mesh &mesh = *fes.GetMesh();
   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter indices and offsets
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      MFEM_ASSERT(!face.IsShared(),
                  "Unexpected shared face in L2FaceRestriction.");
      if ( face.IsOfFaceType(type) )
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            if ( type==FaceType::Interior && face.IsInterior() )
            {
               PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
            }
            else if ( type==FaceType::Boundary && face.IsBoundary() )
            {
               SetBoundaryDofsScatterIndices2(face,f_ind);
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      gather_offsets[i] += gather_offsets[i - 1];
   }
}

void L2FaceRestriction::ComputeGatherIndices()
{
   Mesh &mesh = *fes.GetMesh();
   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      MFEM_ASSERT(!face.IsShared(),
                  "Unexpected shared face in L2FaceRestriction.");
      if ( face.IsOfFaceType(type) )
      {
         SetFaceDofsGatherIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued &&
              type==FaceType::Interior &&
              face.IsLocal())
         {
            PermuteAndSetFaceDofsGatherIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");

   // Reset offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

void L2FaceRestriction::SetFaceDofsScatterIndices1(
   const Mesh::FaceInformation &face,
   const int face_index)
{
   MFEM_ASSERT(!(face.IsNonconformingCoarse()),
               "This method should not be used on nonconforming coarse faces.");
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int face_id1 = face.element[0].local_face_id;
   const int elem_index = face.element[0].index;
   fes.GetFE(0)->GetFaceMap(face_id1, face_map);

   for (int face_dof_elem1 = 0; face_dof_elem1 < face_dofs; ++face_dof_elem1)
   {
      const int volume_dof_elem1 = face_map[face_dof_elem1];
      const int global_dof_elem1 = elem_map[elem_index*elem_dofs + volume_dof_elem1];
      const int restriction_dof_elem1 = face_dofs*face_index + face_dof_elem1;
      scatter_indices1[restriction_dof_elem1] = global_dof_elem1;
      ++gather_offsets[global_dof_elem1 + 1];
   }
}

void L2FaceRestriction::PermuteAndSetFaceDofsScatterIndices2(
   const Mesh::FaceInformation &face,
   const int face_index)
{
   MFEM_ASSERT(face.IsLocal(),
               "This method should only be used on local faces.");
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int elem_index = face.element[1].index;
   const int face_id1 = face.element[0].local_face_id;
   const int face_id2 = face.element[1].local_face_id;
   const int orientation = face.element[1].orientation;
   const int dim = fes.GetMesh()->Dimension();
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   fes.GetFE(0)->GetFaceMap(face_id2, face_map);

   for (int face_dof_elem1 = 0; face_dof_elem1 < face_dofs; ++face_dof_elem1)
   {
      const int face_dof_elem2 = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d,
                                               face_dof_elem1);
      const int volume_dof_elem2 = face_map[face_dof_elem2];
      const int global_dof_elem2 = elem_map[elem_index*elem_dofs + volume_dof_elem2];
      const int restriction_dof_elem2 = face_dofs*face_index + face_dof_elem1;
      scatter_indices2[restriction_dof_elem2] = global_dof_elem2;
      ++gather_offsets[global_dof_elem2 + 1];
   }
}

void L2FaceRestriction::PermuteAndSetSharedFaceDofsScatterIndices2(
   const Mesh::FaceInformation &face,
   const int face_index)
{
#ifdef MFEM_USE_MPI
   MFEM_ASSERT(face.IsShared(),
               "This method should only be used on shared faces.");
   const int elem_index = face.element[1].index;
   const int face_id1 = face.element[0].local_face_id;
   const int face_id2 = face.element[1].local_face_id;
   const int orientation = face.element[1].orientation;
   const int dim = fes.GetMesh()->Dimension();
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   fes.GetFE(0)->GetFaceMap(face_id2, face_map);
   Array<int> face_nbr_dofs;
   const ParFiniteElementSpace &pfes =
      static_cast<const ParFiniteElementSpace&>(this->fes);
   pfes.GetFaceNbrElementVDofs(elem_index, face_nbr_dofs);

   for (int face_dof_elem1 = 0; face_dof_elem1 < face_dofs; ++face_dof_elem1)
   {
      const int face_dof_elem2 = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, face_dof_elem1);
      const int volume_dof_elem2 = face_map[face_dof_elem2];
      const int global_dof_elem2 = face_nbr_dofs[volume_dof_elem2];
      const int restriction_dof_elem2 = face_dofs*face_index + face_dof_elem1;
      // Trick to differentiate dof location inter/shared
      scatter_indices2[restriction_dof_elem2] = ndofs+global_dof_elem2;
   }
#endif
}

void L2FaceRestriction::SetBoundaryDofsScatterIndices2(
   const Mesh::FaceInformation &face,
   const int face_index)
{
   MFEM_ASSERT(face.IsBoundary(),
               "This method should only be used on boundary faces.");

   for (int d = 0; d < face_dofs; ++d)
   {
      const int restriction_dof_elem2 = face_dofs*face_index + d;
      scatter_indices2[restriction_dof_elem2] = -1;
   }
}

void L2FaceRestriction::SetFaceDofsGatherIndices1(
   const Mesh::FaceInformation &face,
   const int face_index)
{
   MFEM_ASSERT(!(face.IsNonconformingCoarse()),
               "This method should not be used on nonconforming coarse faces.");
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int face_id1 = face.element[0].local_face_id;
   const int elem_index = face.element[0].index;
   fes.GetFE(0)->GetFaceMap(face_id1, face_map);

   for (int face_dof_elem1 = 0; face_dof_elem1 < face_dofs; ++face_dof_elem1)
   {
      const int volume_dof_elem1 = face_map[face_dof_elem1];
      const int global_dof_elem1 = elem_map[elem_index*elem_dofs + volume_dof_elem1];
      const int restriction_dof_elem1 = face_dofs*face_index + face_dof_elem1;
      // We don't shift restriction_dof_elem1 to express that it's elem1 of the face
      gather_indices[gather_offsets[global_dof_elem1]++] = restriction_dof_elem1;
   }
}

void L2FaceRestriction::PermuteAndSetFaceDofsGatherIndices2(
   const Mesh::FaceInformation &face,
   const int face_index)
{
   MFEM_ASSERT(face.IsLocal(),
               "This method should only be used on local faces.");
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int elem_index = face.element[1].index;
   const int face_id1 = face.element[0].local_face_id;
   const int face_id2 = face.element[1].local_face_id;
   const int orientation = face.element[1].orientation;
   const int dim = fes.GetMesh()->Dimension();
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   fes.GetFE(0)->GetFaceMap(face_id2, face_map);

   for (int face_dof_elem1 = 0; face_dof_elem1 < face_dofs; ++face_dof_elem1)
   {
      const int face_dof_elem2 = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d,
                                               face_dof_elem1);
      const int volume_dof_elem2 = face_map[face_dof_elem2];
      const int global_dof_elem2 = elem_map[elem_index*elem_dofs + volume_dof_elem2];
      const int restriction_dof_elem2 = face_dofs*face_index + face_dof_elem1;
      // We shift restriction_dof_elem2 to express that it's elem2 of the face
      gather_indices[gather_offsets[global_dof_elem2]++] = nfdofs +
                                                           restriction_dof_elem2;
   }
}

void L2FaceRestriction::NormalDerivativeMult(const Vector &x, Vector &y) const
{
   EnsureNormalDerivativeRestriction();
   normal_deriv_restr->Mult(x, y);
}

void L2FaceRestriction::NormalDerivativeAddMultTranspose(const Vector &x,
                                                         Vector &y) const
{
   EnsureNormalDerivativeRestriction();
   normal_deriv_restr->AddMultTranspose(x, y);
}

void L2FaceRestriction::EnsureNormalDerivativeRestriction() const
{
   if (!normal_deriv_restr)
   {
      normal_deriv_restr.reset(
         new L2NormalDerivativeFaceRestriction(fes, ordering, type));
   }
}

InterpolationManager::InterpolationManager(const FiniteElementSpace &fes,
                                           ElementDofOrdering ordering,
                                           FaceType type)
   : fes(fes),
     ordering(ordering),
     interp_config( fes.GetNFbyType(type) ),
     nc_cpt(0)
{ }

void InterpolationManager::RegisterFaceConformingInterpolation(
   const Mesh::FaceInformation &face,
   int face_index)
{
   interp_config[face_index] = InterpConfig();
}

void InterpolationManager::RegisterFaceCoarseToFineInterpolation(
   const Mesh::FaceInformation &face,
   int face_index)
{
   MFEM_ASSERT(!face.IsConforming(),
               "Registering face as nonconforming even though it is not.");
   const DenseMatrix* ptMat = face.point_matrix;
   // In the case of nonconforming slave shared face the master face is elem1.
   const int master_side =
      face.element[0].conformity == Mesh::ElementConformity::Superset ? 0 : 1;
   const int face_key = (master_side == 0 ? 1000 : 0) +
                        face.element[0].local_face_id +
                        6*face.element[1].local_face_id +
                        36*face.element[1].orientation ;
   // Unfortunately we can't trust unicity of the ptMat to identify the transformation.
   Key key(ptMat, face_key);
   auto itr = interp_map.find(key);
   if ( itr == interp_map.end() )
   {
      const DenseMatrix* interpolator =
         GetCoarseToFineInterpolation(face,ptMat);
      interp_map[key] = {nc_cpt, interpolator};
      interp_config[face_index] = {master_side, nc_cpt};
      nc_cpt++;
   }
   else
   {
      interp_config[face_index] = {master_side, itr->second.first};
   }
}

const DenseMatrix* InterpolationManager::GetCoarseToFineInterpolation(
   const Mesh::FaceInformation &face,
   const DenseMatrix* ptMat)
{
   MFEM_VERIFY(ordering == ElementDofOrdering::LEXICOGRAPHIC,
               "The following interpolation operator is only implemented for"
               "lexicographic ordering.");
   MFEM_VERIFY(!face.IsConforming(),
               "This method should not be called on conforming faces.")
   const int face_id1 = face.element[0].local_face_id;
   const int face_id2 = face.element[1].local_face_id;

   const bool is_ghost_slave =
      face.element[0].conformity == Mesh::ElementConformity::Superset;
   const int master_face_id = is_ghost_slave ? face_id1 : face_id2;

   // Computation of the interpolation matrix from master
   // (coarse) face to slave (fine) face.
   // Assumes all trace elements are the same.
   const FiniteElement *trace_fe =
      fes.GetTraceElement(0, fes.GetMesh()->GetFaceGeometry(0));
   const int face_dofs = trace_fe->GetDof();
   const TensorBasisElement* el =
      dynamic_cast<const TensorBasisElement*>(trace_fe);
   const auto dof_map = el->GetDofMap();
   DenseMatrix* interpolator = new DenseMatrix(face_dofs,face_dofs);
   Vector shape(face_dofs);

   IsoparametricTransformation isotr;
   isotr.SetIdentityTransformation(trace_fe->GetGeomType());
   isotr.SetPointMat(*ptMat);
   DenseMatrix& trans_pt_mat = isotr.GetPointMat();
   // PointMatrix needs to be flipped in 2D
   if ( trace_fe->GetGeomType()==Geometry::SEGMENT && !is_ghost_slave )
   {
      std::swap(trans_pt_mat(0,0),trans_pt_mat(0,1));
   }
   DenseMatrix native_interpolator(face_dofs,face_dofs);
   trace_fe->GetLocalInterpolation(isotr, native_interpolator);
   const int dim = trace_fe->GetDim()+1;
   const int dof1d = trace_fe->GetOrder()+1;
   const int orientation = face.element[1].orientation;
   for (int i = 0; i < face_dofs; i++)
   {
      const int ni = (dof_map.Size()==0) ? i : dof_map[i];
      int li = ToLexOrdering(dim, master_face_id, dof1d, i);
      if ( !is_ghost_slave )
      {
         // master side is elem 2, so we permute to order dofs as elem 1.
         li = PermuteFaceL2(dim, face_id2, face_id1,
                            orientation, dof1d, li);
      }
      for (int j = 0; j < face_dofs; j++)
      {
         int lj = ToLexOrdering(dim, master_face_id, dof1d, j);
         if ( !is_ghost_slave )
         {
            // master side is elem 2, so we permute to order dofs as elem 1.
            lj = PermuteFaceL2(dim, face_id2, face_id1,
                               orientation, dof1d, lj);
         }
         const int nj = (dof_map.Size()==0) ? j : dof_map[j];
         (*interpolator)(li,lj) = native_interpolator(ni,nj);
      }
   }
   return interpolator;
}

void InterpolationManager::LinearizeInterpolatorMapIntoVector()
{
   // Assumes all trace elements are the same.
   const FiniteElement *trace_fe =
      fes.GetTraceElement(0, fes.GetMesh()->GetFaceGeometry(0));
   const int face_dofs = trace_fe->GetDof();
   const int nc_size = static_cast<int>(interp_map.size());
   MFEM_VERIFY(nc_cpt==nc_size, "Unexpected number of interpolators.");
   interpolators.SetSize(face_dofs*face_dofs*nc_size);
   auto d_interp = Reshape(interpolators.HostWrite(),face_dofs,face_dofs,nc_size);
   for (auto val : interp_map)
   {
      const int idx = val.second.first;
      const DenseMatrix &interpolator = *val.second.second;
      for (int i = 0; i < face_dofs; i++)
      {
         for (int j = 0; j < face_dofs; j++)
         {
            d_interp(i,j,idx) = interpolator(i,j);
         }
      }
      delete val.second.second;
   }
   interp_map.clear();
}

void InterpolationManager::InitializeNCInterpConfig()
{
   // Count nonconforming faces
   int num_nc_faces = 0;
   for (int i = 0; i < interp_config.Size(); i++)
   {
      if ( interp_config[i].is_non_conforming )
      {
         num_nc_faces++;
      }
   }
   // Set nc_interp_config
   nc_interp_config.SetSize(num_nc_faces);
   int nc_index = 0;
   for (int i = 0; i < interp_config.Size(); i++)
   {
      auto & config = interp_config[i];
      if ( config.is_non_conforming )
      {
         nc_interp_config[nc_index] = NCInterpConfig(i, config);
         nc_index++;
      }
   }
}

NCL2FaceRestriction::NCL2FaceRestriction(const FiniteElementSpace &fes,
                                         const ElementDofOrdering f_ordering,
                                         const FaceType type,
                                         const L2FaceValues m,
                                         bool build)
   : L2FaceRestriction(fes, f_ordering, type, m, false),
     interpolations(fes, f_ordering, type)
{
   if (!build) { return; }
   x_interp.UseDevice(true);

   CheckFESpace();

   ComputeScatterIndicesAndOffsets();

   ComputeGatherIndices();
}

NCL2FaceRestriction::NCL2FaceRestriction(const FiniteElementSpace &fes,
                                         const ElementDofOrdering f_ordering,
                                         const FaceType type,
                                         const L2FaceValues m)
   : NCL2FaceRestriction(fes, f_ordering, type, m, true)
{ }

void NCL2FaceRestriction::DoubleValuedNonconformingMult(
   const Vector& x, Vector& y) const
{
   DoubleValuedConformingMult(x, y);
   DoubleValuedNonconformingInterpolation(y);
}

void NCL2FaceRestriction::DoubleValuedNonconformingInterpolation(
   Vector& y) const
{
   if (nf == 0) { return; }
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   auto d_y = Reshape(y.ReadWrite(), nface_dofs, vd, 2, nf);
   auto &nc_interp_config = interpolations.GetNCFaceInterpConfig();
   const int num_nc_faces = nc_interp_config.Size();
   if ( num_nc_faces == 0 ) { return; }
   auto interp_config_ptr = nc_interp_config.Read();
   const int nc_size = interpolations.GetNumInterpolators();
   auto d_interp = Reshape(interpolations.GetInterpolators().Read(),
                           nface_dofs, nface_dofs, nc_size);
   static constexpr int max_nd = 16*16;
   MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
   mfem::forall_2D(num_nc_faces, nface_dofs, 1, [=] MFEM_HOST_DEVICE (int nc_face)
   {
      MFEM_SHARED real_t dof_values[max_nd];
      const NCInterpConfig conf = interp_config_ptr[nc_face];
      if ( conf.is_non_conforming )
      {
         const int master_side = conf.master_side;
         const int interp_index = conf.index;
         const int face = conf.face_index;
         for (int c = 0; c < vd; ++c)
         {
            MFEM_FOREACH_THREAD(dof,x,nface_dofs)
            {
               dof_values[dof] = d_y(dof, c, master_side, face);
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
            {
               real_t res = 0.0;
               for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
               {
                  res += d_interp(dof_out, dof_in, interp_index)*dof_values[dof_in];
               }
               d_y(dof_out, c, master_side, face) = res;
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

void NCL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   if ( type==FaceType::Interior && m==L2FaceValues::DoubleValued )
   {
      DoubleValuedNonconformingMult(x, y);
   }
   else if ( type==FaceType::Boundary && m==L2FaceValues::DoubleValued )
   {
      DoubleValuedConformingMult(x, y);
   }
   else // Single valued (assumes no nonconforming master on elem1)
   {
      SingleValuedConformingMult(x, y);
   }
}

void NCL2FaceRestriction::SingleValuedNonconformingTransposeInterpolation(
   const Vector& x) const
{
   MFEM_ASSERT(
      m == L2FaceValues::SingleValued,
      "This method should be called when m == L2FaceValues::SingleValued.");
   if (x_interp.Size()==0)
   {
      x_interp.SetSize(x.Size());
   }
   x_interp = x;
   SingleValuedNonconformingTransposeInterpolationInPlace(x_interp);
}


void NCL2FaceRestriction::SingleValuedNonconformingTransposeInterpolationInPlace(
   Vector& x) const
{
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   // Interpolation
   auto d_x = Reshape(x_interp.ReadWrite(), nface_dofs, vd, nf);
   auto &nc_interp_config = interpolations.GetNCFaceInterpConfig();
   const int num_nc_faces = nc_interp_config.Size();
   if ( num_nc_faces == 0 ) { return; }
   auto interp_config_ptr = nc_interp_config.Read();
   auto interpolators = interpolations.GetInterpolators().Read();
   const int nc_size = interpolations.GetNumInterpolators();
   auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
   static constexpr int max_nd = 16*16;
   MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
   mfem::forall_2D(num_nc_faces, nface_dofs, 1, [=] MFEM_HOST_DEVICE (int nc_face)
   {
      MFEM_SHARED real_t dof_values[max_nd];
      const NCInterpConfig conf = interp_config_ptr[nc_face];
      const int master_side = conf.master_side;
      const int interp_index = conf.index;
      const int face = conf.face_index;
      if ( conf.is_non_conforming && master_side==0 )
      {
         // Interpolation from fine to coarse
         for (int c = 0; c < vd; ++c)
         {
            MFEM_FOREACH_THREAD(dof,x,nface_dofs)
            {
               dof_values[dof] = d_x(dof, c, face);
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
            {
               real_t res = 0.0;
               for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
               {
                  res += d_interp(dof_in, dof_out, interp_index)*dof_values[dof_in];
               }
               d_x(dof_out, c, face) = res;
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

void NCL2FaceRestriction::DoubleValuedNonconformingTransposeInterpolation(
   const Vector& x) const
{
   MFEM_ASSERT(
      m == L2FaceValues::DoubleValued,
      "This method should be called when m == L2FaceValues::DoubleValued.");
   if (x_interp.Size()==0)
   {
      x_interp.SetSize(x.Size());
   }
   x_interp = x;
   DoubleValuedNonconformingTransposeInterpolationInPlace(x_interp);
}

void NCL2FaceRestriction::DoubleValuedNonconformingTransposeInterpolationInPlace(
   Vector& x) const
{
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   // Interpolation
   auto d_x = Reshape(x.ReadWrite(), nface_dofs, vd, 2, nf);
   auto &nc_interp_config = interpolations.GetNCFaceInterpConfig();
   const int num_nc_faces = nc_interp_config.Size();
   if ( num_nc_faces == 0 ) { return; }
   auto interp_config_ptr = nc_interp_config.Read();
   auto interpolators = interpolations.GetInterpolators().Read();
   const int nc_size = interpolations.GetNumInterpolators();
   auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
   static constexpr int max_nd = 16*16;
   MFEM_VERIFY(nface_dofs<=max_nd, "Too many degrees of freedom.");
   mfem::forall_2D(num_nc_faces, nface_dofs, 1, [=] MFEM_HOST_DEVICE (int nc_face)
   {
      MFEM_SHARED real_t dof_values[max_nd];
      const NCInterpConfig conf = interp_config_ptr[nc_face];
      const int master_side = conf.master_side;
      const int interp_index = conf.index;
      const int face = conf.face_index;
      if ( conf.is_non_conforming )
      {
         // Interpolation from fine to coarse
         for (int c = 0; c < vd; ++c)
         {
            MFEM_FOREACH_THREAD(dof,x,nface_dofs)
            {
               dof_values[dof] = d_x(dof, c, master_side, face);
            }
            MFEM_SYNC_THREAD;
            MFEM_FOREACH_THREAD(dof_out,x,nface_dofs)
            {
               real_t res = 0.0;
               for (int dof_in = 0; dof_in<nface_dofs; dof_in++)
               {
                  res += d_interp(dof_in, dof_out, interp_index)*dof_values[dof_in];
               }
               d_x(dof_out, c, master_side, face) = res;
            }
            MFEM_SYNC_THREAD;
         }
      }
   });
}

void NCL2FaceRestriction::AddMultTranspose(const Vector& x, Vector& y,
                                           const real_t a) const
{
   MFEM_VERIFY(a == 1.0, "General coefficient case is not yet supported!");
   if (nf==0) { return; }
   if (type==FaceType::Interior)
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedNonconformingTransposeInterpolation(x);
         DoubleValuedConformingAddMultTranspose(x_interp, y);
      }
      else if ( m==L2FaceValues::SingleValued )
      {
         SingleValuedNonconformingTransposeInterpolation(x);
         SingleValuedConformingAddMultTranspose(x_interp, y);
      }
   }
   else
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedConformingAddMultTranspose(x, y);
      }
      else if ( m==L2FaceValues::SingleValued )
      {
         SingleValuedConformingAddMultTranspose(x, y);
      }
   }
}

void NCL2FaceRestriction::AddMultTransposeInPlace(Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   if (type==FaceType::Interior)
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedNonconformingTransposeInterpolationInPlace(x);
         DoubleValuedConformingAddMultTranspose(x, y);
      }
      else if ( m==L2FaceValues::SingleValued )
      {
         SingleValuedNonconformingTransposeInterpolationInPlace(x);
         SingleValuedConformingAddMultTranspose(x, y);
      }
   }
   else
   {
      if ( m==L2FaceValues::DoubleValued )
      {
         DoubleValuedConformingAddMultTranspose(x, y);
      }
      else if ( m==L2FaceValues::SingleValued )
      {
         SingleValuedConformingAddMultTranspose(x, y);
      }
   }
}

void NCL2FaceRestriction::FillI(SparseMatrix &mat,
                                const bool keep_nbr_block) const
{
   const int nface_dofs = face_dofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int iE1 = d_indices1[fdof];
      const int iE2 = d_indices2[fdof];
      AddNnz(iE1,I,nface_dofs);
      AddNnz(iE2,I,nface_dofs);
   });
}

void NCL2FaceRestriction::FillJAndData(const Vector &fea_data,
                                       SparseMatrix &mat,
                                       const bool keep_nbr_block) const
{
   const int nface_dofs = face_dofs;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   auto mat_fea = Reshape(fea_data.Read(), nface_dofs, nface_dofs, 2, nf);
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
   auto interpolators = interpolations.GetInterpolators().Read();
   const int nc_size = interpolations.GetNumInterpolators();
   auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
   mfem::forall(nf*nface_dofs, [=] MFEM_HOST_DEVICE (int fdof)
   {
      const int f  = fdof/nface_dofs;
      const InterpConfig conf = interp_config_ptr[f];
      const int master_side = conf.master_side;
      const int interp_index = conf.index;
      const int iF = fdof%nface_dofs;
      const int iE1 = d_indices1[f*nface_dofs+iF];
      const int iE2 = d_indices2[f*nface_dofs+iF];
      const int offset1 = AddNnz(iE1,I,nface_dofs);
      const int offset2 = AddNnz(iE2,I,nface_dofs);
      for (int jF = 0; jF < nface_dofs; jF++)
      {
         const int jE1 = d_indices1[f*nface_dofs+jF];
         const int jE2 = d_indices2[f*nface_dofs+jF];
         J[offset2+jF] = jE1;
         J[offset1+jF] = jE2;
         real_t val1 = 0.0;
         real_t val2 = 0.0;
         if ( conf.is_non_conforming && master_side==0 )
         {
            for (int kF = 0; kF < nface_dofs; kF++)
            {
               val1 += mat_fea(kF,iF,0,f) * d_interp(kF, jF, interp_index);
               val2 += d_interp(kF, iF, interp_index) * mat_fea(jF,kF,1,f);
            }
         }
         else if ( conf.is_non_conforming && master_side==1 )
         {
            for (int kF = 0; kF < nface_dofs; kF++)
            {
               val1 += d_interp(kF, iF, interp_index) * mat_fea(jF,kF,0,f);
               val2 += mat_fea(kF,iF,1,f) * d_interp(kF, jF, interp_index);
            }
         }
         else
         {
            val1 = mat_fea(jF,iF,0,f);
            val2 = mat_fea(jF,iF,1,f);
         }
         Data[offset2+jF] = val1;
         Data[offset1+jF] = val2;
      }
   });
}

void NCL2FaceRestriction::AddFaceMatricesToElementMatrices(
   const Vector &fea_data,
   Vector &ea_data)
const
{
   const int nface_dofs = face_dofs;
   const int nelem_dofs = elem_dofs;
   const int NE = ne;
   if (m==L2FaceValues::DoubleValued)
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto mat_fea = Reshape(fea_data.Read(), nface_dofs, nface_dofs, 2, nf);
      auto mat_ea = Reshape(ea_data.ReadWrite(), nelem_dofs, nelem_dofs, ne);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         const InterpConfig conf = interp_config_ptr[f];
         const int master_side = conf.master_side;
         const int interp_index = conf.index;
         const int e1 = d_indices1[f*nface_dofs]/nelem_dofs;
         const int e2 = d_indices2[f*nface_dofs]/nelem_dofs;
         for (int j = 0; j < nface_dofs; j++)
         {
            const int jB1 = d_indices1[f*nface_dofs+j]%nelem_dofs;
            for (int i = 0; i < nface_dofs; i++)
            {
               const int iB1 = d_indices1[f*nface_dofs+i]%nelem_dofs;
               real_t val = 0.0;
               if ( conf.is_non_conforming && master_side==0 )
               {
                  for (int k = 0; k < nface_dofs; k++)
                  {
                     for (int l = 0; l < nface_dofs; l++)
                     {
                        val += d_interp(l, j, interp_index)
                               * mat_fea(k,l,0,f)
                               * d_interp(k, i, interp_index);
                     }
                  }
               }
               else
               {
                  val = mat_fea(i,j,0,f);
               }
               AtomicAdd(mat_ea(iB1,jB1,e1), val);
            }
         }
         if (e2 < NE)
         {
            for (int j = 0; j < nface_dofs; j++)
            {
               const int jB2 = d_indices2[f*nface_dofs+j]%nelem_dofs;
               for (int i = 0; i < nface_dofs; i++)
               {
                  const int iB2 = d_indices2[f*nface_dofs+i]%nelem_dofs;
                  real_t val = 0.0;
                  if ( conf.is_non_conforming && master_side==1 )
                  {
                     for (int k = 0; k < nface_dofs; k++)
                     {
                        for (int l = 0; l < nface_dofs; l++)
                        {
                           val += d_interp(l, j, interp_index)
                                  * mat_fea(k,l,1,f)
                                  * d_interp(k, i, interp_index);
                        }
                     }
                  }
                  else
                  {
                     val = mat_fea(i,j,1,f);
                  }
                  AtomicAdd(mat_ea(iB2,jB2,e2), val);
               }
            }
         }
      });
   }
   else
   {
      auto d_indices = scatter_indices1.Read();
      auto mat_fea = Reshape(fea_data.Read(), nface_dofs, nface_dofs, nf);
      auto mat_ea = Reshape(ea_data.ReadWrite(), nelem_dofs, nelem_dofs, ne);
      auto interp_config_ptr = interpolations.GetFaceInterpConfig().Read();
      auto interpolators = interpolations.GetInterpolators().Read();
      const int nc_size = interpolations.GetNumInterpolators();
      auto d_interp = Reshape(interpolators, nface_dofs, nface_dofs, nc_size);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         const InterpConfig conf = interp_config_ptr[f];
         const int master_side = conf.master_side;
         const int interp_index = conf.index;
         const int e = d_indices[f*nface_dofs]/nelem_dofs;
         for (int j = 0; j < nface_dofs; j++)
         {
            const int jE = d_indices[f*nface_dofs+j]%nelem_dofs;
            for (int i = 0; i < nface_dofs; i++)
            {
               const int iE = d_indices[f*nface_dofs+i]%nelem_dofs;
               real_t val = 0.0;
               if ( conf.is_non_conforming && master_side==0 )
               {
                  for (int k = 0; k < nface_dofs; k++)
                  {
                     for (int l = 0; l < nface_dofs; l++)
                     {
                        val += d_interp(l, j, interp_index)
                               * mat_fea(k,l,f)
                               * d_interp(k, i, interp_index);
                     }
                  }
               }
               else
               {
                  val = mat_fea(i,j,f);
               }
               AtomicAdd(mat_ea(iE,jE,e), val);
            }
         }
      });
   }
}

int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index)
{
   switch (dim)
   {
      case 1:
         return 0;
      case 2:
         return internal::ToLexOrdering2D(face_id, size1d, index);
      case 3:
         return internal::ToLexOrdering3D(face_id, size1d, index%size1d, index/size1d);
      default:
         MFEM_ABORT("Unsupported dimension.");
         return 0;
   }
}

void NCL2FaceRestriction::ComputeScatterIndicesAndOffsets()
{
   Mesh &mesh = *fes.GetMesh();

   // Initialization of the offsets
   for (int i = 0; i <= ndofs; ++i)
   {
      gather_offsets[i] = 0;
   }

   // Computation of scatter and offsets indices
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if ( type==FaceType::Interior && face.IsInterior() )
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
         }
         if ( face.IsConforming() )
         {
            interpolations.RegisterFaceConformingInterpolation(face,f_ind);
         }
         else // Non-conforming face
         {
            interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
         }
         f_ind++;
      }
      else if ( type==FaceType::Boundary && face.IsBoundary() )
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            SetBoundaryDofsScatterIndices2(face,f_ind);
         }
         interpolations.RegisterFaceConformingInterpolation(face,f_ind);
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );

   // Summation of the offsets
   for (int i = 1; i <= ndofs; ++i)
   {
      gather_offsets[i] += gather_offsets[i - 1];
   }

   // Transform the interpolation matrix map into a contiguous memory structure.
   interpolations.LinearizeInterpolatorMapIntoVector();
   interpolations.InitializeNCInterpConfig();
}

void NCL2FaceRestriction::ComputeGatherIndices()
{
   Mesh &mesh = *fes.GetMesh();
   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      MFEM_ASSERT(!face.IsShared(),
                  "Unexpected shared face in NCL2FaceRestriction.");
      if ( face.IsNonconformingCoarse() )
      {
         // We skip nonconforming coarse faces as they are treated
         // by the corresponding nonconforming fine faces.
         continue;
      }
      else if ( face.IsOfFaceType(type) )
      {
         SetFaceDofsGatherIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued &&
              type==FaceType::Interior &&
              face.IsInterior() )
         {
            PermuteAndSetFaceDofsGatherIndices2(face,f_ind);
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );

   // Switch back offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      gather_offsets[i] = gather_offsets[i - 1];
   }
   gather_offsets[0] = 0;
}

Vector GetLVectorFaceNbrData(
   const FiniteElementSpace &fes, const Vector &x, FaceType ftype)
{
#ifdef MFEM_USE_MPI
   if (ftype == FaceType::Interior)
   {
      if (auto *pfes = const_cast<ParFiniteElementSpace*>
                       (dynamic_cast<const ParFiniteElementSpace*>(&fes)))
      {
         if (auto *x_gf = const_cast<ParGridFunction*>
                          (dynamic_cast<const ParGridFunction*>(&x)))
         {
            Vector &gf_face_nbr = x_gf->FaceNbrData();
            if (gf_face_nbr.Size() == 0) { x_gf->ExchangeFaceNbrData(); }
            gf_face_nbr.Read();
            return Vector(gf_face_nbr, 0, gf_face_nbr.Size());
         }
         else
         {
            ParGridFunction gf(pfes, const_cast<Vector&>(x));
            gf.ExchangeFaceNbrData();
            return std::move(gf.FaceNbrData());
         }
      }
   }
#endif
   return Vector();
}

} // namespace mfem
