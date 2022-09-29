// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
#include "gridfunc.hpp"
#include "fespace.hpp"
#include "../general/forall.hpp"
#include <climits>

#ifdef MFEM_USE_MPI

#include "pfespace.hpp"

#endif

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
   MFEM_FORALL(i, dof*ne,
   {
      const int gid = d_gather_map[i];
      const bool plus = gid >= 0;
      const int j = plus ? gid : -1-gid;
      for (int c = 0; c < vd; ++c)
      {
         const double dof_value = d_x(t?c:j, t?j:c);
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

   MFEM_FORALL(i, dof*ne,
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
void ElementRestriction::AddMultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(ADD ? y.ReadWrite() : y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dof_value = 0;
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
   AddMultTranspose<ADD>(x, y);
}

void ElementRestriction::AddMultTranspose(const Vector& x, Vector& y) const
{
   constexpr bool ADD = true;
   AddMultTranspose<ADD>(x, y);
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
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dof_value = 0;
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
   MFEM_FORALL(i, ndofs,
   {
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dof_value = 0;
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
   MFEM_FORALL(i_L, vd*all_dofs+1,
   {
      I[i_L] = 0;
   });
   MFEM_FORALL(l_dof, ne*elt_dofs,
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
   MFEM_FORALL(l_dof, ne*elt_dofs,
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
               double val = 0.0;
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
   MFEM_FORALL(i, ndofs,
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
void L2ElementRestriction::AddMultTranspose(const Vector &x, Vector &y) const
{
   const int nd = ndof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(ADD ? y.ReadWrite() : y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
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
   AddMultTranspose<ADD>(x, y);
}

void L2ElementRestriction::AddMultTranspose(const Vector &x, Vector &y) const
{
   constexpr bool ADD = true;
   AddMultTranspose<ADD>(x, y);
}

void L2ElementRestriction::FillI(SparseMatrix &mat) const
{
   const int elem_dofs = ndof;
   const int vd = vdim;
   auto I = mat.WriteI();
   const int isize = mat.Height() + 1;
   const int interior_dofs = ne*elem_dofs*vd;
   MFEM_FORALL(dof, isize,
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
   MFEM_FORALL(iE, ne*elem_dofs*vd,
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

/** Return the face degrees of freedom returned in Lexicographic order.
    Note: Only for quad and hex */
void GetFaceDofs(const int dim, const int face_id,
                 const int dof1d, Array<int> &face_map)
{
   switch (dim)
   {
      case 1:
         switch (face_id)
         {
            case 0: // WEST
               face_map[0] = 0;
               break;
            case 1: // EAST
               face_map[0] = dof1d-1;
               break;
         }
         break;
      case 2:
         switch (face_id)
         {
            case 0: // SOUTH
               for (int i = 0; i < dof1d; ++i)
               {
                  face_map[i] = i;
               }
               break;
            case 1: // EAST
               for (int i = 0; i < dof1d; ++i)
               {
                  face_map[i] = dof1d-1 + i*dof1d;
               }
               break;
            case 2: // NORTH
               for (int i = 0; i < dof1d; ++i)
               {
                  face_map[i] = (dof1d-1)*dof1d + i;
               }
               break;
            case 3: // WEST
               for (int i = 0; i < dof1d; ++i)
               {
                  face_map[i] = i*dof1d;
               }
               break;
         }
         break;
      case 3:
         switch (face_id)
         {
            case 0: // BOTTOM
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     face_map[i+j*dof1d] = i + j*dof1d;
                  }
               }
               break;
            case 1: // SOUTH
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     face_map[i+j*dof1d] = i + j*dof1d*dof1d;
                  }
               }
               break;
            case 2: // EAST
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     face_map[i+j*dof1d] = dof1d-1 + i*dof1d + j*dof1d*dof1d;
                  }
               }
               break;
            case 3: // NORTH
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     face_map[i+j*dof1d] = (dof1d-1)*dof1d + i + j*dof1d*dof1d;
                  }
               }
               break;
            case 4: // WEST
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     face_map[i+j*dof1d] = i*dof1d + j*dof1d*dof1d;
                  }
               }
               break;
            case 5: // TOP
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     face_map[i+j*dof1d] = (dof1d-1)*dof1d*dof1d + i + j*dof1d;
                  }
               }
               break;
         }
         break;
   }
}

H1FaceRestriction::H1FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
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
   if (!build) { return; }
   if (nf==0) { return; }

   CheckFESpace(e_ordering);

   ComputeScatterIndicesAndOffsets(e_ordering, type);

   ComputeGatherIndices(e_ordering,type);
}

H1FaceRestriction::H1FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
                                     const FaceType type)
   : H1FaceRestriction(fes, e_ordering, type, true)
{ }

void H1FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_indices = scatter_indices.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nface_dofs, vd, nf);
   MFEM_FORALL(i, nfdofs,
   {
      const int idx = d_indices[i];
      const int dof = i % nface_dofs;
      const int face = i / nface_dofs;
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
      }
   });
}

void H1FaceRestriction::AddMultTranspose(const Vector& x, Vector& y) const
{
   if (nf==0) { return; }
   // Assumes all elements have the same number of dofs
   const int nface_dofs = face_dofs;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = gather_offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x.Read(), nface_dofs, vd, nf);
   auto d_y = Reshape(y.ReadWrite(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dof_value = 0;
         for (int j = offset; j < next_offset; ++j)
         {
            const int idx_j = d_indices[j];
            dof_value +=  d_x(idx_j % nface_dofs, c, idx_j / nface_dofs);
         }
         d_y(t?c:i,t?i:c) += dof_value;
      }
   });
}

void H1FaceRestriction::CheckFESpace(const ElementDofOrdering e_ordering)
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
   // If fespace == H1
   const FiniteElement *fe0 = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe0);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "H1FaceRestriction.");

   // Assuming all finite elements are using Gauss-Lobatto.
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
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
      const FiniteElement *fe = fes.GetFaceElement(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
   }
#endif
}

void H1FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
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
         SetFaceDofsScatterIndices(face, f_ind, ordering);
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

void H1FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
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
         SetFaceDofsGatherIndices(face, f_ind, ordering);
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

void H1FaceRestriction::SetFaceDofsScatterIndices(
   const Mesh::FaceInformation &face,
   const int face_index,
   const ElementDofOrdering ordering)
{
   MFEM_ASSERT(!(face.IsNonconformingCoarse()),
               "This method should not be used on nonconforming coarse faces.");
   MFEM_ASSERT(face.element[0].orientation==0,
               "FaceRestriction used on degenerated mesh.");

   const TensorBasisElement* el =
      dynamic_cast<const TensorBasisElement*>(fes.GetFE(0));
   const int *dof_map = el->GetDofMap().GetData();
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int face_id = face.element[0].local_face_id;
   const int dim = fes.GetMesh()->Dimension();
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_index = face.element[0].index;
   const bool dof_reorder = (ordering == ElementDofOrdering::LEXICOGRAPHIC);
   GetFaceDofs(dim, face_id, dof1d, face_map); // Only for quad and hex

   for (int face_dof = 0; face_dof < face_dofs; ++face_dof)
   {
      const int nat_volume_dof = face_map[face_dof];
      const int volume_dof = (!dof_reorder)?
                             nat_volume_dof:
                             dof_map[nat_volume_dof];
      const int global_dof = elem_map[elem_index*elem_dofs + volume_dof];
      const int restriction_dof = face_dofs*face_index + face_dof;
      scatter_indices[restriction_dof] = global_dof;
      ++gather_offsets[global_dof + 1];
   }
}

void H1FaceRestriction::SetFaceDofsGatherIndices(
   const Mesh::FaceInformation &face,
   const int face_index,
   const ElementDofOrdering ordering)
{
   MFEM_ASSERT(!(face.IsNonconformingCoarse()),
               "This method should not be used on nonconforming coarse faces.");

   const TensorBasisElement* el =
      dynamic_cast<const TensorBasisElement*>(fes.GetFE(0));
   const int *dof_map = el->GetDofMap().GetData();
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elem_map = e2dTable.GetJ();
   const int face_id = face.element[0].local_face_id;
   const int dim = fes.GetMesh()->Dimension();
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_index = face.element[0].index;
   const bool dof_reorder = (ordering == ElementDofOrdering::LEXICOGRAPHIC);
   GetFaceDofs(dim, face_id, dof1d, face_map); // Only for quad and hex

   for (int face_dof = 0; face_dof < face_dofs; ++face_dof)
   {
      const int nat_volume_dof = face_map[face_dof];
      const int volume_dof = (!dof_reorder)?nat_volume_dof:dof_map[nat_volume_dof];
      const int global_dof = elem_map[elem_index*elem_dofs + volume_dof];
      const int restriction_dof = face_dofs*face_index + face_dof;
      gather_indices[gather_offsets[global_dof]++] = restriction_dof;
   }
}

static int ToLexOrdering2D(const int face_id, const int size1d, const int i)
{
   if (face_id==2 || face_id==3)
   {
      return size1d-1-i;
   }
   else
   {
      return i;
   }
}

static int PermuteFace2D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int new_index;
   // Convert from lex ordering
   if (face_id1==2 || face_id1==3)
   {
      new_index = size1d-1-index;
   }
   else
   {
      new_index = index;
   }
   // Permute based on face orientations
   if (orientation==1)
   {
      new_index = size1d-1-new_index;
   }
   return ToLexOrdering2D(face_id2, size1d, new_index);
}

static int ToLexOrdering3D(const int face_id, const int size1d, const int i,
                           const int j)
{
   if (face_id==2 || face_id==1 || face_id==5)
   {
      return i + j*size1d;
   }
   else if (face_id==3 || face_id==4)
   {
      return (size1d-1-i) + j*size1d;
   }
   else // face_id==0
   {
      return i + (size1d-1-j)*size1d;
   }
}

static int PermuteFace3D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int i=0, j=0, new_i=0, new_j=0;
   i = index%size1d;
   j = index/size1d;
   // Convert from lex ordering
   if (face_id1==3 || face_id1==4)
   {
      i = size1d-1-i;
   }
   else if (face_id1==0)
   {
      j = size1d-1-j;
   }
   // Permute based on face orientations
   switch (orientation)
   {
      case 0:
         new_i = i;
         new_j = j;
         break;
      case 1:
         new_i = j;
         new_j = i;
         break;
      case 2:
         new_i = j;
         new_j = (size1d-1-i);
         break;
      case 3:
         new_i = (size1d-1-i);
         new_j = j;
         break;
      case 4:
         new_i = (size1d-1-i);
         new_j = (size1d-1-j);
         break;
      case 5:
         new_i = (size1d-1-j);
         new_j = (size1d-1-i);
         break;
      case 6:
         new_i = (size1d-1-j);
         new_j = i;
         break;
      case 7:
         new_i = i;
         new_j = (size1d-1-j);
         break;
   }
   return ToLexOrdering3D(face_id2, size1d, new_i, new_j);
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
         return PermuteFace2D(face_id1, face_id2, orientation, size1d, index);
      case 3:
         return PermuteFace3D(face_id1, face_id2, orientation, size1d, index);
      default:
         MFEM_ABORT("Unsupported dimension.");
         return 0;
   }
}

L2FaceRestriction::L2FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
                                     const FaceType type,
                                     const L2FaceValues m,
                                     bool build)
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     face_dofs(nf > 0 ?
               fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof()
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

   CheckFESpace(e_ordering);

   ComputeScatterIndicesAndOffsets(e_ordering,type);

   ComputeGatherIndices(e_ordering, type);
}

L2FaceRestriction::L2FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
                                     const FaceType type,
                                     const L2FaceValues m)
   : L2FaceRestriction(fes, e_ordering, type, m, true)
{ }

void L2FaceRestriction::SingleValuedConformingMult(const Vector& x,
                                                   Vector& y) const
{
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
   MFEM_FORALL(i, nfdofs,
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
   MFEM_FORALL(i, nfdofs,
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
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dof_value = 0;
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
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int next_offset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dof_value = 0;
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

void L2FaceRestriction::AddMultTranspose(const Vector& x, Vector& y) const
{
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
   MFEM_FORALL(fdof, nf*nface_dofs,
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
   MFEM_FORALL(fdof, nf*nface_dofs,
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
      MFEM_FORALL(f, nf,
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
      MFEM_FORALL(f, nf,
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

void L2FaceRestriction::CheckFESpace(const ElementDofOrdering e_ordering)
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
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            fes.GetTraceElement(f, fes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
#endif
}

void L2FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType face_type)
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
      if ( face.IsOfFaceType(face_type) )
      {
         SetFaceDofsScatterIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued )
         {
            if ( face_type==FaceType::Interior && face.IsInterior() )
            {
               PermuteAndSetFaceDofsScatterIndices2(face,f_ind);
            }
            else if ( face_type==FaceType::Boundary && face.IsBoundary() )
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

void L2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType face_type)
{
   Mesh &mesh = *fes.GetMesh();
   // Computation of gather_indices
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      MFEM_ASSERT(!face.IsShared(),
                  "Unexpected shared face in L2FaceRestriction.");
      if ( face.IsOfFaceType(face_type) )
      {
         SetFaceDofsGatherIndices1(face,f_ind);
         if ( m==L2FaceValues::DoubleValued &&
              face_type==FaceType::Interior &&
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
   const int dim = fes.GetMesh()->Dimension();
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_index = face.element[0].index;
   GetFaceDofs(dim, face_id1, dof1d, face_map); // Only for quad and hex

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
   GetFaceDofs(dim, face_id2, dof1d, face_map); // Only for quad and hex

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
   GetFaceDofs(dim, face_id2, dof1d, face_map); // Only for quad and hex
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
   const int dim = fes.GetMesh()->Dimension();
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_index = face.element[0].index;
   GetFaceDofs(dim, face_id1, dof1d, face_map); // Only for quad and hex

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
   GetFaceDofs(dim, face_id2, dof1d, face_map); // Only for quad and hex

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

InterpolationManager::InterpolationManager(const FiniteElementSpace &fes,
                                           ElementDofOrdering ordering,
                                           FaceType type)
   : fes(fes),
     ordering(ordering),
     interp_config(type==FaceType::Interior ? fes.GetNFbyType(type) : 0),
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
      fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0));
   const int face_dofs = trace_fe->GetDof();
   const int nc_size = interp_map.size();
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
                                         const ElementDofOrdering ordering,
                                         const FaceType type,
                                         const L2FaceValues m,
                                         bool build)
   : L2FaceRestriction(fes, ordering, type, m, false),
     interpolations(fes, ordering, type)
{
   if (!build) { return; }
   x_interp.UseDevice(true);

   CheckFESpace(ordering);

   ComputeScatterIndicesAndOffsets(ordering, type);

   ComputeGatherIndices(ordering, type);
}

NCL2FaceRestriction::NCL2FaceRestriction(const FiniteElementSpace &fes,
                                         const ElementDofOrdering ordering,
                                         const FaceType type,
                                         const L2FaceValues m)
   : NCL2FaceRestriction(fes, ordering, type, m, true)
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
   MFEM_FORALL_3D(nc_face, num_nc_faces, nface_dofs, 1, 1,
   {
      MFEM_SHARED double dof_values[max_nd];
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
               double res = 0.0;
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
   if (nf==0) { return; }
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
   MFEM_FORALL_3D(nc_face, num_nc_faces, nface_dofs, 1, 1,
   {
      MFEM_SHARED double dof_values[max_nd];
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
               double res = 0.0;
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
   MFEM_FORALL_3D(nc_face, num_nc_faces, nface_dofs, 1, 1,
   {
      MFEM_SHARED double dof_values[max_nd];
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
               double res = 0.0;
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

void NCL2FaceRestriction::AddMultTranspose(const Vector& x, Vector& y) const
{
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
   MFEM_ABORT("Not yet implemented.");
}

void NCL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                       SparseMatrix &mat,
                                       const bool keep_nbr_block) const
{
   MFEM_ABORT("Not yet implemented.");
}

void NCL2FaceRestriction::AddFaceMatricesToElementMatrices(
   const Vector &fea_data,
   Vector &ea_data)
const
{
   MFEM_ABORT("Not yet implemented.");
}

int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index)
{
   switch (dim)
   {
      case 1:
         return 0;
      case 2:
         return ToLexOrdering2D(face_id, size1d, index);
      case 3:
         return ToLexOrdering3D(face_id, size1d, index%size1d, index/size1d);
      default:
         MFEM_ABORT("Unsupported dimension.");
         return 0;
   }
}

void NCL2FaceRestriction::ComputeScatterIndicesAndOffsets(
   const ElementDofOrdering ordering,
   const FaceType type)
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
            if ( face.IsConforming() )
            {
               interpolations.RegisterFaceConformingInterpolation(face,f_ind);
            }
            else // Non-conforming face
            {
               interpolations.RegisterFaceCoarseToFineInterpolation(face,f_ind);
            }
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

void NCL2FaceRestriction::ComputeGatherIndices(
   const ElementDofOrdering ordering,
   const FaceType type)
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

} // namespace mfem
