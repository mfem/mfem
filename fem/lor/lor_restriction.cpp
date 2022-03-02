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

#include "lor.hpp"
#include "lor_restriction.hpp"
#include "../pbilinearform.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

const Array<int> &GetDofMap(const FiniteElement *fe)
{
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL, "!TensorBasisElement");
   const Array<int> &dof_map = tfe->GetDofMap();
   MFEM_VERIFY(dof_map.Size() > 0, "Invalid DOF map.");
   return dof_map;
}

int LORRestriction::GetNRefinedElements(const FiniteElementSpace &fes)
{
   int ref = fes.GetMaxElementOrder();
   int dim = fes.GetMesh()->Dimension();
   return pow(ref, dim);
}

FiniteElementCollection *LORRestriction::MakeLowOrderFEC(
   const FiniteElementSpace &fes)
{
   return fes.FEColl()->Clone(1);
}

LORRestriction::LORRestriction(const FiniteElementSpace &fes_ho)
   : fes_ho(fes_ho),
     fec_lo(MakeLowOrderFEC(fes_ho)),
     geom(fes_ho.GetMesh()->GetElementGeometry(0)),
     order(fes_ho.GetMaxElementOrder()),
     ne_ref(GetNRefinedElements(fes_ho)),
     ne(fes_ho.GetNE()*ne_ref),
     vdim(fes_ho.GetVDim()),
     byvdim(fes_ho.GetOrdering() == Ordering::byVDIM),
     ndofs(fes_ho.GetNDofs()),
     lo_dof_per_el(fec_lo->GetFE(geom, 1)->GetDof()),
     offsets(ndofs+1),
     indices(ne*lo_dof_per_el),
     gatherMap(ne*lo_dof_per_el)
{
   Setup();
}

void LORRestriction::Setup()
{
   const Array<int> &fe_dof_map = GetDofMap(fec_lo->GetFE(geom, 1));
   const Array<int> &fe_dof_map_ho = GetDofMap(fes_ho.GetFE(0));

   // Form local_dof_map, which maps from the vertex of a LO element to the
   // vertex in the macro element
   RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, order);
   Array<int> local_dof_map(lo_dof_per_el*ne_ref);
   for (int ie_lo = 0; ie_lo < ne_ref; ++ie_lo)
   {
      for (int i = 0; i < lo_dof_per_el; ++i)
      {
         int cart_idx = RG.RefGeoms[i + lo_dof_per_el*ie_lo]; // local Cartesian index
         local_dof_map[i + lo_dof_per_el*ie_lo] = fe_dof_map_ho[cart_idx];
      }
   }

   const Table& e2dTable_ho = fes_ho.GetElementToDofTable();

   auto d_offsets = offsets.Write();
   const int NDOFS = ndofs;
   MFEM_FORALL(i, NDOFS+1, d_offsets[i] = 0;);

   const Memory<int> &J = e2dTable_ho.GetJMemory();
   const MemoryClass mc = Device::GetDeviceMemoryClass();
   const int *d_element_map = J.Read(mc, J.Capacity());
   const int *d_local_dof_map = local_dof_map.Read();
   const int DOF = lo_dof_per_el;
   const int DOF_ho = fes_ho.GetFE(0)->GetDof();
   const int NE = ne;
   const int NR_REF = ne_ref;

   MFEM_FORALL(e, NE,
   {
      const int e_ho = e/NR_REF;
      const int i_ref = e%NR_REF;
      for (int d = 0; d < DOF; ++d)
      {
         const int d_ho = d_local_dof_map[d + i_ref*DOF];
         const int sgid = d_element_map[DOF_ho*e_ho + d_ho];  // signed
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         AtomicAdd(d_offsets[gid+1], 1);
      }
   });

   // TODO: on device
   // Aggregate to find offsets for each global dof
   offsets.HostReadWrite();
   for (int i = 1; i <= ndofs; ++i) { offsets[i] += offsets[i - 1]; }

   // For each global dof, fill in all local nodes that point to it
   auto d_gather = gatherMap.Write();
   auto d_indices = indices.Write();
   auto drw_offsets = offsets.ReadWrite();
   const auto dof_map_mem = fe_dof_map.GetMemory();
   const auto d_dof_map = fe_dof_map.GetMemory().Read(mc,dof_map_mem.Capacity());

   MFEM_FORALL(e, NE,
   {
      const int e_ho = e/NR_REF;
      const int i_ref = e%NR_REF;
      for (int d = 0; d < DOF; ++d)
      {
         int d_ho = d_local_dof_map[d + i_ref*DOF];
         const int sdid = d_dof_map[d];  // signed
         const int sgid = d_element_map[DOF_ho*e_ho + d_ho];  // signed
         const int gid = (sgid >= 0) ? sgid : -1-sgid;
         const int lid = DOF*e + d;
         const bool plus = (sgid >= 0 && sdid >= 0) || (sgid < 0 && sdid < 0);
         d_gather[lid] = plus ? gid : -1-gid;
         d_indices[AtomicAdd(drw_offsets[gid], 1)] = plus ? lid : -1-lid;
      }
   });

   // TODO: on device
   offsets.HostReadWrite();
   for (int i = ndofs; i > 0; --i) { offsets[i] = offsets[i - 1]; }
   offsets[0] = 0;
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

int LORRestriction::FillI(SparseMatrix &mat) const
{
   static constexpr int Max = 16;
   const int all_dofs = ndofs;
   const int vd = vdim;
   const int elt_dofs = lo_dof_per_el;
   auto I = mat.WriteI();
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gatherMap = gatherMap.Read();
   MFEM_FORALL(i_L, vd*all_dofs+1, { I[i_L] = 0; });
   MFEM_FORALL(e, ne,
   {
      for (int i = 0; i < elt_dofs; i++)
      {
         int i_elts[Max];
         const int i_E = e*elt_dofs + i;
         const int i_L = d_gatherMap[i_E];
         const int i_offset = d_offsets[i_L];
         const int i_nextOffset = d_offsets[i_L+1];
         const int i_nbElts = i_nextOffset - i_offset;
         for (int e_i = 0; e_i < i_nbElts; ++e_i)
         {
            const int i_E = d_indices[i_offset+e_i];
            i_elts[e_i] = i_E/elt_dofs;
         }
         for (int j = 0; j < elt_dofs; j++)
         {
            const int j_E = e*elt_dofs + j;
            const int j_L = d_gatherMap[j_E];
            const int j_offset = d_offsets[j_L];
            const int j_nextOffset = d_offsets[j_L+1];
            const int j_nbElts = j_nextOffset - j_offset;
            if (i_nbElts == 1 || j_nbElts == 1) // no assembly required
            {
               AtomicAdd(I[i_L],1);
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
               const int min_e = GetMinElt(i_elts, i_nbElts, j_elts, j_nbElts);
               if (e == min_e) // add the nnz only once
               {
                  AtomicAdd(I[i_L],1);
               }
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

/** Returns the index where a non-zero entry should be added and increment the
    number of non-zeros for the row i_L. */
static MFEM_HOST_DEVICE int GetAndIncrementNnzIndex(const int i_L, int* I)
{
   int ind = AtomicAdd(I[i_L],1);
   return ind;
}

void LORRestriction::FillJAndData(SparseMatrix &A, const Vector &sparse_ij,
                                  const DenseMatrix &sparse_mapping) const
{
   const int ndof_per_el = fes_ho.GetFE(0)->GetDof();
   const int nel_ho = fes_ho.GetNE();
   const int nnz_per_row = sparse_mapping.Height();

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *op = fes_ho.GetElementRestriction(ordering);
   const ElementRestriction *el_restr =
      dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(el_restr != nullptr, "");

   const Array<int> &el_dof_lex_ = el_restr->GatherMap();
   const Array<int> &dof_glob2loc_ = el_restr->Indices();
   const Array<int> &dof_glob2loc_offsets_ = el_restr->Offsets();

   const auto el_dof_lex = Reshape(el_dof_lex_.Read(), ndof_per_el, nel_ho);
   const auto dof_glob2loc = dof_glob2loc_.Read();
   const auto K = dof_glob2loc_offsets_.Read();

   const auto V = Reshape(sparse_ij.Read(), nnz_per_row, ndof_per_el, nel_ho);
   const auto map = Reshape(sparse_mapping.Read(), nnz_per_row, ndof_per_el);

   const auto I = A.ReadWriteI();
   const auto J = A.ReadWriteJ();
   auto AV = A.ReadWriteData();

   static constexpr int Max = 16;

   MFEM_FORALL(iel_ho, nel_ho,
   {
      for (int ii_el = 0; ii_el < ndof_per_el; ++ii_el)
      {
         // LDOF index of current row
         const int ii = el_dof_lex(ii_el, iel_ho);
         // Get number and list of elements containing this DOF
         int i_elts[Max];
         int i_B[Max];
         const int i_offset = K[ii];
         const int i_next_offset = K[ii+1];
         const int i_ne = i_next_offset - i_offset;
         for (int e_i = 0; e_i < i_ne; ++e_i)
         {
            const int i_E = dof_glob2loc[i_offset+e_i];
            i_elts[e_i] = i_E/ndof_per_el;
            i_B[e_i] = i_E%ndof_per_el;
         }
         for (int j=0; j<nnz_per_row; ++j)
         {
            int jj_el = map(j, ii_el);
            if (jj_el < 0) { continue; }
            // LDOF index of column
            int jj = el_dof_lex(jj_el, iel_ho);
            const int j_offset = K[jj];
            const int j_next_offset = K[jj+1];
            const int j_ne = j_next_offset - j_offset;
            if (i_ne == 1 || j_ne == 1) // no assembly required
            {
               const int nnz = GetAndIncrementNnzIndex(ii, I);
               J[nnz] = jj;
               AV[nnz] = V(j, ii_el, iel_ho);
            }
            else // assembly required
            {
               int j_elts[Max];
               int j_B[Max];
               for (int e_j = 0; e_j < j_ne; ++e_j)
               {
                  const int j_E = dof_glob2loc[j_offset+e_j];
                  j_elts[e_j] = j_E/ndof_per_el;
                  j_B[e_j] = j_E%ndof_per_el;
               }
               const int min_e = GetMinElt(i_elts, i_ne, j_elts, j_ne);
               if (iel_ho == min_e) // add the nnz only once
               {
                  double val = 0.0;
                  for (int k = 0; k < i_ne; k++)
                  {
                     const int iel_ho_2 = i_elts[k];
                     const int ii_el_2 = i_B[k];
                     for (int l = 0; l < j_ne; l++)
                     {
                        const int jel_ho_2 = j_elts[l];
                        if (iel_ho_2 == jel_ho_2)
                        {
                           const int jj_el_2 = j_B[l];
                           int j2 = -1;
                           for (int m = 0; m < nnz_per_row; ++m)
                           {
                              if (map(m, ii_el_2) == jj_el_2)
                              {
                                 j2 = m;
                                 break;
                              }
                           }
                           MFEM_ASSERT_KERNEL(j >= 0, "");
                           val += V(j2, ii_el_2, iel_ho_2);
                        }
                     }
                  }
                  const int nnz = GetAndIncrementNnzIndex(ii, I);
                  J[nnz] = jj;
                  AV[nnz] = val;
               }
            }
         }
      }
   });

   // TODO: on device
   // We need to shift again the entries of I, we do it on CPU as it is very
   // sequential.
   auto h_I = A.HostReadWriteI();
   const int size = fes_ho.GetVSize();
   for (int i = 0; i < size; i++) { h_I[size-i] = h_I[size-(i+1)]; }
   h_I[0] = 0;
}

LORRestriction::~LORRestriction()
{
   delete fec_lo;
}

} // namespace mfem
