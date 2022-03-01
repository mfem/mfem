// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

void LORRestriction::FillJAndZeroData(SparseMatrix &mat) const
{
   static constexpr int Max = 8;
   const int all_dofs = ndofs;
   const int vd = vdim;
   const int elt_dofs = lo_dof_per_el;
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   const int NE = ne;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gatherMap = gatherMap.Read();

   MFEM_FORALL(e, NE,
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
               const int nnz = AtomicAdd(I[i_L],1);
               J[nnz] = j_L;
               Data[nnz] = 0.0;
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
                  const int nnz = AtomicAdd(I[i_L],1);
                  J[nnz] = j_L;
                  Data[nnz] = 0.0;
               }
            }
         }
      }
   });
   // We need to shift again the entries of I, we do it on CPU as it is very
   // sequential.
   auto h_I = mat.HostReadWriteI();
   const int size = vd*all_dofs;
   for (int i = 0; i < size; i++) { h_I[size-i] = h_I[size-(i+1)]; }
   h_I[0] = 0;
}

LORRestriction::~LORRestriction()
{
   delete fec_lo;
}

} // namespace mfem
