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

#include "lor_assembly.hpp"
#include "../linalg/dtensor.hpp"
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

#include "../general/nvvp.hpp"
//#define NvtxPush(...)
//#define NvtxPop(...)

#ifdef MFEM_USE_CUDA
#include <cub/device/device_scan.cuh>
#endif

namespace mfem
{

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

struct XRestriction
{
   const FiniteElementSpace &fes;
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;
   const int nedofs;
   Array<int> offsets;
   Array<int> indices;
   Array<int> gatherMap;

   XRestriction(const FiniteElementSpace &f)
      : fes(f),
        ne(fes.GetNE()),
        vdim(fes.GetVDim()),
        byvdim(fes.GetOrdering() == Ordering::byVDIM),
        ndofs(fes.GetNDofs()),
        dof(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
        nedofs(ne*dof),
        offsets(ndofs+1),
        indices(ne*dof),
        gatherMap(ne*dof)
   {}

   void Setup()
   {
      NvtxPush(Setup,Chocolate);

      NvtxPush(Ini,LightBlue);
      assert(ne>0);
      const FiniteElement *fe = fes.GetFE(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      assert(el);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
      NvtxPop(Ini);

      const Table& e2dTable = fes.GetElementToDofTable();
      const int* elementMap = e2dTable.GetJ();

      auto d_offsets = offsets.Write();
      const int NDOFS = ndofs;
      NvtxPush(Flush,DarkSalmon);
      MFEM_FORALL(i, NDOFS+1, d_offsets[i] = 0;);
      NvtxPop(Flush);

      NvtxPush(offsets,IndianRed);
      const Memory<int> &J = e2dTable.GetJMemory();
      const MemoryClass mc = Device::GetDeviceMemoryClass();
      const int *d_elementMap = J.Read(mc, J.Capacity());
      const int DOF = dof;
      MFEM_FORALL(e, ne,
      {
         for (int d = 0; d < DOF; ++d)
         {
            const int sgid = d_elementMap[DOF*e + d];  // signed
            const int gid = (sgid >= 0) ? sgid : -1 - sgid;
            AtomicAdd(d_offsets[gid+1], 1);
         }
      });
      NvtxPop(offsets);

      NvtxPush(Aggregate,Moccasin);
      // Aggregate to find offsets for each global dof
      if (!Device::IsEnabled())
      {
         offsets.HostReadWrite();
         for (int i = 1; i <= ndofs; ++i)
         {
            offsets[i] += offsets[i - 1];
         }
      }
      else
      {
#ifdef MFEM_USE_CUDA
         const int N = offsets.Size();
         Array<int> out(N);

         const auto d_in = d_offsets;
         auto d_out = out.Write();

         void     *d_temp_storage = nullptr;
         size_t   temp_storage_bytes = 0;
         cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                       d_in, d_out, N);
         static Array<char> *tmp = nullptr;
         if (!tmp) { tmp = new Array<char>(); }
         tmp->SetSize(temp_storage_bytes);
         auto d_tmp = tmp->Write();
         // Run inclusive prefix sum
         cub::DeviceScan::InclusiveSum(d_tmp, temp_storage_bytes,
                                       d_in, d_out, N);
         // copy back
         offsets = out;
#else
         MFEM_ABORT("Not supported!");
#endif
      }
      NvtxPop(Aggregate);

      NvtxPush(Fill,DarkOrange);
      // For each global dof, fill in all local nodes that point to it
      auto d_gather = gatherMap.Write();
      auto d_indices = indices.Write();
      auto drw_offsets = offsets.ReadWrite();
      const auto dof_map_mem = fe_dof_map.GetMemory();
      const auto d_dof_map = fe_dof_map.GetMemory().Read(mc,dof_map_mem.Capacity());
      MFEM_FORALL(e, ne,
      {
         for (int d = 0; d < DOF; ++d)
         {
            const int sdid = d_dof_map[d];  // signed
            const int did = d;
            const int sgid = d_elementMap[DOF*e + did];  // signed
            const int gid = (sgid >= 0) ? sgid : -1-sgid;
            const int lid = DOF*e + d;
            const bool plus = (sgid >= 0 && sdid >= 0) || (sgid < 0 && sdid < 0);
            d_gather[lid] = plus ? gid : -1-gid;
            d_indices[AtomicAdd(d_offsets[gid], 1)] = plus ? lid : -1-lid;
         }
      });
      NvtxPop(Fill);

      NvtxPush(Shift,YellowGreen);
      offsets.HostReadWrite();
      for (int i = ndofs; i > 0; --i)
      {
         offsets[i] = offsets[i - 1];
      }
      offsets[0] = 0;
      NvtxPop(Shift);

      NvtxPop(Setup);
   }

   int FillI(SparseMatrix &mat) const
   {
      static constexpr int Max = 16;
      const int all_dofs = ndofs;
      const int vd = vdim;
      const int elt_dofs = dof;
      auto I = mat.ReadWriteI();
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
                  int min_e = GetMinElt(i_elts, i_nbElts, j_elts, j_nbElts);
                  if (e == min_e) // add the nnz only once
                  {
                     //GetAndIncrementNnzIndex(i_L, I);
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

   const Array<int> &GatherMap() const { return gatherMap; }
   const Array<int> &Indices() const { return indices; }
   const Array<int> &Offsets() const { return offsets; }
};


template <int order>
void Assemble3DBatchedLOR_GPU(Mesh &mesh_lor,
                              Array<int> &dof_glob2loc_,
                              Array<int> &dof_glob2loc_offsets_,
                              Array<int> &el_dof_lex_,
                              Vector &Q_,
                              Mesh &mesh_ho,
                              FiniteElementSpace &fes_ho,
                              SparseMatrix &A_mat);

void AssembleBatchedLOR_GPU(BilinearForm &form_lor,
                            FiniteElementSpace &fes_ho,
                            const Array<int> &ess_dofs,
                            OperatorHandle &Ah)
{
   NvtxPush(AssembleBatchedLOR_GPU, LawnGreen);

   Mesh &mesh_lor = *form_lor.FESpace()->GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();
   const int ndofs = fes_ho.GetTrueVSize();

   const int nel_ho = mesh_ho.GetNE();
   const int ndof = fes_ho.GetVSize();
   constexpr int nv = 8;
   const int ddm2 = (dim*(dim+1))/2;
   const int nd1d = order + 1;
   const int ndof_per_el = nd1d*nd1d*nd1d;

   const bool has_to_init = Ah.Ptr() == nullptr;
   //dbg("has_to_init: %s", has_to_init?"yes":"no");
   SparseMatrix *mat = has_to_init ? nullptr : Ah.As<SparseMatrix>();

   NvtxPush(Arrays,WebMaroon);
   Array<int> dof_glob2loc_(2*ndof_per_el*nel_ho);
   Array<int> dof_glob2loc_offsets_(ndof+1);
   Array<int> el_dof_lex_(ndof_per_el*nel_ho);
   Vector Q_(nel_ho*pow(order,dim)*nv*ddm2);
   NvtxPop();

   NvtxPush(EnsureNodes,Chocolate);
   // nodes will be ordered byVDIM but won't use SetCurvature each time
   mesh_lor.EnsureNodes();
   NvtxPop();

   if (has_to_init)
   {
      NvtxPush(Sparsity, PaleTurquoise);

      if (Device::IsEnabled())
      {
         NvtxPush(Dev, LightGoldenrod);
         FiniteElementSpace &fes_lo = *form_lor.FESpace();
         const int width = fes_lo.GetVSize();
         const int height = fes_lo.GetVSize();

         NvtxPush(new mat,PapayaWhip);
         mat = new SparseMatrix(height, width, 0);
         NvtxPop();

         NvtxPush(R,Purple);
         //const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
         //const Operator *Rop = fes_lo.GetElementRestriction(ordering);
         //const ElementRestriction &R = static_cast<const ElementRestriction&>(*Rop);
         static XRestriction *R = nullptr;
         if (!R)
         {
            R = new XRestriction(fes_lo);
            R->Setup();
         }
         NvtxPop();

         NvtxPush(newI,Purple);
         mat->GetMemoryI().New(mat->Height()+1, mat->GetMemoryI().GetMemoryType());
         NvtxPop();

         NvtxPush(FillI,Azure);
         const int nnz = R->FillI(*mat);
         NvtxPop();

         NvtxPush(NewJ,Orchid);
         mat->GetMemoryJ().New(nnz, mat->GetMemoryJ().GetMemoryType());
         NvtxPop();

         NvtxPush(NewData,Khaki);
         mat->GetMemoryData().New(nnz, mat->GetMemoryData().GetMemoryType());
         NvtxPop();

         {
            NvtxPush(J,Chartreuse);
            static constexpr int Max = 8;
            const int all_dofs = ndofs;
            const int vd = fes_lo.GetVDim();
            const int elt_dofs = fes_lo.GetFE(0)->GetDof();
            auto I = mat->ReadWriteI();
            auto J = mat->WriteJ();
            const int NE = fes_lo.GetNE();

            auto d_offsets = R->Offsets().Read();
            auto d_indices = R->Indices().Read();
            auto d_gatherMap = R->GatherMap().Read();

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
                        J[AtomicAdd(I[i_L],1)] = j_L;
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
                           J[AtomicAdd(I[i_L],1)] = j_L;
                        }
                     }
                  }
               }
            });
            NvtxPop();
            NvtxPush(Shift,LightCyan);
            // We need to shift again the entries of I, we do it on CPU as it is very
            // sequential.
            auto h_I = mat->HostReadWriteI();
            const int size = vd*all_dofs;
            for (int i = 0; i < size; i++) { h_I[size-i] = h_I[size-(i+1)]; }
            h_I[0] = 0;
            NvtxPop();
         }
         NvtxPop(Dev);
      }
      else
      {
         NvtxPush(CPU,DarkKhaki);
         MFEM_VERIFY(UsesTensorBasis(fes_ho),
                     "Batched LOR assembly requires tensor basis");

         // the sparsity pattern is defined from the map: element->dof
         const Table &elem_dof = form_lor.FESpace()->GetElementToDofTable();

         Table dof_dof, dof_elem;

         NvtxPush(Transpose, LightGoldenrod);
         Transpose(elem_dof, dof_elem, ndofs);
         NvtxPop(Transpose);

         NvtxPush(Mult, LightGoldenrod);
         mfem::Mult(dof_elem, elem_dof, dof_dof);
         NvtxPop();

         NvtxPush(SortRows, LightGoldenrod);
         dof_dof.SortRows();
         int *I = dof_dof.GetI();
         int *J = dof_dof.GetJ();
         NvtxPop();

         NvtxPush(A_Allocate, Cyan);
         double *data = Memory<double>(I[ndofs]);
         NvtxPop();

         NvtxPush(newSparseMatrix, PeachPuff);
         mat = new SparseMatrix(I,J,data,ndofs,ndofs,true,true,true);
         NvtxPop();

         NvtxPush(LoseData, PaleTurquoise);
         dof_dof.LoseData();
         NvtxPop();
         NvtxPop(CPU);
      } // IJ

      NvtxPush(A=0.0, Peru);
      *mat = 0.0;
      NvtxPop();

      {
         NvtxPush(BlockMapping, Olive);
         Array<int> dofs;
         const Array<int> &lex_map =
            dynamic_cast<const NodalFiniteElement&>
            (*fes_ho.GetFE(0)).GetLexicographicOrdering();
         dof_glob2loc_offsets_ = 0;
         for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
         {
            fes_ho.GetElementDofs(iel_ho, dofs);
            for (int i=0; i<ndof_per_el; ++i)
            {
               const int dof = dofs[lex_map[i]];
               el_dof_lex_[i + iel_ho*ndof_per_el] = dof;
               dof_glob2loc_offsets_[dof+1] += 2;
            }
         }
         dof_glob2loc_offsets_.PartialSum();
         // Sanity check
         MFEM_VERIFY(dof_glob2loc_offsets_[ndof] == dof_glob2loc_.Size(), "");
         Array<int> dof_ptr(ndof);
         for (int i=0; i<ndof; ++i) { dof_ptr[i] = dof_glob2loc_offsets_[i]; }
         for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
         {
            fes_ho.GetElementDofs(iel_ho, dofs);
            for (int i=0; i<ndof_per_el; ++i)
            {
               const int dof = dofs[lex_map[i]];
               dof_glob2loc_[dof_ptr[dof]++] = iel_ho;
               dof_glob2loc_[dof_ptr[dof]++] = i;
            }
         }
         NvtxPop(BlockMapping);
      }
      NvtxPop(Sparsity);
   }

   void (*Kernel)(Mesh &mesh_lor,
                  Array<int> &dof_glob2loc_,
                  Array<int> &dof_glob2loc_offsets_,
                  Array<int> &el_dof_lex_,
                  Vector &Q_,
                  Mesh &mesh_ho,
                  FiniteElementSpace &fes_ho,
                  SparseMatrix &A_mat) = nullptr;

   if (dim == 2) { MFEM_ABORT("Unsuported!"); }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Kernel = Assemble3DBatchedLOR_GPU<1>; break;
         case 2: Kernel = Assemble3DBatchedLOR_GPU<2>; break;
         case 3: Kernel = Assemble3DBatchedLOR_GPU<3>; break;
         case 4: Kernel = Assemble3DBatchedLOR_GPU<4>; break;
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   NvtxPush(Kernel, GreenYellow);
   Kernel(mesh_lor,
          dof_glob2loc_,
          dof_glob2loc_offsets_,
          el_dof_lex_,
          Q_,
          mesh_ho,fes_ho,*mat);
   NvtxPop(Kernel);

   {
      NvtxPush(D=0.0, DarkGoldenrod);
      const auto I_d = mat->ReadI();
      const auto J_d = mat->ReadJ();
      auto A_d = mat->ReadWriteData();
      const int n_ess_dofs = ess_dofs.Size();

      MFEM_FORALL(i, n_ess_dofs,
      {
         for (int j=I_d[i]; j<I_d[i+1]; ++j)
         {
            if (J_d[j] != i)
            {
               A_d[j] = 0.0;
            }
         }
      });
      NvtxPop();
   }

   if (has_to_init) { Ah.Reset(mat); } // A now owns A_mat
   NvtxPop(AssembleBatchedLOR_GPU);
}

} // namespace mfem
