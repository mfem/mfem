// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

#include "util.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

static void InitNativeRestr(const mfem::FiniteElementSpace &fes,
                            bool use_bdr,
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   const int nelem = use_bdr ? fes.GetNBE() : fes.GetNE();
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0) :
                                   fes.GetFE(0);
   const int P = fe->GetDof();
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const mfem::Array<int>& dof_map = tfe->GetDofMap();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   const mfem::Table &el_dof = use_bdr ? fes.GetBdrElementToDofTable() :
                               fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());

   for (int i = 0; i < nelem; i++)
   {
      // TODO: Implement DofTransformation support
      const int el_offset = P * i;
      for (int j = 0; j < P; j++)
      {
         tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
      }
   }

   CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                             compstride, fes.GetVDim() * fes.GetNDofs(),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitLexicoRestr(const mfem::FiniteElementSpace &fes,
                            bool use_bdr,
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   const int nelem = use_bdr ? fes.GetNBE() : fes.GetNE();
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0) :
                                   fes.GetFE(0);
   const int P = fe->GetDof();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   const mfem::Table &el_dof = use_bdr ? fes.GetBdrElementToDofTable() :
                               fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());

   for (int e = 0; e < nelem; e++)
   {
      // TODO: Implement DofTransformation support
      for (int i = 0; i < P; i++)
      {
         tp_el_dof[i + e*P] = stride*el_dof.GetJ()[i + e*P];
      }
   }

   CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                             compstride, fes.GetVDim() * fes.GetNDofs(),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitNativeRestrWithIndices(const mfem::FiniteElementSpace &fes,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Ceed ceed,
                                       CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(indices[0]) :
                                   fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const mfem::Array<int> &dof_map = tfe->GetDofMap();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   mfem::Array<int> tp_el_dof(nelem * P), dofs;

   for (int i = 0; i < nelem; i++)
   {
      // TODO: Implement DofTransformation support
      const int elem_index = indices[i];
      if (use_bdr)
      {
         fes.GetBdrElementDofs(elem_index, dofs);
      }
      else
      {
         fes.GetElementDofs(elem_index, dofs);
      }
      const int el_offset = P * i;
      for (int j = 0; j < P; j++)
      {
         tp_el_dof[j + el_offset] = stride*dofs[dof_map[j]];
      }
   }

   CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                             compstride, fes.GetVDim() * fes.GetNDofs(),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitLexicoRestrWithIndices(const mfem::FiniteElementSpace &fes,
                                       bool use_bdr,
                                       int nelem,
                                       const int *indices,
                                       Ceed ceed,
                                       CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(indices[0]) :
                                   fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   CeedInt compstride =
      (fes.GetOrdering() == Ordering::byVDIM) ? 1 : fes.GetNDofs();
   const int stride = (compstride == 1) ? fes.GetVDim() : 1;
   mfem::Array<int> tp_el_dof(nelem * P), dofs;

   for (int i = 0; i < nelem; i++)
   {
      // TODO: Implement DofTransformation support
      const int elem_index = indices[i];
      if (use_bdr)
      {
         fes.GetBdrElementDofs(elem_index, dofs);
      }
      else
      {
         fes.GetElementDofs(elem_index, dofs);
      }
      const int el_offset = P * i;
      for (int j = 0; j < P; j++)
      {
         tp_el_dof[j + el_offset] = stride*dofs[j];
      }
   }

   CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                             compstride, fes.GetVDim() * fes.GetNDofs(),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitRestrictionImpl(const mfem::FiniteElementSpace &fes,
                                bool use_bdr,
                                Ceed ceed,
                                CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0): fes.GetFE(0);
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   if (tfe && tfe->GetDofMap().Size() > 0)  // Native ordering using dof_map
   {
      InitNativeRestr(fes, use_bdr, ceed, restr);
   }
   else  // Lexicographic ordering
   {
      InitLexicoRestr(fes, use_bdr, ceed, restr);
   }
}

static void InitRestrictionWithIndicesImpl(const mfem::FiniteElementSpace &fes,
                                           bool use_bdr,
                                           int nelem,
                                           const int *indices,
                                           Ceed ceed,
                                           CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(indices[0]) :
                                   fes.GetFE(indices[0]);
   const mfem::TensorBasisElement *tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   if (tfe && tfe->GetDofMap().Size() > 0)  // Native ordering using dof_map
   {
      InitNativeRestrWithIndices(fes, use_bdr, nelem, indices, ceed, restr);
   }
   else  // Lexicographic ordering
   {
      InitLexicoRestrWithIndices(fes, use_bdr, nelem, indices, ceed, restr);
   }
}

void InitRestriction(const FiniteElementSpace &fes,
                     bool use_bdr,
                     Ceed ceed,
                     CeedElemRestriction *restr)
{
   // Check for fes -> restriction in hash table
   // {-1, -1, -1} is unique from CEED_STRIDES_BACKEND for non-strided restrictions
   const int nelem = use_bdr ? fes.GetNBE() : fes.GetNE();
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(0) : fes.GetFE(0);
   const int P = fe->GetDof();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, -1, -1, -1);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      InitRestrictionImpl(fes, use_bdr, ceed, restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitRestrictionWithIndices(const FiniteElementSpace &fes,
                                bool use_bdr,
                                int nelem,
                                const int *indices,
                                Ceed ceed,
                                CeedElemRestriction *restr)
{
   // Check for fes -> restriction in hash table
   // {-1, -1, -1} is unique from CEED_STRIDES_BACKEND for non-strided restrictions
   const mfem::FiniteElement *fe = use_bdr ? fes.GetBE(indices[0]) :
                                   fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, -1, -1, -1);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      InitRestrictionWithIndicesImpl(fes, use_bdr, nelem, indices, ceed, restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            CeedInt nelem,
                            CeedInt nqpts,
                            CeedInt qdatasize,
                            const CeedInt strides[3],
                            Ceed ceed,
                            CeedElemRestriction *restr)
{
   // Check for fes -> restriction in hash table
   RestrKey restr_key(&fes, nelem, nqpts, qdatasize,
                      strides[0], strides[1], strides[2]);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      CeedElemRestrictionCreateStrided(ceed, nelem, nqpts, qdatasize,
                                       nelem * nqpts * qdatasize, strides,
                                       restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

#endif

} // namespace ceed

} // namespace mfem
