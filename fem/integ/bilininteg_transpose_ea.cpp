// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"

namespace mfem
{

void TransposeIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                     Vector &ea_data, const bool add)
{
   if (add)
   {
      Vector ea_data_tmp(ea_data.Size());
      bfi->AssembleEA(fes, ea_data_tmp, false);
      const int ne = fes.GetNE();
      const int dofs = fes.GetTypicalFE()->GetDof();
      auto A = Reshape(ea_data_tmp.Read(), dofs, dofs, ne);
      auto AT = Reshape(ea_data.ReadWrite(), dofs, dofs, ne);
      mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int i = 0; i < dofs; i++)
         {
            for (int j = 0; j < dofs; j++)
            {
               const real_t a = A(i, j, e);
               AT(j, i, e) += a;
            }
         }
      });
   }
   else
   {
      bfi->AssembleEA(fes, ea_data, false);
      const int ne = fes.GetNE();
      const int dofs = fes.GetTypicalFE()->GetDof();
      auto A = Reshape(ea_data.ReadWrite(), dofs, dofs, ne);
      mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int i = 0; i < dofs; i++)
         {
            for (int j = i+1; j < dofs; j++)
            {
               const real_t aij = A(i, j, e);
               const real_t aji = A(j, i, e);
               A(j, i, e) = aij;
               A(i, j, e) = aji;
            }
         }
      });
   }
}

void TransposeIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace& fes,
                                                  Vector &ea_data_int,
                                                  Vector &ea_data_ext,
                                                  const bool add)
{
   const int nf = fes.GetNFbyType(FaceType::Interior);
   if (nf == 0) { return; }
   if (add)
   {
      Vector ea_data_int_tmp(ea_data_int.Size());
      Vector ea_data_ext_tmp(ea_data_ext.Size());
      bfi->AssembleEAInteriorFaces(fes, ea_data_int_tmp, ea_data_ext_tmp, false);
      const int faceDofs = fes.GetTypicalTraceElement()->GetDof();
      auto A_int = Reshape(ea_data_int_tmp.Read(), faceDofs, faceDofs, 2, nf);
      auto A_ext = Reshape(ea_data_ext_tmp.Read(), faceDofs, faceDofs, 2, nf);
      auto AT_int = Reshape(ea_data_int.ReadWrite(), faceDofs, faceDofs, 2, nf);
      auto AT_ext = Reshape(ea_data_ext.ReadWrite(), faceDofs, faceDofs, 2, nf);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         for (int i = 0; i < faceDofs; i++)
         {
            for (int j = 0; j < faceDofs; j++)
            {
               const real_t a_int0 = A_int(i, j, 0, f);
               const real_t a_int1 = A_int(i, j, 1, f);
               const real_t a_ext0 = A_ext(i, j, 0, f);
               const real_t a_ext1 = A_ext(i, j, 1, f);
               AT_int(j, i, 0, f) += a_int0;
               AT_int(j, i, 1, f) += a_int1;
               AT_ext(j, i, 0, f) += a_ext1;
               AT_ext(j, i, 1, f) += a_ext0;
            }
         }
      });
   }
   else
   {
      bfi->AssembleEAInteriorFaces(fes, ea_data_int, ea_data_ext, false);
      const int faceDofs = fes.GetTypicalTraceElement()->GetDof();
      auto A_int = Reshape(ea_data_int.ReadWrite(), faceDofs, faceDofs, 2, nf);
      auto A_ext = Reshape(ea_data_ext.ReadWrite(), faceDofs, faceDofs, 2, nf);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         for (int i = 0; i < faceDofs; i++)
         {
            for (int j = i+1; j < faceDofs; j++)
            {
               const real_t aij_int0 = A_int(i, j, 0, f);
               const real_t aij_int1 = A_int(i, j, 1, f);
               const real_t aji_int0 = A_int(j, i, 0, f);
               const real_t aji_int1 = A_int(j, i, 1, f);
               A_int(j, i, 0, f) = aij_int0;
               A_int(j, i, 1, f) = aij_int1;
               A_int(i, j, 0, f) = aji_int0;
               A_int(i, j, 1, f) = aji_int1;
            }
         }
         for (int i = 0; i < faceDofs; i++)
         {
            for (int j = 0; j < faceDofs; j++)
            {
               const real_t aij_ext0 = A_ext(i, j, 0, f);
               const real_t aji_ext1 = A_ext(j, i, 1, f);
               A_ext(j, i, 1, f) = aij_ext0;
               A_ext(i, j, 0, f) = aji_ext1;
            }
         }
      });
   }
}

void TransposeIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace& fes,
                                                  Vector &ea_data_bdr,
                                                  const bool add)
{
   const int nf = fes.GetNFbyType(FaceType::Boundary);
   if (nf == 0) { return; }
   if (add)
   {
      Vector ea_data_bdr_tmp(ea_data_bdr.Size());
      bfi->AssembleEABoundaryFaces(fes, ea_data_bdr_tmp, false);
      const int faceDofs = fes.GetTypicalTraceElement()->GetDof();
      auto A_bdr = Reshape(ea_data_bdr_tmp.Read(), faceDofs, faceDofs, nf);
      auto AT_bdr = Reshape(ea_data_bdr.ReadWrite(), faceDofs, faceDofs, nf);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         for (int i = 0; i < faceDofs; i++)
         {
            for (int j = 0; j < faceDofs; j++)
            {
               const real_t a_bdr = A_bdr(i, j, f);
               AT_bdr(j, i, f) += a_bdr;
            }
         }
      });
   }
   else
   {
      bfi->AssembleEABoundaryFaces(fes, ea_data_bdr, false);
      const int faceDofs = fes.GetTypicalTraceElement()->GetDof();
      auto A_bdr = Reshape(ea_data_bdr.ReadWrite(), faceDofs, faceDofs, nf);
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         for (int i = 0; i < faceDofs; i++)
         {
            for (int j = i+1; j < faceDofs; j++)
            {
               const real_t aij_bdr = A_bdr(i, j, f);
               const real_t aji_bdr = A_bdr(j, i, f);
               A_bdr(j, i, f) = aij_bdr;
               A_bdr(i, j, f) = aji_bdr;
            }
         }
      });
   }
}

}
