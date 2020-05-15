// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../general/forall.hpp"
#include "bilininteg.hpp"

namespace mfem
{

void TransposeIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                     Vector &ea_data)
{
   Vector ea_data_tmp(ea_data.Size());
   ea_data_tmp = 0.0;
   bfi->AssembleEA(fes, ea_data_tmp);
   const int ne = fes.GetNE();
   if (ne == 0) { return; }
   const int dofs = fes.GetFE(0)->GetDof();
   auto A = Reshape(ea_data_tmp.Write(), dofs, dofs, ne);
   auto AT = Reshape(ea_data.Write(), dofs, dofs, ne);
   MFEM_FORALL(e, ne,
   {
      for (int i = 0; i < dofs; i++)
      {
         for (int j = 0; j < dofs; j++)
         {
            const double a = A(i, j, e);
            AT(j, i, e) += a;
         }
      }
   });
}

void TransposeIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace& fes,
                                                  Vector &ea_data_int,
                                                  Vector &ea_data_ext)
{
   const int nf = fes.GetNFbyType(FaceType::Interior);
   if (nf == 0) { return; }
   Vector ea_data_int_tmp(ea_data_int.Size());
   Vector ea_data_ext_tmp(ea_data_ext.Size());
   ea_data_int_tmp = 0.0;
   ea_data_ext_tmp = 0.0;
   bfi->AssembleEAInteriorFaces(fes, ea_data_int_tmp, ea_data_ext_tmp);
   const int faceDofs = fes.GetTraceElement(0,
                                            fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof();
   auto A_int = Reshape(ea_data_int_tmp.Read(), faceDofs, faceDofs, 2, nf);
   auto A_ext = Reshape(ea_data_ext_tmp.Read(), faceDofs, faceDofs, 2, nf);
   auto AT_int = Reshape(ea_data_int.ReadWrite(), faceDofs, faceDofs, 2, nf);
   auto AT_ext = Reshape(ea_data_ext.ReadWrite(), faceDofs, faceDofs, 2, nf);
   MFEM_FORALL(f, nf,
   {
      for (int i = 0; i < faceDofs; i++)
      {
         for (int j = 0; j < faceDofs; j++)
         {
            const double a_int0 = A_int(i, j, 0, f);
            const double a_int1 = A_int(i, j, 1, f);
            const double a_ext0 = A_ext(i, j, 0, f);
            const double a_ext1 = A_ext(i, j, 1, f);
            AT_int(j, i, 0, f) += a_int0;
            AT_int(j, i, 1, f) += a_int1;
            AT_ext(j, i, 0, f) += a_ext1;
            AT_ext(j, i, 1, f) += a_ext0;
         }
      }
   });
}

void TransposeIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace& fes,
                                                  Vector &ea_data_bdr)
{
   const int nf = fes.GetNFbyType(FaceType::Boundary);
   if (nf == 0) { return; }
   Vector ea_data_bdr_tmp(ea_data_bdr.Size());
   ea_data_bdr_tmp = 0.0;
   bfi->AssembleEABoundaryFaces(fes, ea_data_bdr_tmp);
   const int faceDofs = fes.GetTraceElement(0,
                                            fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof();
   auto A_bdr = Reshape(ea_data_bdr_tmp.Read(), faceDofs, faceDofs, nf);
   auto AT_bdr = Reshape(ea_data_bdr.ReadWrite(), faceDofs, faceDofs, nf);
   MFEM_FORALL(f, nf,
   {
      for (int i = 0; i < faceDofs; i++)
      {
         for (int j = 0; j < faceDofs; j++)
         {
            const double a_bdr = A_bdr(i, j, f);
            AT_bdr(j, i, f) += a_bdr;
         }
      }
   });
}

}
