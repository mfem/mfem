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

#include "unit_tests.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

namespace pa_kernels
{

TEST_CASE("H1 SumIntegrator", "[SumIntegrator][PartialAssembly]")
{
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
   H1_FECollection fec(2, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   MassIntegrator integ1;
   DiffusionIntegrator integ2;

   SumIntegrator integ_sum(true);
   integ_sum.AddIntegrator(new MassIntegrator);
   integ_sum.AddIntegrator(new DiffusionIntegrator);

   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &T = *mesh.GetTypicalElementTransformation();
   DenseMatrix m1, m_tmp, m2;

   // AssembleElementMatrix
   integ1.AssembleElementMatrix(el, T, m1);
   integ2.AssembleElementMatrix(el, T, m_tmp);
   m1 += m_tmp;
   integ_sum.AssembleElementMatrix(el, T, m2);
   m1 -= m2;
   REQUIRE(m1.MaxMaxNorm() == MFEM_Approx(0.0));

   // AssembleElementMatrix2
   integ1.AssembleElementMatrix2(el, el, T, m1);
   integ2.AssembleElementMatrix2(el, el, T, m_tmp);
   m1 += m_tmp;
   integ_sum.AssembleElementMatrix2(el, el, T, m2);
   m1 -= m2;
   REQUIRE(m1.MaxMaxNorm() == MFEM_Approx(0.0));

   // PA
   integ1.AssemblePA(fes);
   integ2.AssemblePA(fes);
   integ_sum.AssemblePA(fes);

   int n = fes.GetTrueVSize();
   Vector x(n), y1(n), y2(n);
   Vector diag1(n), diag_tmp(n), diag2(n);
   x.Randomize(1);

   // AddMultPA
   y1 = 0.0;
   y2 = 0.0;
   integ1.AddMultPA(x, y1);
   integ2.AddMultPA(x, y1);
   integ_sum.AddMultPA(x, y2);
   y1 -= y2;
   REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));

   // AddMultTransposePA
   y1 = 0.0;
   y2 = 0.0;
   integ1.AddMultTransposePA(x, y1);
   integ2.AddMultTransposePA(x, y1);
   integ_sum.AddMultTransposePA(x, y2);
   y1 -= y2;
   REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));

   // AssembleDiagonalPA
   diag1 = 0.0;
   diag_tmp = 0.0;
   diag2 = 0.0;
   integ1.AssembleDiagonalPA(diag1);
   integ2.AssembleDiagonalPA(diag_tmp);
   diag1 += diag_tmp;
   integ_sum.AssembleDiagonalPA(diag2);
   diag1 -= diag2;
   REQUIRE(diag1.Normlinf() == MFEM_Approx(0.0));

   // MF
#ifdef MFEM_USE_CEED
   if (DeviceCanUseCeed())
   {
      integ1.AssembleMF(fes);
      integ2.AssembleMF(fes);
      integ_sum.AssembleMF(fes);

      // AddMultMF
      y1 = 0.0;
      y2 = 0.0;
      integ1.AddMultMF(x, y1);
      integ2.AddMultMF(x, y1);
      integ_sum.AddMultMF(x, y2);
      y1 -= y2;
      REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));

      // AddMultTransposeMF
      y1 = 0.0;
      y2 = 0.0;
      integ1.AddMultTransposeMF(x, y1);
      integ2.AddMultTransposeMF(x, y1);
      integ_sum.AddMultTransposeMF(x, y2);
      y1 -= y2;
      REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));

      // AssembleDiagonalMF
      integ1.AssembleDiagonalMF(diag1);
      integ2.AssembleDiagonalMF(diag_tmp);
      diag1 += diag_tmp;
      integ_sum.AssembleDiagonalMF(diag2);
      diag1 -= diag2;
      REQUIRE(diag1.Normlinf() == MFEM_Approx(0.0));
   }
#endif
}

TEST_CASE("DG SumIntegrator", "[SumIntegrator][PartialAssembly]")
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
   DG_FECollection fec(2, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   Vector v(mesh.Dimension());
   v = 1.0;
   VectorConstantCoefficient v_coeff(v);

   DGTraceIntegrator integ1(v_coeff, 1.0, 2.0);
   DGTraceIntegrator integ2(v_coeff, 3.0, 4.0);

   SumIntegrator integ_sum(true);
   integ_sum.AddIntegrator(new DGTraceIntegrator(v_coeff, 1.0, 2.0));
   integ_sum.AddIntegrator(new DGTraceIntegrator(v_coeff, 3.0, 4.0));

   DenseMatrix m1, m_tmp, m2;

   // AssembleFaceMatrix
   int nfaces = mesh.GetNumFaces();
   for (int i = 0; i < nfaces; i++)
   {
      FaceElementTransformations *tr = mesh.GetFaceElementTransformations(i);
      const FiniteElement &el0 = *fes.GetFE(tr->Elem1No);
      const FiniteElement &el1 = (tr->Elem2No >= 0) ? *fes.GetFE(tr->Elem2No) : el0;
      integ1.AssembleFaceMatrix(el0, el1, *tr, m1);
      integ2.AssembleFaceMatrix(el0, el1, *tr, m_tmp);
      m1 += m_tmp;
      integ_sum.AssembleFaceMatrix(el0, el1, *tr, m2);
      m1 -= m2;
      REQUIRE(m1.MaxMaxNorm() == MFEM_Approx(0.0));
   }

   // PA interior
   integ1.AssemblePAInteriorFaces(fes);
   integ2.AssemblePAInteriorFaces(fes);
   integ_sum.AssemblePAInteriorFaces(fes);

   const FaceRestriction *R_int = fes.GetFaceRestriction(
                                     ElementDofOrdering::LEXICOGRAPHIC,
                                     FaceType::Interior);

   int n_int = R_int->Height();
   Vector x(n_int), y1(n_int), y2(n_int);
   x.Randomize(1);

   // AddMultPA
   y1 = 0.0;
   y2 = 0.0;
   integ1.AddMultPA(x, y1);
   integ2.AddMultPA(x, y1);
   integ_sum.AddMultPA(x, y2);
   y1 -= y2;
   REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));

   // AddMultTransposePA
   y1 = 0.0;
   y2 = 0.0;
   integ1.AddMultTransposePA(x, y1);
   integ2.AddMultTransposePA(x, y1);
   integ_sum.AddMultTransposePA(x, y2);
   y1 -= y2;
   REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));

   // PA boundary
   integ1.AssemblePABoundaryFaces(fes);
   integ2.AssemblePABoundaryFaces(fes);
   integ_sum.AssemblePABoundaryFaces(fes);

   const FaceRestriction *R_bdr = fes.GetFaceRestriction(
                                     ElementDofOrdering::LEXICOGRAPHIC,
                                     FaceType::Boundary,
                                     L2FaceValues::DoubleValued);

   int n_bdr = R_bdr->Height();
   x.SetSize(n_bdr);
   y1.SetSize(n_bdr);
   y2.SetSize(n_bdr);
   x.Randomize(1);

   // AddMultPA
   y1 = 0.0;
   y2 = 0.0;
   integ1.AddMultPA(x, y1);
   integ2.AddMultPA(x, y1);
   integ_sum.AddMultPA(x, y2);
   y1 -= y2;
   REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));

   // AddMultTransposePA
   y1 = 0.0;
   y2 = 0.0;
   integ1.AddMultTransposePA(x, y1);
   integ2.AddMultTransposePA(x, y1);
   integ_sum.AddMultTransposePA(x, y2);
   y1 -= y2;
   REQUIRE(y1.Normlinf() == MFEM_Approx(0.0));
}

} // namespace pa_kernels
