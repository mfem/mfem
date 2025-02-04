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

#include "mfem.hpp"
#include "unit_tests.hpp"
#include <sstream>

namespace mfem
{

TEST_CASE("SparseMatrixAbsMult", "[SparseMatrixAbsMult]")
{
   int dim = 2;
   int ne = 4;
   for (int order = 1; order <= 3; ++order)
   {
      CAPTURE(order);

      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
      FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
      FiniteElementSpace R_space(&mesh, hdiv_coll);
      FiniteElementSpace W_space(&mesh, l2_coll);

      int n = R_space.GetTrueVSize();
      int m = W_space.GetTrueVSize();
      MixedBilinearForm a(&R_space, &W_space);
      a.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      a.Assemble();
      a.Finalize();

      SparseMatrix &A = a.SpMat();
      SparseMatrix *Aabs = new SparseMatrix(A);

      int nnz = Aabs->NumNonZeroElems();
      for (int j = 0; j < nnz; j++)
      {
         Aabs->GetData()[j] = fabs(Aabs->GetData()[j]);
      }

      Vector X0(n), X1(n);
      Vector Y0(m), Y1(m);

      X0.Randomize();
      Y0.Randomize(1);
      Y1.Randomize(1);
      A.AbsMult(X0,Y0);
      Aabs->Mult(X0,Y1);

      Y1 -= Y0;
      double error = Y1.Norml2();

      REQUIRE(error == MFEM_Approx(0.0));

      Y0.Randomize();
      X0.Randomize(1);
      X1.Randomize(1);
      A.AbsMultTranspose(Y0,X0);
      Aabs->MultTranspose(Y0,X1);
      X1 -= X0;

      error = X1.Norml2();

      REQUIRE(error == MFEM_Approx(0.0));

      delete Aabs;
      delete hdiv_coll;
      delete l2_coll;
   }
}

TEST_CASE("SparseMatrixPowAbsMult", "[SparseMatrixPowAbsMult]")
{
   int dim = 2;
   int ne = 4;
   real_t power = 1.5;
   for (int order = 1; order <= 3; ++order)
   {
      CAPTURE(order);

      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
      FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
      FiniteElementSpace R_space(&mesh, hdiv_coll);
      FiniteElementSpace W_space(&mesh, l2_coll);

      int n = R_space.GetTrueVSize();
      int m = W_space.GetTrueVSize();
      MixedBilinearForm a(&R_space, &W_space);
      a.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      a.Assemble();
      a.Finalize();

      SparseMatrix &A = a.SpMat();
      SparseMatrix *Aabs = new SparseMatrix(A);

      int nnz = Aabs->NumNonZeroElems();
      for (int j = 0; j < nnz; j++)
      {
         Aabs->GetData()[j] = std::pow(fabs(Aabs->GetData()[j]), power);
      }

      Vector X0(n), X1(n);
      Vector Y0(m), Y1(m);

      X0.Randomize();
      Y0.Randomize(1);
      Y1.Randomize(1);
      A.PowAbsMult(power,X0,Y0);
      Aabs->Mult(X0,Y1);

      Y1 -= Y0;
      double error = Y1.Norml2();

      REQUIRE(error == MFEM_Approx(0.0));

      Y0.Randomize();
      X0.Randomize(1);
      X1.Randomize(1);
      A.PowAbsMultTranspose(power,Y0,X0);
      Aabs->MultTranspose(Y0,X1);
      X1 -= X0;

      error = X1.Norml2();

      REQUIRE(error == MFEM_Approx(0.0));

      delete Aabs;
      delete hdiv_coll;
      delete l2_coll;
   }
}

TEST_CASE("SparseMatrix printing", "[SparseMatrix]")
{
   // Create a test sparse matrix and print it using different methods
   // and compare the output with the reference one

   DenseMatrix dense(
   {
      {0.0, 4.0, 0.0},
      {5.0, 0.0, 1.0},
      {2.0, 0.0, 0.0}
   });

   const int width = dense.Width();
   const int height = dense.Height();
   int nonzero = 0;

   // Non-finalized matrix (LIL)
   SparseMatrix mat_lil(height, width);
   for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
      {
         if (dense(i,j) != 0.0)
         {
            mat_lil.Add(i, j, dense(i,j));
            nonzero++;
         }
      }

   // Finalized matrix (CSR)
   SparseMatrix mat_csr(mat_lil);
   mat_csr.Finalize();

   std::stringstream ss, ss_ref;

   SECTION("Print")
   {
      //assume print width >= matrix width
      ss_ref.str("");
      for (int i = 0; i < height; i++)
      {
         ss_ref << "[row " << i << "]\n";
         for (int j = width-1; j >=0 ; j--)
            if (dense(i,j) != 0.0)
            {
               ss_ref << " (" << j << "," << dense(i,j) << ")";
            }
         ss_ref << "\n";
      }

      ss.str("");
      mat_lil.Print(ss);
      REQUIRE(ss.str() == ss_ref.str());

      ss.str("");
      mat_csr.Print(ss);
      REQUIRE(ss.str() == ss_ref.str());
   }

   SECTION("PrintMatlab")
   {
      ss_ref.str("");
      ss_ref << "% size " << height << " " << width << "\n";
      ss_ref << "% Non Zeros " << nonzero << "\n";

      std::ios::fmtflags old_fmt = ss_ref.flags();
      ss_ref.setf(std::ios::scientific);
      std::streamsize old_prec = ss_ref.precision(14);

      for (int i = 0; i < height; i++)
         for (int j = width-1; j >=0 ; j--)
            if (dense(i,j) != 0.0)
            {
               ss_ref << i+1 << " " << j+1 << " " << dense(i,j) << "\n";
            }

      ss_ref << height << " " << width << " 0.0\n";
      ss_ref.precision(old_prec);
      ss_ref.flags(old_fmt);

      ss.str("");
      mat_lil.PrintMatlab(ss);
      REQUIRE(ss.str() == ss_ref.str());

      ss.str("");
      mat_csr.PrintMatlab(ss);
      REQUIRE(ss.str() == ss_ref.str());
   }

   SECTION("PrintMM")
   {
      ss_ref.str("");
      ss_ref << "%%MatrixMarket matrix coordinate real general" << '\n'
             << "% Generated by MFEM" << '\n';
      ss_ref << height << " " << width << " " << nonzero << "\n";

      std::ios::fmtflags old_fmt = ss_ref.flags();
      ss_ref.setf(std::ios::scientific);
      std::streamsize old_prec = ss_ref.precision(14);

      for (int i = 0; i < height; i++)
         for (int j = width-1; j >=0 ; j--)
            if (dense(i,j) != 0.0)
            {
               ss_ref << i+1 << " " << j+1 << " " << dense(i,j) << "\n";
            }

      ss_ref.precision(old_prec);
      ss_ref.flags(old_fmt);

      ss.str("");
      mat_lil.PrintMM(ss);
      REQUIRE(ss.str() == ss_ref.str());

      ss.str("");
      mat_csr.PrintMM(ss);
      REQUIRE(ss.str() == ss_ref.str());
   }

   SECTION("PrintCSR")
   {
      ss_ref.str("");
      ss_ref << height << "\n";

      Array<int> I(height+1);
      Array<int> J(nonzero);
      Vector A(nonzero);

      int idx = 0;
      for (int i = 0; i < height; i++)
      {
         I[i] = idx;
         for (int j = width-1; j >=0 ; j--)
            if (dense(i,j) != 0.0)
            {
               J[idx] = j;
               A[idx] = dense(i,j);
               idx++;
            }
      }
      I[height] = idx;

      for (int i = 0; i <= height; i++)
      {
         ss_ref << I[i]+1 << '\n';
      }

      for (int i = 0; i < I[height]; i++)
      {
         ss_ref << J[i]+1 << '\n';
      }

      for (int i = 0; i < I[height]; i++)
      {
         ss_ref << A[i] << '\n';
      }

      ss.str("");
      mat_csr.PrintCSR(ss);
      REQUIRE(ss.str() == ss_ref.str());
   }
}

} // namespace mfem
