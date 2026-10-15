// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

TEST_CASE("ComplexDenseMatrixData storage and arithmetic",
          "[ComplexDenseMatrixData]")
{
   DenseMatrix real_part({{1.0, 2.0}, {3.0, 4.0}});
   DenseMatrix imag_part({{-5.0, 6.0}, {7.0, -8.0}});

   ComplexDenseMatrixData matrix(2);
   matrix.UseDevice(true);
   matrix.Set(real_part, imag_part);

   const DenseMatrix &real_values = matrix.real();
   const DenseMatrix &imag_values = matrix.imag();
   for (int j = 0; j < matrix.Width(); j++)
   {
      for (int i = 0; i < matrix.Height(); i++)
      {
         REQUIRE(real_values(i, j) == MFEM_Approx(real_part(i, j)));
         REQUIRE(imag_values(i, j) == MFEM_Approx(imag_part(i, j)));
      }
   }

   ComplexDenseMatrixData copy(matrix);
   copy += real_part;
   copy -= real_part;
   copy *= complex_t(2.0, -1.0);
   copy *= complex_t(2.0, 1.0) / 5.0;
   const complex_t *copy_data = copy.HostRead();
   const complex_t *matrix_data = matrix.HostRead();
   for (int j = 0; j < matrix.Width(); j++)
   {
      for (int i = 0; i < matrix.Height(); i++)
      {
         const int k = i + j * matrix.Height();
         REQUIRE(copy_data[k].real() == MFEM_Approx(matrix_data[k].real()));
         REQUIRE(copy_data[k].imag() == MFEM_Approx(matrix_data[k].imag()));
      }
   }

   ComplexDenseMatrixData assigned;
   assigned = matrix;
   assigned += matrix;
   assigned -= matrix;
   const complex_t *assigned_data = assigned.HostRead();
   matrix_data = matrix.HostRead();
   for (int j = 0; j < matrix.Width(); j++)
   {
      for (int i = 0; i < matrix.Height(); i++)
      {
         const int k = i + j * matrix.Height();
         REQUIRE(assigned_data[k].real() == MFEM_Approx(matrix_data[k].real()));
         REQUIRE(assigned_data[k].imag() == MFEM_Approx(matrix_data[k].imag()));
      }
   }

   ComplexDenseMatrixData moved(std::move(assigned));
   const complex_t *moved_data = moved.HostRead();
   REQUIRE(moved.Height() == 2);
   REQUIRE(moved.Width() == 2);
   REQUIRE(moved_data[1].real() == MFEM_Approx(3.0));
   REQUIRE(moved_data[1].imag() == MFEM_Approx(7.0));

   ComplexDenseMatrixData move_assigned;
   move_assigned = std::move(moved);
   const complex_t *move_assigned_data = move_assigned.HostRead();
   REQUIRE(move_assigned.Height() == 2);
   REQUIRE(move_assigned.Width() == 2);
   REQUIRE(move_assigned_data[2].real() == MFEM_Approx(2.0));
   REQUIRE(move_assigned_data[2].imag() == MFEM_Approx(6.0));

   complex_t external_data[2] = {complex_t(1.0, 2.0), complex_t(3.0, 4.0)};
   ComplexDenseMatrixData external(external_data, 2, 1);
   external *= 2.0;
   REQUIRE(external_data[0].real() == MFEM_Approx(2.0));
   REQUIRE(external_data[0].imag() == MFEM_Approx(4.0));
   REQUIRE(external_data[1].real() == MFEM_Approx(6.0));
   REQUIRE(external_data[1].imag() == MFEM_Approx(8.0));
}
