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

using namespace mfem;
using namespace mfem::future;

TEST_CASE("Tensor array generic tests", "[TensorArray][GPU]")
{
   const int size = 10;
   static constexpr int dim = 3; // 'static' is needed for MSVC

   Vector vec_data(size*dim), mat_data(size*dim*dim);
   vec_data.UseDevice(true);
   vec_data = 1_r;
   mat_data.UseDevice(true);
   mat_data = 1_r;

   auto vec = make_tensor_array<dim>(vec_data.Write(), size);
   auto mat = make_tensor_array<dim,dim>(mat_data.Read(), size);

   mfem::forall(size, [=] MFEM_HOST_DEVICE (int i)
   {
      // Test the usage of tensor array methods inside a kernel
      auto mat_t = make_tensor_array<dim,dim>(mat.data(), size);
      mat_t.set_layout({0, 2, 1});

      [[maybe_unused]] auto v_rank = vec.rank();
      [[maybe_unused]] auto v_size_0 = vec.size(0);
      [[maybe_unused]] auto v_total_size = vec.total_size();

      [[maybe_unused]] auto v_tensor_rank = vec.tensor_rank();
      [[maybe_unused]] auto v_tensor_size_0 = vec.tensor_size(0);
      [[maybe_unused]] auto v_total_tensor_size = vec.total_tensor_size();

      [[maybe_unused]] auto v_get_tensor_i = vec.get_tensor(i);
      [[maybe_unused]] auto m_get_tensor_i = mat.get_tensor(i);

      // vec(i)   ->   vec.get_accessor(i)
      // mat(i)   ->   mat.get_tensor(i)
      // mat_t(i) -> mat_t.get_tensor(i)
      auto ones = make_tensor<dim>([](int) { return 1_r; });
      vec(i) = (mat(i) + mat_t(i)) * ones;

      // tensor_accessor methods
      auto v_i = vec(i);
      v_i(0) *= 2;
   });
}
