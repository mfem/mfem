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

#include "mfem.hpp"
using namespace mfem;

#include "unit_tests.hpp"

#ifdef MFEM_USE_SYCL

#undef NDEBUG
#define SYCL_FALLBACK_ASSERT 1
#include <cassert>

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 191
#include "general/debug.hpp"

#include "linalg/tensor.hpp"
using mfem::internal::tensor;

void sycl_lmem()
{
   dbg();
   auto Q = Sycl::Queue();

   constexpr int N = 2, X = 3, Y = 4, BZ = 1, G = 0;

#ifdef __SYCL_DEVICE_ONLY__
   dbg("Local Memory Size: %d",
       Q.get_device().get_info<sycl::info::device::local_mem_size>());
#endif

   Q.submit([&](auto &h) // 2D
   {
      sycl::stream sycl_out(65536, 512, h);
#if defined(__SYCL_DEVICE_ONLY__)
      const int L = static_cast<int>(std::ceil(std::sqrt((N+BZ-1)/BZ)));
      const sycl::range<3> grid(L*BZ, L*Y, L*X), group(BZ, Y, X);
#else // HOST
      MFEM_CONTRACT_VAR(X);
      MFEM_CONTRACT_VAR(Y);
      MFEM_CONTRACT_VAR(BZ);
      MFEM_CONTRACT_VAR(G);
      const sycl::range<3> grid(1, 1, N), group(1, 1, 1);
#endif
      h.parallel_for(sycl::nd_range<3>(grid,group), [=](sycl::nd_item<3> itm)
      {
         int k =
            itm.get_group(2)*itm.get_local_range().get(0) + itm.get_local_id(0);
         if (k >= N) { return; }

         /// ASSERT ARE NOT WORKING ///
#ifdef __SYCL_DEVICE_ONLY__
         sycl_out << "SYCL-[C|G]PU" << "\n";

         auto D = mfem_shared<double[8]>();
         D[0] = M_PI;
         if (D[0] != M_PI) { sycl_out << "❌"; }

         auto H = mfem_shared<double[4]>();
         H[0] = H[1] = H[2] = H[3] = M_PI;
         if (H[0] != M_PI || H[1] != M_PI) { sycl_out << "❌"; }
         if (H[2] != M_PI || H[3] != M_PI) { sycl_out << "❌"; }

         auto J = mfem_shared<double[2][3]>();
         J[1][2] = M_PI;
         if (J[1][2] != M_PI) { sycl_out << "❌"; }

         auto tensor_2_3 = mfem_shared<tensor<double,2,3>>();
         tensor_2_3[1][2] = M_PI;
         if (tensor_2_3[1][2] != M_PI) { sycl_out << "❌"; }

         auto int_ptr = mfem_shared<int[64]>();
         int_ptr[0] = 1234;
         if (int_ptr[0] != 1234) { sycl_out << "❌"; }

         auto fp_ptr = mfem_shared<double[64]>();
         fp_ptr[0] = M_PI;
         if (fp_ptr[0] != M_PI) { sycl_out << "❌"; }

         auto GD = mfem_shared<double[2][3][3]>();
         GD[0][0][1] = 1.234f;
         GD[0][2][0] = 1.234f;
         GD[1][2][2] = 1.234f;
         if (GD[0][0][1] != 1.234f) { sycl_out << "❌"; }
         if (GD[0][2][0] != 1.234f) { sycl_out << "❌"; }
         if (GD[1][2][2] != 1.234f) { sycl_out << "❌"; }

         auto mlm_f8 = mfem_shared<double[8]>();
         mlm_f8[4] = M_PI;
         if (mlm_f8[4] != M_PI) { sycl_out << "❌"; }

         // proposed, but not yet available
         // https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_local_memory.asciidoc
         // sycl::ext::oneapi::experimental::work_group_local<int[16]> program_scope_array;
#else // SYCL_HOST:
         sycl_out << "SYCL_HOST" << "\n";

         auto J = mfem_shared<tensor<double,2,3>>();
         J[1][2] = M_PI;
         if (J[1][2] != M_PI) { sycl_out << "❌"; }

         auto tensor_2_3 = mfem_shared<tensor<double,2,3>>();
         tensor_2_3[0][1] = M_PI;
         if (tensor_2_3[0][1] != M_PI) { sycl_out << "❌"; }

         sycl_out << tensor_2_3[0][1] << "\n";
         sycl_out << "SYCL-HOST" << "\n";
#endif
      });
   });
   Q.wait();
} // Pure SYCL, LocalMemory

#endif // MFEM_USE_SYCL
