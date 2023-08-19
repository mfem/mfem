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

// sycl_ext_oneapi_work_group_local: this extension defines a
// sycl::ext::oneapi::experimental::work_group_local class template
// with behavior inspired by the C++ thread_local keyword and
// the CUDA __shared__ keyword.

template <typename T>
inline T& mfem_local_memory()
{
   static_assert(std::is_trivial_v<T>, "T should be trivial!");
#ifdef __SYCL_DEVICE_ONLY__
   __attribute__((opencl_local)) std::uint8_t *lmem =
      __sycl_allocateLocalMemory(sizeof(T), alignof(T));
   return reinterpret_cast<__attribute__((opencl_local)) T&>(*lmem);
#else
   //std::uint8_t *lmem = __mfem_allocateLocalMemory(sizeof(T), alignof(T));
   return *new T;
#endif
}


// sycl::group_local_memory
template <typename T, typename... Args>
sycl::multi_ptr<T, sycl::access::address_space::local_space>
inline mfem_local_memory(Args &&...args)
{
#ifdef __SYCL_DEVICE_ONLY__
   __attribute__((opencl_local)) std::uint8_t *AllocatedMem =
      __sycl_allocateLocalMemory(sizeof(T), alignof(T));

   sycl::id<3> Id = __spirv::initLocalInvocationId<3, sycl::id<3>>();
   if (Id == sycl::id<3>(0, 0, 0))
   {
      new (AllocatedMem) T(std::forward<Args>(args)...);
   }
   sycl::detail::workGroupBarrier();
   return reinterpret_cast<__attribute__((opencl_local)) T *>(AllocatedMem);
#else
   [&args...] {}();
   return new T(std::forward<Args>(args)...);
#endif
}


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
      // Device("cpu")
      const sycl::range<3> grid(1, 1, N), group(1, 1, 1);
#endif
      //sycl::accessor<double, 1, sycl::access_mode::read_write, sycl::access::target::local>
      //slm(sycl::range<1>(1024), cgh);
      h.parallel_for(sycl::nd_range<3>(grid,group), [=](sycl::nd_item<3> itm)
      {
         int k =
            itm.get_group(2)*itm.get_local_range().get(0) + itm.get_local_id(0);
         if (k >= N) { return; }

         sycl_out << "✅ ";

         /// ASSERT ARE NOT WORKING ///
#ifdef __SYCL_DEVICE_ONLY__
         //#warning SYCL-[C|G]PU

         constexpr float M_PIf = static_cast<float>(M_PI);

         __attribute__((opencl_local)) float *D=
            reinterpret_cast<__attribute__((opencl_local)) float*>
            (__sycl_allocateLocalMemory(sizeof(float)*8,32));
         D[0] = M_PIf;
         if (D[0] != M_PIf) { sycl_out << "❌"; }

         __attribute__((opencl_local)) float (&H)[4] =
            *reinterpret_cast<__attribute__((opencl_local)) float(*)[4]>
            (__sycl_allocateLocalMemory(sizeof(float[4]),32));
         H[0] = H[1] = H[2] = H[3] = M_PIf;
         if (H[0] != M_PIf || H[1] != M_PIf) { sycl_out << "❌"; }
         if (H[2] != M_PIf || H[3] != M_PIf) { sycl_out << "❌"; }

         auto J = mfem_local_memory<float[2][3]>();
         J[1][2] = M_PIf;
         if (J[1][2] != M_PIf) { sycl_out << "❌"; }

         __attribute__((opencl_local)) tensor<float,2,3> &tensor_2_3 =
            *reinterpret_cast<__attribute__((opencl_local)) tensor<float,2,3>*>
            (__sycl_allocateLocalMemory(sizeof(tensor<float,2,3>),32));
         tensor_2_3[1][2] = M_PIf;
         if (tensor_2_3[1][2] != M_PIf) { sycl_out << "❌"; }

         auto int_ptr =
            *sycl::group_local_memory<int[64]>(itm.get_group());
         int_ptr[0] = 1234;
         if (int_ptr[0] != 1234) { sycl_out << "❌"; }

         auto fp_ptr =
            *sycl::group_local_memory<float[64]>(itm.get_group());
         fp_ptr[0] = M_PIf;
         if (fp_ptr[0] != M_PIf) { sycl_out << "❌"; }

         auto GD =
            *sycl::group_local_memory<float[2][3][3]>(itm.get_group());
         GD[0][0][1] = 1.234f;
         GD[0][2][0] = 1.234f;
         GD[1][2][2] = 1.234f;
         if (GD[0][0][1] != 1.234f) { sycl_out << "❌"; }
         if (GD[0][2][0] != 1.234f) { sycl_out << "❌"; }
         if (GD[1][2][2] != 1.234f) { sycl_out << "❌"; }

         // does not work
         /*auto tensor_2_3 =
            *sycl::group_local_memory_for_overwrite<tensor<float,2,3>>(itm.get_group());
         tensor_2_3[1][2] = M_PIf;
         if (tensor_2_3[1][2] != M_PI) { sycl_out << "❌"; }*/

         auto mlm_f8 = mfem_local_memory<float[8]>();
         mlm_f8[4] = M_PIf;
         if (mlm_f8[4] != M_PIf) { sycl_out << "❌"; }

         sycl_out << "SYCL-[C|G]PU" << "\n";

         // proposed, but not available
         // sycl::ext::oneapi::experimental::work_group_local<int[16]> program_scope_array;
#else // SYCL_HOST:
         //#warning SYCL_HOST
         sycl_out << "SYCL_HOST" << "\n";

         // MFEM_SHARED tensor<double,2,3> tensor_2_3_s;
         // MFEM_SHARED(tensor<double,2,3>, tensor_2_3_s);
         auto tensor_2_3 = mfem_local_memory<tensor<double,2,3>>();
         tensor_2_3[0][1] = M_PI;
         if (tensor_2_3[0][1] != M_PI) { sycl_out << "❌"; }

         sycl_out << tensor_2_3[0][1] << "\n";
         sycl_out << "SYCL-HOST" << "\n";
#endif
      });
   });
   Q.wait();
   //Q.wait_and_throw();
} // Pure SYCL, LocalMemory

#endif // MFEM_USE_SYCL
