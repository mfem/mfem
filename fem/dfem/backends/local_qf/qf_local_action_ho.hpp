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
#pragma once

#include "../../util.hpp" // for ThreadBlocks

#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;

#include "qf_local_types.hpp"

namespace mfem::future
{

struct LocalQFHOBackend
{
   static constexpr int MQ1 = 16;

   //////////////////////////////////////////////////////////////////
   template <int DIM> static
   ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_VERIFY(q1d <= MQ1, "q1d must be less than or equal to MQ1:" << MQ1);
      return {q1d, (DIM >= 2) ? q1d : 1, 1};
   }

   template <int T_Q1D> static inline
   constexpr int MAX_THREADS_PER_BLOCK() { return T_Q1D*T_Q1D; }

   //////////////////////////////////////////////////////////////////
   template<int MQ1>
   struct Shared
   {
      real_t M[MQ1][MQ1], B[MQ1][MQ1], G[MQ1][MQ1];
   };

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1T> &s,
                  const int e, const int d, const int q, const int,
                  const real_t *B, const XE_T &XE, ArgRegT &rarg)
   {
      static_assert(DIM == 3);
      static_assert(ext_sz == 1);
      ker::LoadMatrix(d, q, B, s.B);
      {
         ker::vd_regs3d_t<1, 3, MQ1T> dofs;
         ker::LoadDofs3d(e, d, XE, dofs);
         ker::Eval3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadGradient(Shared<MQ1T> &s,
                     const int e, const int d, const int q, const int,
                     const real_t *B, const real_t *G,
                     const XE_T &XE, ArgRegT &rarg)
   {
      static_assert(ext_sz == 1 || ext_sz == 2);
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      if constexpr (ext_sz == 1)
      {
         ker::vd_regs3d_t<1, 3, MQ1T> dofs;
         ker::LoadDofs3d(e, d, XE, dofs);
         ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
      if constexpr (ext_sz == 2)
      {
         ker::vd_regs3d_t<3, 3, MQ1T> dofs;
         ker::LoadDofs3d(e, d, XE, dofs);
         ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
   }

   //////////////////////////////////////////////////////////////////
   template<typename DecayT, int MQ1T>
   using QPReg = ho_qp_reg_for_decay_t<DecayT, MQ1T>;

   template<typename DecayT, int MQ1T>
   static MFEM_HOST_DEVICE inline
   auto qp_load(QPReg<DecayT, MQ1T> &reg, int qz, int qy, int qx)
   {
      return ho_input_qp_reg_as_arg_at<DecayT, MQ1T>(reg, qz, qy, qx);
   }

   template<typename DecayT, int MQ1T>
   static MFEM_HOST_DEVICE inline
   void qp_store(QPReg<DecayT, MQ1T> &reg, int qz, int qy, int qx,
                 const DecayT &out)
   {
      ho_output_qp_reg_assign_at<DecayT, MQ1T>(reg, qz, qy, qx, out);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1T>&,
                   const int, const int, const int, const int,
                   const real_t*, const YE_T &, ArgRegT &)
   {
      static_assert(sizeof(ArgRegT) == 0,
                    "LocalQFHOBackend::WriteValue is not implemented");
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1> &s,
                      const int e, const int d, const int q, const int,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &rarg)
   {
      if constexpr (ext_sz == 1)
      {
         ker::LoadMatrix(d, q, B, s.B);
         ker::LoadMatrix(d, q, G, s.G);
         ker::vd_regs3d_t<1, 3, MQ1> dofs;
         ker::GradTranspose3d(d, q, s.M, s.B, s.G, rarg, dofs);
         ker::WriteDofs3d(e, d, dofs, YE);
      }
      else { static_assert(ext_sz == 1, "Unsupported gradient rank"); }
   }
};

} // namespace mfem::future