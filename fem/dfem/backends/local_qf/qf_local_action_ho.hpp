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
   template<int Q>
   struct Shared
   {
      real_t M[Q][Q], B[Q][Q], G[Q][Q];
   };

   //////////////////////////////////////////////////////////////////
   template<int Q>
   struct Exclusive
   {
      ker::vd_regs3d_t<1, 3, Q> del; // dofs buffer for rank-1 gradient
      ker::vd_regs3d_t<3, 3, Q> mat; // dofs buffer for rank-2 gradient
   };

   //////////////////////////////////////////////////////////////////
   // Per-QP register layout
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
   template<int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1T> &, Exclusive<MQ1T> &,
                  const int, const int, const int, const int,
                  const real_t*, const XE_T &, ArgRegT &)
   {
      static_assert(sizeof(ArgRegT) == 0,
                    "LocalQFHOBackend::LoadValue is not implemented");
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadGradient(Shared<MQ1T> &s, Exclusive<MQ1T> &r,
                     const int e, const int d, const int q, const int,
                     const real_t *B, const real_t *G,
                     const XE_T &XE, ArgRegT &arg_reg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      if constexpr (ext_sz == 1)
      {
         ker::LoadDofs3d(e, d, XE, r.del);
         ker::Grad3d(d, q, s.M, s.B, s.G, r.del, arg_reg);
      }
      else if constexpr (ext_sz == 2)
      {
         ker::LoadDofs3d(e, d, XE, r.mat);
         ker::Grad3d(d, q, s.M, s.B, s.G, r.mat, arg_reg);
      }
      else
      {
         static_assert(ext_sz == 1 || ext_sz == 2,
                       "Unsupported gradient rank");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1T>&, Exclusive<MQ1T>&,
                   const int, const int, const int, const int,
                   const real_t*, const YE_T &, ArgRegT &)
   {
      static_assert(sizeof(ArgRegT) == 0,
                    "LocalQFHOBackend::WriteValue is not implemented");
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1T> &s, Exclusive<MQ1T> &r,
                      const int e, const int d, const int q, const int,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &arg_reg)
   {
      if constexpr (ext_sz == 1)
      {
         ker::LoadMatrix(d, q, B, s.B);
         ker::LoadMatrix(d, q, G, s.G);
         ker::GradTranspose3d(d, q, s.M, s.B, s.G, arg_reg, r.del);
         ker::WriteDofs3d(e, d, r.del, YE);
      }
      else { static_assert(ext_sz == 1, "Unsupported gradient rank"); }
   }
};

} // namespace mfem::future