#pragma once

#include "../../util.hpp" // for ThreadBlocks

#include "fem/kernels3d.hpp"
namespace low = mfem::kernels::internal::low;

#include "qf_local_types.hpp"

namespace mfem::future
{

struct LocalQFLOBackend
{
   static constexpr int MQ1 = 8;

   //////////////////////////////////////////////////////////////////
   template <int DIM> static
   inline ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_VERIFY(q1d <= MQ1, "q1d must be less than or equal to MQ1:" << MQ1);
      return {q1d, (DIM >= 2) ? q1d : 1, (DIM >= 3) ? q1d : 1};
   }

   template <int T_Q1D> static inline
   constexpr int MAX_THREADS_PER_BLOCK() { return T_Q1D*T_Q1D*T_Q1D; }

   //////////////////////////////////////////////////////////////////
   template<int Q>
   struct Shared
   {
      real_t M[2][Q][Q][Q][3];
      real_t B[Q][Q], G[Q][Q];
   };

   //////////////////////////////////////////////////////////////////
   template<int Q>
   struct Exclusive { };

   //////////////////////////////////////////////////////////////////
   // Per-QP register layout
   template<typename DecayT, int MQ1T>
   using QPReg = lo_qp_reg_for_decay_t<DecayT, MQ1T>;

   template<typename DecayT, int MQ1T>
   static MFEM_HOST_DEVICE inline
   auto qp_load(QPReg<DecayT, MQ1T> &reg, int qz, int qy, int qx)
   {
      return lo_input_qp_reg_as_arg_at<DecayT, MQ1T>(reg, qz, qy, qx);
   }

   template<typename DecayT, int MQ1T>
   static MFEM_HOST_DEVICE inline
   void qp_store(QPReg<DecayT, MQ1T> &reg, int qz, int qy, int qx,
                 const DecayT &out)
   {
      output_qp_reg_assign_at<DecayT, MQ1T>(reg, qz, qy, qx, out);
   }

   //////////////////////////////////////////////////////////////////
   template<int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1T> &s, Exclusive<MQ1T>&,
                  const int e, const int d, const int q, const int,
                  const real_t *B, const XE_T &XE, ArgRegT &arg_reg)
   {
      low::LoadMatrix<MQ1T>(d, q, B, s.B);
      low::LoadDofs3d(e, d, XE, s.M[0]);
      low::Eval3d(d, q, s.B, s.M[0], s.M[1], arg_reg);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadGradient(Shared<MQ1T> &s, Exclusive<MQ1T> &,
                     const int e, const int d, const int q, const int,
                     const real_t *B, const real_t *G,
                     const XE_T &XE, ArgRegT &arg_reg)
   {
      low::LoadMatrix<MQ1T>(d, q, B, s.B);
      low::LoadMatrix<MQ1T>(d, q, G, s.G);
      if constexpr (ext_sz == 1)
      {
         low::LoadDofs3d(e, d, XE, s.M[0]);
         low::Grad3d(d, q, s.B, s.G, s.M[0], s.M[1], arg_reg);
      }
      else if constexpr (ext_sz == 2)
      {
         for (int c = 0; c < DIM; c++)
         {
            low::LoadDofs3d(e, d, c, XE, s.M[0]);
            low::VectorGrad3d(d, q, c, s.B, s.G, s.M[0], s.M[1], arg_reg);
         }
      }
      else
      {
         static_assert(ext_sz == 1 || ext_sz == 2, "Unsupported gradient rank");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1T> &s, Exclusive<MQ1T>&,
                   const int e, const int d, const int q, const int,
                   const real_t *B, const YE_T &YE, ArgRegT &arg_reg)
   {
      MFEM_CONTRACT_VAR(ext_sz);
      low::LoadMatrix<MQ1T>(d, q, B, s.B);
      low::EvalTranspose3d(d, q, s.B, arg_reg, s.M[1], s.M[0]);
      low::WriteDofs3d(d, 0, e, arg_reg, YE);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1T> &s, Exclusive<MQ1T> &,
                      const int e, const int d, const int q, const int,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &arg_reg)
   {
      if constexpr (ext_sz == 1)
      {
         low::LoadMatrix<MQ1T>(d, q, B, s.B);
         low::LoadMatrix<MQ1T>(d, q, G, s.G);
         low::GradTranspose3d(d, q, s.B, s.G, arg_reg, s.M[1], s.M[0]);
         low::WriteDofs3d(d, 0, e, arg_reg, YE);
      }
      else { static_assert(ext_sz == 1, "Unsupported gradient rank"); }
   }
};

} // namespace mfem::future