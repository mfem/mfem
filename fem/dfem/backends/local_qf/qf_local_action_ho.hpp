#pragma once

#include "../../util.hpp" // for ThreadBlocks

#include "general/forall.hpp"
#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;

namespace mfem::future
{

struct LocalQFHOBackend
{
   static constexpr int MQ1 = 8;

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
      // ker::v_regs3d_t<1, Q> val[2];
      ker::vd_regs3d_t<1, 3, Q> del[2];
      ker::vd_regs3d_t<3, 3, Q> mat[2];
   };

   //////////////////////////////////////////////////////////////////
   template<int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadValue(Shared<MQ1T> &, Exclusive<MQ1T> &,
                  const int, const int, const int, const int,
                  const real_t*, const XE_T &, ArgRegT &)
   {
      static_assert(false);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE
   void LoadGradient(Shared<MQ1T> &s, Exclusive<MQ1T> &r,
                     const int e, const int d, const int q, const int q1d,
                     const real_t *B, const real_t *G,
                     const XE_T &XE, ArgRegT &arg_reg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      if constexpr (ext_sz == 1)
      {
         ker::LoadDofs3d(e, d, XE, r.del[0]);
         ker::Grad3d(d, q, s.M, s.B, s.G, r.del[0], r.del[1]);
         for (int qz = 0; qz < q1d; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
               {
                  MFEM_UNROLL(DIM)
                  for (int dd = 0; dd < DIM; ++dd)
                  {
                     arg_reg[qz][qy][qx][dd] =
                        r.del[1][0][dd][qz][qy][qx];
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else if constexpr (ext_sz == 2)
      {
         ker::LoadDofs3d(e, d, XE, r.mat[0]);
         ker::Grad3d(d, q, s.M, s.B, s.G, r.mat[0], r.mat[1]);
         for (int qz = 0; qz < q1d; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
               {
                  MFEM_UNROLL(DIM)
                  for (int ii = 0; ii < DIM; ++ii)
                  {
                     MFEM_UNROLL(DIM)
                     for (int jj = 0; jj < DIM; ++jj)
                     {
                        arg_reg[qz][qy][qx][ii][jj] =
                           r.mat[1][ii][jj][qz][qy][qx];
                     }
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else
      {
         static_assert(false, "Unsupported gradient rank");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteValue(Shared<MQ1T>&, Exclusive<MQ1T>&,
                   const int, const int, const int, const int,
                   const real_t*, const YE_T &, ArgRegT &)
   {
      static_assert(false);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, int MQ1T, typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE
   void WriteGradient(Shared<MQ1T> &s, Exclusive<MQ1T> &r,
                      const int e, const int d, const int q, const int q1d,
                      const real_t *B, const real_t *G,
                      YE_T &YE, ArgRegT &arg_reg)
   {
      if constexpr (ext_sz == 1)
      {
         ker::LoadMatrix(d, q, B, s.B);
         ker::LoadMatrix(d, q, G, s.G);
         for (int qz = 0; qz < q1d; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,q1d)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,q1d)
               {
                  MFEM_UNROLL(DIM)
                  for (int dd = 0; dd < DIM; ++dd)
                  {
                     r.del[0][0][dd][qz][qy][qx] =
                        arg_reg[qz][qy][qx][dd];
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
         ker::GradTranspose3d(d, q, s.M, s.B, s.G, r.del[0], r.del[1]);
         ker::WriteDofs3d(e, d, r.del[1], YE);
      }
      else { static_assert(false, "Unsupported gradient rank"); }
   }
};

} // namespace mfem::future