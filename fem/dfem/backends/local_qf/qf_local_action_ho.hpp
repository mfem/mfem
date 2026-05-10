#pragma once

#include "general/forall.hpp"
#include "fem/kernels.hpp"
namespace ker = mfem::kernels::internal;

namespace mfem::future
{

struct LocalQFHOBackend
{
   static constexpr int MQ1 = 8;

   //////////////////////////////////////////////////////////////////
   // Shared memory used by the load/write helpers below
   struct Scratch
   {
      real_t M[MQ1][MQ1], B[MQ1][MQ1], G[MQ1][MQ1];
   };

   //////////////////////////////////////////////////////////////////
   static ker::v_regs3d_t<1, MQ1> val_reg[2];
   static ker::vd_regs3d_t<1, 3, MQ1> del_reg[2];
   static ker::vd_regs3d_t<3, 3, MQ1> mat_reg[2];

   //////////////////////////////////////////////////////////////////
   template<typename ArgRegT, typename XE_T>
   static auto LoadValue([[maybe_unused]] Scratch &s,
                         [[maybe_unused]] const int e,
                         [[maybe_unused]] const int d,
                         [[maybe_unused]] const int q,
                         [[maybe_unused]] const int q1d,
                         [[maybe_unused]] const real_t *B,
                         [[maybe_unused]] const XE_T &XE,
                         [[maybe_unused]] ArgRegT &arg_reg)
   {
      static_assert(false);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename XE_T>
   static auto LoadGradient(Scratch &s,
                            const int e,
                            const int d,
                            const int q,
                            const int q1d,
                            const real_t *B,
                            const real_t *G,
                            const XE_T &XE,
                            ArgRegT &arg_reg)
   {
      ker::LoadMatrix(d, q, B, s.B);
      ker::LoadMatrix(d, q, G, s.G);
      if constexpr (ext_sz == 1)
      {
         ker::LoadDofs3d(e, d, XE, del_reg[0]);
         ker::Grad3d(d, q, s.M, s.B, s.G, del_reg[0], del_reg[1]);
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
                        del_reg[1][0][dd][qz][qy][qx];
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      else if constexpr (ext_sz == 2)
      {
         ker::LoadDofs3d(e, d, XE, mat_reg[0]);
         ker::Grad3d(d, q, s.M, s.B, s.G, mat_reg[0], mat_reg[1]);
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
                           mat_reg[1][ii][jj][qz][qy][qx];
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
   template<int DIM, int ext_sz, typename ArgRegT, typename YE_T>
   static auto WriteValue([[maybe_unused]] Scratch &s,
                          [[maybe_unused]] const int e,
                          [[maybe_unused]] const int d,
                          [[maybe_unused]] const int q,
                          [[maybe_unused]] const int q1d,
                          [[maybe_unused]] const real_t *B,
                          [[maybe_unused]] const YE_T &YE,
                          [[maybe_unused]] ArgRegT &arg_reg)
   {
      static_assert(false);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename YE_T>
   static auto WriteGradient(Scratch &s,
                             const int e,
                             const int d,
                             const int q,
                             const int q1d,
                             const real_t *B,
                             const real_t *G,
                             YE_T &YE,
                             ArgRegT &arg_reg)
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
                     del_reg[0][0][dd][qz][qy][qx] =
                        arg_reg[qz][qy][qx][dd];
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
         ker::GradTranspose3d(d, q, s.M, s.B, s.G, del_reg[0], del_reg[1]);
         ker::WriteDofs3d(e, d, del_reg[1], YE);
      }
      else { static_assert(false, "Unsupported gradient rank"); }
   }
};

ker::v_regs3d_t<1, LocalQFHOBackend::MQ1> LocalQFHOBackend::val_reg[2];
ker::vd_regs3d_t<1, 3, LocalQFHOBackend::MQ1> LocalQFHOBackend::del_reg[2];
ker::vd_regs3d_t<3, 3, LocalQFHOBackend::MQ1> LocalQFHOBackend::mat_reg[2];

} // namespace mfem::future