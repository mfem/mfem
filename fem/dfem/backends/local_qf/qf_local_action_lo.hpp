#pragma once

#include "fem/kernels3d.hpp"
namespace low = mfem::kernels::internal::low;

namespace mfem::future
{

struct LocalQFLOBackend
{
   static constexpr int MQ1 = 8;

   //////////////////////////////////////////////////////////////////
   // Shared memory used by the load/write helpers below
   struct Scratch
   {
      real_t M[2][MQ1][MQ1][MQ1][3];
      real_t B[MQ1][MQ1], G[MQ1][MQ1];
   };

   //////////////////////////////////////////////////////////////////
   template<typename ArgRegT, typename XE_T>
   static auto LoadValue(Scratch &s,
                         const int e,
                         const int d,
                         const int q,
                         [[maybe_unused]] const int q1d,
                         const real_t *B,
                         const XE_T &XE,
                         ArgRegT &arg_reg)
   {
      low::LoadMatrix(d, q, B, s.B);
      low::LoadDofs3d(e, d, XE, s.M[0]);
      low::Eval3d(d, q, s.B, s.M[0], s.M[1], arg_reg);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename XE_T>
   static auto LoadGradient(Scratch &s,
                            const int e,
                            const int d,
                            const int q,
                            [[maybe_unused]] const int q1d,
                            const real_t *B,
                            const real_t *G,
                            const XE_T &XE,
                            ArgRegT &arg_reg)
   {
      low::LoadMatrix(d, q, B, s.B);
      low::LoadMatrix(d, q, G, s.G);
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
         static_assert(false, "Unsupported gradient rank");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename YE_T>
   static auto WriteValue(Scratch &s,
                          const int e,
                          const int d,
                          const int q,
                          [[maybe_unused]] const int q1d,
                          const real_t *B,
                          const YE_T &YE,
                          ArgRegT &arg_reg)
   {
      low::LoadMatrix(d, q, B, s.B);
      low::EvalTranspose3d(d, q, s.B, arg_reg, s.M[1], s.M[0]);
      low::WriteDofs3d(d, 0, e, arg_reg, YE);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename YE_T>
   static auto WriteGradient(Scratch &s,
                             const int e,
                             const int d,
                             const int q,
                             [[maybe_unused]] const int q1d,
                             const real_t *B,
                             const real_t *G,
                             YE_T &YE,
                             ArgRegT &arg_reg)
   {
      if constexpr (ext_sz == 1)
      {
         low::LoadMatrix(d, q, B, s.B);
         low::LoadMatrix(d, q, G, s.G);
         low::GradTranspose3d(d, q, s.B, s.G, arg_reg, s.M[1], s.M[0]);
         low::WriteDofs3d(d, 0, e, arg_reg, YE);
      }
      else { static_assert(false, "Unsupported gradient rank"); }
   }
};

} // namespace mfem::future