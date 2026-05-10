#pragma once

#include "fem/kernels3d.hpp"
namespace low = mfem::kernels::internal::low;

namespace mfem::future
{

struct LocalQFLOBackend
{
   static constexpr int MQ1 = 8;

   //////////////////////////////////////////////////////////////////
   static MFEM_SHARED real_t sm[2][MQ1][MQ1][MQ1][3];
   static MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];

   //////////////////////////////////////////////////////////////////
   template<typename ArgRegT, typename XE_T>
   static auto LoadValue(const int e,
                         const int d,
                         const int q,
                         [[maybe_unused]] const int q1d,
                         const real_t *B,
                         const XE_T &XE,
                         ArgRegT &arg_reg)
   {
      low::LoadMatrix(d, q, B, sB);
      low::LoadDofs3d(e, d, XE, sm[0]);
      low::Eval3d(d, q, sB, sm[0], sm[1], arg_reg);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename XE_T>
   static auto LoadGradient(const int e,
                            const int d,
                            const int q,
                            [[maybe_unused]] const int q1d,
                            const real_t *B,
                            const real_t *G,
                            const XE_T &XE,
                            ArgRegT &arg_reg)
   {
      low::LoadMatrix(d, q, B, sB);
      low::LoadMatrix(d, q, G, sG);
      if constexpr (ext_sz == 1)
      {
         low::LoadDofs3d(e, d, XE, sm[0]);
         low::Grad3d(d, q, sB, sG, sm[0], sm[1], arg_reg);
      }
      else if constexpr (ext_sz == 2)
      {
         for (int c = 0; c < DIM; c++)
         {
            low::LoadDofs3d(e, d, c, XE, sm[0]);
            low::VectorGrad3d(d, q, c, sB, sG, sm[0], sm[1], arg_reg);
         }
      }
      else
      {
         static_assert(false, "Unsupported gradient rank");
      }
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename YE_T>
   static auto WriteValue(const int e,
                          const int d,
                          const int q,
                          [[maybe_unused]] const int q1d,
                          const real_t *B,
                          const YE_T &YE,
                          ArgRegT &arg_reg)
   {
      low::LoadMatrix(d, q, B, sB);
      low::EvalTranspose3d(d, q, sB, arg_reg, sm[1], sm[0]);
      low::WriteDofs3d(d, 0, e, arg_reg, YE);
   }

   //////////////////////////////////////////////////////////////////
   template<int DIM, int ext_sz, typename ArgRegT, typename YE_T>
   static auto WriteGradient(const int e,
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
         low::LoadMatrix(d, q, B, sB);
         low::LoadMatrix(d, q, G, sG);
         low::GradTranspose3d(d, q, sB, sG, arg_reg, sm[1], sm[0]);
         low::WriteDofs3d(d, 0, e, arg_reg, YE);
      }
      else { static_assert(false, "Unsupported gradient rank"); }
   }
};

MFEM_SHARED real_t
LocalQFLOBackend::sm[2][LocalQFLOBackend::MQ1][LocalQFLOBackend::MQ1][LocalQFLOBackend::MQ1][3];
MFEM_SHARED real_t
LocalQFLOBackend::sB[LocalQFLOBackend::MQ1][LocalQFLOBackend::MQ1];
MFEM_SHARED real_t
LocalQFLOBackend::sG[LocalQFLOBackend::MQ1][LocalQFLOBackend::MQ1];

} // namespace mfem::future