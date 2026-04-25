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

#include "../kernels.hpp"
#include "../nonlininteg.hpp"
#include "../../general/forall.hpp"
#include "../ceed/integrators/nlconvection/nlconvection.hpp"

namespace mfem
{

void VectorConvectionNLFIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &Tr = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Tr);

   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAVectorConvectionNLIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PAVectorConvectionNLFIntegrator(fes, *ir, Q);
      }
      return;
   }

   ne = mesh->GetNE();
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   pa_adj.SetSize(ne * nq * dim * dim, Device::GetMemoryType());
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   d1d = maps->ndof;
   q1d = maps->nqpt;

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs);

   if (auto cc = dynamic_cast<ConstantCoefficient *>(Q))
   {
      coeff.SetConstant(cc->constant);
   }
   else if (Q)
   {
      coeff.Project(*Q);
   }
   else
   {
      coeff.SetConstant(1.0);
   }

   const auto w_r = ir->GetWeights().Read();

   if (dim == 2)
   {
      const int Q1D = q1d;
      constexpr int VDIM = 2, DIM = 2;
      const auto W = Reshape(w_r, Q1D, Q1D);
      const auto C = Reshape(coeff.Read(), Q1D, Q1D, ne);
      const auto J = Reshape(geom->J.Read(), Q1D, Q1D, VDIM, DIM, ne);
      auto A = Reshape(pa_adj.Write(), VDIM, DIM, Q1D, Q1D, ne);

      mfem::forall_2D(ne, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t J11 = J(qx, qy, 0, 0, e), J12 = J(qx, qy, 0, 1, e);
               const real_t J21 = J(qx, qy, 1, 0, e), J22 = J(qx, qy, 1, 1, e);
               // adj(J)
               const real_t A11 = +J22, A12 = -J12;
               const real_t A21 = -J21, A22 = +J11;
               // Store w * coeff * adj(J)
               const real_t w = W(qx, qy);
               const real_t c = C(qx, qy, e);
               A(0, 0, qx, qy, e) = w * c * A11;
               A(1, 0, qx, qy, e) = w * c * A12;
               A(0, 1, qx, qy, e) = w * c * A21;
               A(1, 1, qx, qy, e) = w * c * A22;
            }
         }
      });
   }
   else if (dim == 3)
   {
      const int Q1D = q1d;
      constexpr int VDIM = 3, DIM = 3;
      const auto W = Reshape(w_r, Q1D, Q1D, Q1D);
      const auto C = Reshape(coeff.Read(), Q1D, Q1D, Q1D, ne);
      const auto J = Reshape(geom->J.Read(), Q1D, Q1D, Q1D, VDIM, DIM, ne);
      auto A = Reshape(pa_adj.Write(), VDIM, DIM, Q1D, Q1D, Q1D, ne);

      mfem::forall_3D(ne, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t J11 = J(qx, qy, qz, 0, 0, e),
                               J12 = J(qx, qy, qz, 0, 1, e),
                               J13 = J(qx, qy, qz, 0, 2, e);
                  const real_t J21 = J(qx, qy, qz, 1, 0, e),
                               J22 = J(qx, qy, qz, 1, 1, e),
                               J23 = J(qx, qy, qz, 1, 2, e);
                  const real_t J31 = J(qx, qy, qz, 2, 0, e),
                               J32 = J(qx, qy, qz, 2, 1, e),
                               J33 = J(qx, qy, qz, 2, 2, e);
                  const real_t cw = W(qx, qy, qz) * C(qx, qy, qz, e);
                  // adj(J)
                  const real_t A11 = (J22 * J33) - (J23 * J32);
                  const real_t A12 = (J32 * J13) - (J12 * J33);
                  const real_t A13 = (J12 * J23) - (J22 * J13);
                  const real_t A21 = (J31 * J23) - (J21 * J33);
                  const real_t A22 = (J11 * J33) - (J13 * J31);
                  const real_t A23 = (J21 * J13) - (J11 * J23);
                  const real_t A31 = (J21 * J32) - (J31 * J22);
                  const real_t A32 = (J31 * J12) - (J11 * J32);
                  const real_t A33 = (J11 * J22) - (J12 * J21);
                  // Store wq * coeff * adj(J)
                  A(0, 0, qx, qy, qz, e) = cw * A11;
                  A(1, 0, qx, qy, qz, e) = cw * A12;
                  A(2, 0, qx, qy, qz, e) = cw * A13;
                  A(0, 1, qx, qy, qz, e) = cw * A21;
                  A(1, 1, qx, qy, qz, e) = cw * A22;
                  A(2, 1, qx, qy, qz, e) = cw * A23;
                  A(0, 2, qx, qy, qz, e) = cw * A31;
                  A(1, 2, qx, qy, qz, e) = cw * A32;
                  A(2, 2, qx, qy, qz, e) = cw * A33;
               }
            }
         }
      });
   }
   else
   {
      MFEM_ABORT("dim " << dim << " not supported!");
   }

   if (static auto ini = false; !std::exchange(ini, true))
   {
      // 2D
      VectorConvectionNLFAddMultPA::Specialization<2, 2,2>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 2,3>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 3,4>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 3,5>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 4,5>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 4,6>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 5,7>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 5,8>::Add();
      VectorConvectionNLFAddMultPA::Specialization<2, 6,8>::Add();
      // 3D
      VectorConvectionNLFAddMultPA::Specialization<3, 2,3>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 2,4>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 2,5>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 3,4>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 3,5>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 3,6>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 4,6>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 4,7>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 4,8>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 5,7>::Add();
      VectorConvectionNLFAddMultPA::Specialization<3, 5,8>::Add();
   }
}

// PA Convection NL 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_MDQ = 16>
static void SmemPAConvectionNLApply2D(const int NE,
                                      const real_t *b,
                                      const real_t *g,
                                      const real_t *a,
                                      const real_t *x,
                                      real_t *y,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   constexpr int VDIM = 2, DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, NE);
   const auto X = Reshape(x, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D * T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : T_MDQ;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MDQ;

      MFEM_SHARED real_t smem[MQ1][MQ1], sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> g0, g1;
      kernels::internal::v_regs2d_t<VDIM, MQ1> r0, r1;
      kernels::internal::v_regs2d_t<VDIM, MQ1> s0, s1;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs2d(e, D1D, X, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r1); // u vector-value
      kernels::internal::LoadDofs2d(e, D1D, X, g0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, g0, g1); // u vector-gradient

      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const future::tensor<real_t, 2> U =
            {
               r1[0][qy][qx], r1[1][qy][qx]
            };
            const future::tensor<real_t, 2,2> gradU = {{
                  {g1[0][0][qy][qx], g1[1][0][qy][qx]},
                  {g1[0][1][qy][qx], g1[1][1][qy][qx]},
               }
            };
            const future::tensor<real_t, 2,2> Q = {{
                  {A(0,0,qx,qy,e), A(1,0,qx,qy,e)},
                  {A(0,1,qx,qy,e), A(1,1,qx,qy,e)},
               }
            };
            const future::tensor<real_t, 2> conv = transpose(gradU) * (Q * U);
            s0[0][qy][qx] = conv[0];
            s0[1][qy][qx] = conv[1];
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, s0, s1);
      kernels::internal::WriteDofs2d(e, D1D, s1, Y);
   });
}

// PA Convection NL 3D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_MDQ = 16>
static void SmemPAConvectionNLApply3D(const int NE,
                                      const real_t *b,
                                      const real_t *g,
                                      const real_t *a,
                                      const real_t *x,
                                      real_t *y,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   constexpr int VDIM = 3, DIM = 3;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto A = Reshape(a, VDIM, DIM, Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x, D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y, D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D*T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : T_MDQ;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MDQ;

      MFEM_SHARED real_t smem[MQ1][MQ1], sB[MD1][MQ1], sG[MD1][MQ1];

      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> g0, g1;
      kernels::internal::v_regs3d_t<VDIM, MQ1> r0, r1;
      kernels::internal::v_regs3d_t<VDIM, MQ1> s0, s1;

      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      kernels::internal::LoadDofs3d(e, D1D, X, r0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r0, r1); // u vector-value
      kernels::internal::LoadDofs3d(e, D1D, X, g0);
      kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, g0, g1); // u vector-gradient

      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const future::tensor<real_t, 3> U =
               {
                  r1[0][qz][qy][qx], r1[1][qz][qy][qx], r1[2][qz][qy][qx]
               };
               const future::tensor<real_t, 3,3> gradU = {{
                     {g1[0][0][qz][qy][qx], g1[1][0][qz][qy][qx], g1[2][0][qz][qy][qx]},
                     {g1[0][1][qz][qy][qx], g1[1][1][qz][qy][qx], g1[2][1][qz][qy][qx]},
                     {g1[0][2][qz][qy][qx], g1[1][2][qz][qy][qx], g1[2][2][qz][qy][qx]}
                  }
               };
               const future::tensor<real_t, 3,3> Q = {{
                     {A(0,0,qx,qy,qz,e), A(1,0,qx,qy,qz,e), A(2,0,qx,qy,qz,e)},
                     {A(0,1,qx,qy,qz,e), A(1,1,qx,qy,qz,e), A(2,1,qx,qy,qz,e)},
                     {A(0,2,qx,qy,qz,e), A(1,2,qx,qy,qz,e), A(2,2,qx,qy,qz,e)}
                  }
               };
               const future::tensor<real_t, 3> conv = transpose(gradU) * (Q * U);
               s0[0][qz][qy][qx] = conv[0];
               s0[1][qz][qy][qx] = conv[1];
               s0[2][qz][qy][qx] = conv[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      kernels::internal::EvalTranspose3d(D1D, Q1D, smem, sB, s0, s1);
      kernels::internal::WriteDofs3d(e, D1D, s1, Y);
   });
}

void VectorConvectionNLFIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      VectorConvectionNLFAddMultPA::Run(dim, d1d, q1d, ne,
                                        maps->B.Read(),
                                        maps->G.Read(),
                                        pa_adj.Read(),
                                        x.Read(),
                                        y.ReadWrite(),
                                        d1d, q1d);
   }
}

/// \cond DO_NOT_DOCUMENT

template<int DIM, int T_D1D, int T_Q1D>
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultPA::Kernel()
{
   static_assert(T_D1D <= T_Q1D, "d1d > q1d is not supported");
   if constexpr (DIM == 2)
   {
      return SmemPAConvectionNLApply2D<T_D1D, T_Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return SmemPAConvectionNLApply3D<T_D1D, T_Q1D>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultPAType
VectorConvectionNLFIntegrator::VectorConvectionNLFAddMultPA::Fallback
(int dim, int d1d, int q1d)
{
   MFEM_VERIFY(d1d <= q1d, "d1d > q1d is not supported");
   MFEM_VERIFY(d1d <= 16, "d1d > 16 is not supported");
   MFEM_VERIFY(q1d <= 16, "q1d > 16 is not supported");
   if (dim == 2)
   {
      return SmemPAConvectionNLApply2D<>;
   }
   else if (dim == 3)
   {
      return SmemPAConvectionNLApply3D<>;
   }
   else { MFEM_ABORT("Unsupported kernel"); }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
