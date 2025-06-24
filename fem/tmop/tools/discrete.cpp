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

#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../kernels.hpp"
#include "../../../general/forall.hpp"

namespace mfem
{

using namespace kernels::internal;

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_DatcSize_2D(const int NE,
                      const int ncomp,
                      const int sizeidx,
                      const real_t input_min_size,
                      const real_t *nc_reduce,
                      const ConstDeviceMatrix &W,
                      const real_t *b,
                      const DeviceTensor<4, const real_t> &X,
                      DeviceTensor<5> &J,
                      const int d1d = 0,
                      const int q1d = 0)
{
   static constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= Q1D, "");

   const real_t infinity = std::numeric_limits<real_t>::infinity();

   mfem::forall_3D_grid(NE, Q1D, Q1D, 1, 512, [=] MFEM_HOST_DEVICE(int e)
   {
      static constexpr int BLOCK_DIM = 512;

      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t min_size[BLOCK_DIM];

      vd_regs2d_t<1,1,MQ1> r0, r1; // scalar X (component sizeidx)

      LoadDofs2d(e, D1D, X, r0);

      DeviceTensor<2, real_t> M((real_t *)(min_size), D1D, D1D);
      MFEM_FOREACH_THREAD(t, x, BLOCK_DIM) { min_size[t] = infinity; }
      MFEM_SYNC_THREAD;
      mfem::tmop::foreach_y_thread(D1D, [&](int dy)
      {
         mfem::tmop::foreach_x_thread(D1D, [&](int dx)
         {
            M(dy, dx) = r0(sizeidx, 0, dy, dx);
         });
      });

      MFEM_SYNC_THREAD;
      for (int wrk = BLOCK_DIM >> 1; wrk > 0; wrk >>= 1)
      {
         MFEM_FOREACH_THREAD(t, x, BLOCK_DIM)
         {
            if (t < wrk && MFEM_THREAD_ID(y) == 0 && MFEM_THREAD_ID(z) == 0)
            {
               min_size[t] = fmin(min_size[t], min_size[t + wrk]);
            }
         }
         MFEM_SYNC_THREAD;
      }
      real_t min = min_size[0];
      if (input_min_size > 0.0) { min = input_min_size; }

      LoadMatrix(D1D, Q1D, b, sB);
      Eval2d(D1D, Q1D, smem, sB, r0, r1);

      mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
      {
         mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
         {
            const real_t T = r1(0, 0, qy, qx);

            const real_t shape_par_vals = T;
            const real_t size = fmax(shape_par_vals, min) / nc_reduce[e];
            const real_t alpha = std::pow(size, 1.0 / DIM);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  J(i, j, qx, qy, e) = alpha * W(i, j);
               }
            }
         });
      });
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPDatc2Size, TMOP_DatcSize_2D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPDatc2Size);

template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
void TMOP_DatcSize_3D(const int NE,
                      const int ncomp,
                      const int sizeidx,
                      const real_t input_min_size,
                      const real_t *nc_reduce,
                      const ConstDeviceMatrix &W,
                      const real_t *b,
                      const DeviceTensor<5, const real_t> &X,
                      DeviceTensor<6> &J,
                      const int d1d = 0,
                      const int q1d = 0)
{
   static constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const real_t infinity = std::numeric_limits<real_t>::infinity();

   mfem::forall_3D_grid(NE, Q1D, Q1D, 1, 512, [=] MFEM_HOST_DEVICE(int e)
   {
      static constexpr int BLOCK_DIM = 512;

      MFEM_SHARED real_t sB[MD1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t min_size[BLOCK_DIM];

      vd_regs3d_t<1,1,MQ1> r0, r1; // scalar X (component sizeidx)

      LoadDofs3d(e, D1D, X, r0);

      DeviceTensor<3, real_t> M((real_t *)(min_size), D1D, D1D, D1D);
      MFEM_FOREACH_THREAD(t, x, BLOCK_DIM) { min_size[t] = infinity; }
      MFEM_SYNC_THREAD;
      for (int dz = 0; dz < D1D; ++dz)
      {
         mfem::tmop::foreach_y_thread(D1D, [&](int dy)
         {
            mfem::tmop::foreach_x_thread(D1D, [&](int dx)
            {
               M(dz, dy, dx) = r0(sizeidx, 0, dz, dy, dx);
            });
         });
      }
      MFEM_SYNC_THREAD;
      for (int wrk = BLOCK_DIM >> 1; wrk > 0; wrk >>= 1)
      {
         MFEM_FOREACH_THREAD(t, x, BLOCK_DIM)
         {
            if (t < wrk && MFEM_THREAD_ID(y) == 0 && MFEM_THREAD_ID(z) == 0)
            {
               min_size[t] = fmin(min_size[t], min_size[t + wrk]);
            }
         }
         MFEM_SYNC_THREAD;
      }
      real_t min = min_size[0];
      if (input_min_size > 0.0) { min = input_min_size; }

      LoadMatrix(D1D, Q1D, b, sB);
      Eval3d(D1D, Q1D, smem, sB, r0, r1);

      for (int qz = 0; qz < Q1D; ++qz)
      {
         mfem::tmop::foreach_y_thread(Q1D, [&](int qy)
         {
            mfem::tmop::foreach_x_thread(Q1D, [&](int qx)
            {
               const real_t T = r1(0, 0, qz, qy, qx);

               const real_t shape_par_vals = T;
               const real_t size = fmax(shape_par_vals, min) / nc_reduce[e];
               const real_t alpha = std::pow(size, 1.0 / DIM);
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     J(i, j, qx, qy, qz, e) = alpha * W(i, j);
                  }
               }
            });
         });
      }
   });
}

MFEM_TMOP_MDQ_REGISTER(TMOPDatc3Size, TMOP_DatcSize_3D);
MFEM_TMOP_MDQ_SPECIALIZE(TMOPDatc3Size);

// PA.Jtr Size = (dim, dim, PA.ne*PA.nq);
void DiscreteAdaptTC::ComputeAllElementTargets(const FiniteElementSpace &pa_fes,
                                               const IntegrationRule &ir,
                                               const Vector &xe,
                                               DenseTensor &Jtr) const
{
   MFEM_VERIFY(target_type == IDEAL_SHAPE_GIVEN_SIZE ||
               target_type == GIVEN_SHAPE_AND_SIZE, "Wrong target type.");

   MFEM_VERIFY(tspec_fesv, "No target specifications have been set.");
   const FiniteElementSpace *fes = tspec_fesv;

   // The kernels are implemented only for size-adaptivity with ideal shape.
   const bool use_kernel = (skewidx == -1 && aspectratioidx == -1 &&
                            orientationidx == -1 && sizeidx != -1 &&
                            target_type == IDEAL_SHAPE_GIVEN_SIZE);
   if (use_kernel == false)
   {
      return ComputeAllElementTargets_Fallback(pa_fes, ir, xe, Jtr);
   }

   const Mesh *mesh = fes->GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported");
   MFEM_VERIFY(!fes->IsVariableOrder(), "variable orders are not supported");
   const FiniteElement &fe = *fes->GetTypicalFE();
   const DenseMatrix &w = Geometries.GetGeomToPerfGeomJac(fe.GetGeomType());
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &maps = fe.GetDofToQuad(ir, mode);
   const int d = maps.ndof, q = maps.nqpt;
   const real_t input_min_size = lim_min_size;

   MFEM_VERIFY(ncomp == 1 && sizeidx == 0, "ncomp or sizeidx error.");
   MFEM_VERIFY(d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   Vector nc_size_red(NE, Device::GetDeviceMemoryType());
   nc_size_red.HostWrite();
   NCMesh *ncmesh = tspec_fesv->GetMesh()->ncmesh;
   for (int e = 0; e < NE; e++)
   {
      nc_size_red(e) = (ncmesh) ? ncmesh->GetElementSizeReduction(e) : 1.0;
   }
   const real_t *nc_reduce = nc_size_red.Read();

   auto R = fes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(R, "Invalid ElementRestriction.");
   const int R_height = (dim == 2) ? NE*ncomp*d*d : NE*ncomp*d*d*d;
   MFEM_VERIFY(R->Height() == R_height, "Incorrect ElementRestriction size.");
   Vector tspec_e(R->Height(), Device::GetDeviceMemoryType());
   tspec_e.UseDevice(true);
   tspec.UseDevice(true);
   R->Mult(tspec, tspec_e);

   const auto *b = maps.B.Read();

   if (dim == 2)
   {
      const auto W = Reshape(w.Read(), 2, 2);
      const auto X = Reshape(tspec_e.Read(), d, d, ncomp, NE);
      auto J = Reshape(Jtr.Write(), 2, 2, q, q, NE);
      TMOPDatc2Size::Run(d, q,
                         NE, ncomp, sizeidx, input_min_size, nc_reduce, W, b, X, J,
                         d, q);
   }
   else
   {
      const auto W = Reshape(w.Read(), 3, 3);
      const auto X = Reshape(tspec_e.Read(), d, d, d, ncomp, NE);
      auto J = Reshape(Jtr.Write(), 3, 3, q, q, q, NE);
      TMOPDatc3Size::Run(d, q,
                         NE, ncomp, sizeidx, input_min_size, nc_reduce, W, b, X, J,
                         d, q);
   }
}

} // namespace mfem
