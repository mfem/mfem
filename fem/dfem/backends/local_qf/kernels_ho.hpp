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

#include "../../../kernels.hpp"
namespace ker = mfem::kernels::internal;

#include "../../util.hpp" // for ThreadBlocks
#include "util.hpp"

namespace mfem::future
{

// ────────────────────────────────────────────────────────────────────────────
inline constexpr int LocalQFHOBackendMQ1() { return 16; }

// ────────────────────────────────────────────────────────────────────────────
/// Register type for one HO q-function parameter
template<typename KerOps, typename T, int rank = qf_param_shape<T>::rank>
struct ho_qreg;

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 0>
{
   using type = typename KerOps::template val_reg_t<1>;
};

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 1>
{
   static constexpr int e0 = qf_param_shape<T>::extents[0];
   using type = typename KerOps::template val_reg_t<e0>;
};

template<typename KerOps, typename T>
struct ho_qreg<KerOps, T, 2>
{
   static constexpr int VDIM = qf_param_shape<T>::extents[0];
   static constexpr int SDIM = qf_param_shape<T>::extents[1];
   using type = typename KerOps::template del_reg_t<VDIM, SDIM>;
};

template<typename KerOps, typename T>
using ho_qreg_t = typename ho_qreg<KerOps, T>::type;

// ────────────────────────────────────────────────────────────────────────────
namespace hok
{

/// Load one quadrature-point value
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto load_at(Reg &reg, int qx, int qy, int qz)
{
   static_assert(DIM == 2 || DIM == 3);
   constexpr int RNK = qf_param_shape<T>::rank;
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      if constexpr (RNK == 0) { return T{ reg(0, qy, qx) }; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         T t{};
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd) { t(dd) = reg(dd, qy, qx); }
         return t;
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         T t;
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j) { t(i, j) = reg(i, j, qy, qx); }
         }
         return t;
      }
   }
   else
   {
      if constexpr (RNK == 0) { return T{ reg(0, qz, qy, qx) }; }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         T t{};
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd) { t(dd) = reg(dd, qz, qy, qx); }
         return t;
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         T t;
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j) { t(i, j) = reg(i, j, qz, qy, qx); }
         }
         return t;
      }
   }
}

template<bool tangent, typename U>
MFEM_HOST_DEVICE inline auto qp_store(const U &v)
{
   if constexpr (tangent) { return qf_store_gradient(v); }
   else
   {
      return qf_store_value(v);
   }
}

// Store primal value or dual tangent at one quadrature point
template<int DIM, typename T, typename Reg, bool tangent>
MFEM_HOST_DEVICE inline void
store_at(Reg &reg, int qx, int qy, int qz, const T &out)
{
   static_assert(DIM == 2 || DIM == 3);
   constexpr int RNK = qf_param_shape<T>::rank;
   if constexpr (DIM == 2)
   {
      MFEM_CONTRACT_VAR(qz);
      if constexpr (RNK == 0) { reg(0, qy, qx) = qp_store<tangent>(out); }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd)
         {
            reg(dd, qy, qx) = qp_store<tangent>(out(dd));
         }
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               reg(i, j, qy, qx) = qp_store<tangent>(out(i, j));
            }
         }
      }
   }
   else
   {
      if constexpr (RNK == 0) { reg(0, qz, qy, qx) = qp_store<tangent>(out); }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd)
         {
            reg(dd, qz, qy, qx) = qp_store<tangent>(out(dd));
         }
      }
      else
      {
         constexpr int e0 = qf_param_shape<T>::extents[0];
         constexpr int e1 = qf_param_shape<T>::extents[1];
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               reg(i, j, qz, qy, qx) = qp_store<tangent>(out(i, j));
            }
         }
      }
   }
}

// Pull primal/tangent pair into a dual q-function argument
template<int DIM, typename T, typename Reg>
MFEM_HOST_DEVICE inline auto
pull_directional(Reg &preg, Reg &sreg, int qx, int qy, int qz, bool dependent)
{
   if constexpr (!qf_param_uses_dual_v<T>)
   {
      return load_at<DIM, T>(preg, qx, qy, qz);
   }
   else
   {
      if (!dependent) { return load_at<DIM, T>(preg, qx, qy, qz); }
      constexpr int RNK = qf_param_shape<T>::rank;
      if constexpr (DIM == 2)
      {
         MFEM_CONTRACT_VAR(qz);
         if constexpr (RNK == 0)
         {
            return T{ preg(0, qy, qx), sreg(0, qy, qx) };
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            T t{};
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd)
            {
               t(dd) = { preg(dd, qy, qx), sreg(dd, qy, qx) };
            }
            return t;
         }
         else
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            T t;
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  t(i, j) = { preg(i, j, qy, qx), sreg(i, j, qy, qx) };
               }
            }
            return t;
         }
      }
      else
      {
         if constexpr (RNK == 0)
         {
            return T{ preg(0, qz, qy, qx), sreg(0, qz, qy, qx) };
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            T t{};
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd)
            {
               t(dd) = { preg(dd, qz, qy, qx), sreg(dd, qz, qy, qx) };
            }
            return t;
         }
         else
         {
            constexpr int e0 = qf_param_shape<T>::extents[0];
            constexpr int e1 = qf_param_shape<T>::extents[1];
            T t;
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  t(i, j) = { preg(i, j, qz, qy, qx), sreg(i, j, qz, qy, qx) };
               }
            }
            return t;
         }
      }
   }
}

} // namespace hok

// ────────────────────────────────────────────────────────────────────────────
/// HO tensor-product kernels
template<int T_DIM, int MQ1>
struct ho_ker_backend
{
   static constexpr int DIM = T_DIM;
   static_assert(DIM == 2 || DIM == 3);

   template<int VDIM>
   using val_reg_t = std::conditional_t<(DIM == 2),
         ker::v_regs2d_t<VDIM, MQ1>,
         ker::v_regs3d_t<VDIM, MQ1>>;

   template<int VDIM, int SDIM>
   using del_reg_t = std::conditional_t<(DIM == 2),
         ker::vd_regs2d_t<VDIM, SDIM, MQ1>,
         ker::vd_regs3d_t<VDIM, SDIM, MQ1>>;

   using vector_dofs_t = std::conditional_t<(DIM == 2),
         ker::s_regs2d_t<MQ1>, ker::s_regs3d_t<MQ1>>;

   struct Shared
   {
      real_t M[MQ1][MQ1], B[MQ1][MQ1], G[MQ1][MQ1];
   };

   template<typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void
   load_dofs(const int e, const int d, const XE_t &XE, Dofs &dofs)
   {
      if constexpr (DIM == 2) { ker::LoadDofs2d(e, d, XE, dofs); }
      else
      {
         ker::LoadDofs3d(e, d, XE, dofs);
      }
   }

   // Load the vector degrees of freedom for a given component of a vector element.
   template<typename XE_t>
   static MFEM_HOST_DEVICE void load_vector_dofs(
      const int e, const DofToQuadMap &m, const int c,
      const XE_t &XE, vector_dofs_t &dofs)
   {
      // Loading the vector degrees of freedom into the local register `dofs`
      // We assume they have a flat view due to the ND/RT per axis extents of the element
      const int ex = m.Extent(c, 0), ey = m.Extent(c, 1);
      const int ez = (DIM == 2) ? 1 : m.Extent(c, 2);
      const int off = m.Offset(c);
      for (int dz = 0; dz < ez; dz++)
      {
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(dx, x, ex)
         {
            const real_t value = XE(off + dx + ex * (dy + ey * dz),
                                    0, 0, 0, e);
            if constexpr (DIM == 2) { dofs[dy][dx] = value; }
            else { dofs[dz][dy][dx] = value; }
         }
      }
      MFEM_SYNC_THREAD;
   }

   /// Run sweep @a t of component block @a c under field operator @a FOP, accumulating
   /// into the q-function register bank @a rarg.
   /// Bx -> By -> Bz
   ///
   /// Basically the vector-FE counterpart of ker::Contract3d<false>. Follows
   /// the same axis sweep order, with two differences forced by ND/RT: the 1D
   /// factors are taken per component, since a component uses the closed basis
   /// along some axes and the open basis along the others, and the destination
   /// is a bank slot picked by the operator rather than the component index.
   template<typename FOP, typename Smem, typename ArgReg>
   static MFEM_HOST_DEVICE void contract_vector_component(
      const DofToQuadMap &m, const int c, const int t,
      Smem &s, const vector_dofs_t &dofs, ArgReg &rarg)
   {
      // Determine the sweep parameters for this component and term.
      // Based on the operator type and the component index
      const VecTerm vt = vector_term<FOP>(c, t);
      const int deriv_dir = vt.deriv_dir;
      const int q1d = m.Q1D();

      // The 1D factor for a given (component, axis, deriv) is fixed for the
      // whole sweep, so it is resolved once here instead of per multiply-add.
      const real_t *Bx = m.Basis(c, 0, deriv_dir == 0);
      const real_t *By = m.Basis(c, 1, deriv_dir == 1);
      const real_t *Bz = m.Basis(c, 2, deriv_dir == 2);
      MFEM_CONTRACT_VAR(Bz);

      // Get the extents of the element along each axis.
      const int ex = m.Extent(c, 0), ey = m.Extent(c, 1);
      const int ez = (DIM == 2) ? 1 : m.Extent(c, 2);

      for (int dz = 0; dz < ez; dz++)
      {
         // Sweep along the x-axis for the current z-slice.
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            for (int dx = 0; dx < ex; dx++)
            {
               if constexpr (DIM == 2)
               {
                  value += Bx[qx + q1d * dx] * dofs[dy][dx];
               }
               else
               {
                  value += Bx[qx + q1d * dx] *
                           dofs[dz][dy][dx];
               }
            }
            s.M[dy][qx] = value;
         }
         MFEM_SYNC_THREAD;

         // Sweep along the y-axis for the current z-slice.
         MFEM_FOREACH_THREAD(qy, y, q1d)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            for (int dy = 0; dy < ey; dy++)
            {
               value += By[qy + q1d * dy] * s.M[dy][qx];
            }
            if constexpr (DIM == 2) { vector_accum<FOP>(rarg, qx, qy, 0, vt, value); }
            else
            {
               for (int qz = 0; qz < q1d; qz++)
               {
                  vector_accum<FOP>(rarg, qx, qy, qz, vt,
                                    Bz[qz + q1d * dz] * value);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }

   /// Integrate quadrature-point values against one component block,
   /// producing that block's degrees of freedom.
   /// Same assumptions as for the forward contraction above.
   /// Bzt -> Byt -> Bxt
   template<typename FOP, typename Smem, typename ArgReg>
   static MFEM_HOST_DEVICE void contract_vector_component_transpose(
      const DofToQuadMap &m, const int c, const int t,
      Smem &s, ArgReg &rarg, vector_dofs_t &dofs)
   {
      const VecTerm vt = vector_term<FOP>(c, t);
      const int deriv_dir = vt.deriv_dir;
      const int q1d = m.Q1D();
      // The 1D factor for a given (component, axis, deriv) is fixed for the
      // whole sweep, so it is resolved once here instead of per multiply-add.
      const real_t *Bx = m.Basis(c, 0, deriv_dir == 0);
      const real_t *By = m.Basis(c, 1, deriv_dir == 1);
      const real_t *Bz = m.Basis(c, 2, deriv_dir == 2);
      MFEM_CONTRACT_VAR(Bz);

      // Get the extents of the element along each axis.
      const int ex = m.Extent(c, 0), ey = m.Extent(c, 1);
      const int ez = (DIM == 2) ? 1 : m.Extent(c, 2);
      for (int dz = 0; dz < ez; dz++)
      {
         // Sweep along the z-axis for the current element slice.
         MFEM_FOREACH_THREAD(qy, y, q1d)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            if constexpr (DIM == 2) { value = vector_src<FOP>(rarg, qx, qy, 0, vt); }
            else
            {
               for (int qz = 0; qz < q1d; qz++)
               {
                  value += Bz[qz + q1d * dz] *
                           vector_src<FOP>(rarg, qx, qy, qz, vt);
               }
            }
            s.M[qy][qx] = value;
         }
         MFEM_SYNC_THREAD;

         // Sweep along the y-axis for the current z-slice.
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            real_t value = 0.0;
            for (int qy = 0; qy < q1d; qy++)
            {
               value += By[qy + q1d * dy] * s.M[qy][qx];
            }
            s.B[dy][qx] = value;
         }
         MFEM_SYNC_THREAD;

         // Sweep along the x-axis for the current y-slice.
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(dx, x, ex)
         {
            real_t value = 0.0;
            for (int qx = 0; qx < q1d; qx++)
            {
               value += Bx[qx + q1d * dx] * s.B[dy][qx];
            }
            if constexpr (DIM == 2) { dofs[dy][dx] = value; }
            else { dofs[dz][dy][dx] = value; }
         }
         MFEM_SYNC_THREAD;
      }
   }

   template<typename YE_t>
   static MFEM_HOST_DEVICE void write_vector_dofs(
      const int e, const DofToQuadMap &m, const int c,
      const vector_dofs_t &dofs, const YE_t &YE)
   {
      const int ex = m.Extent(c, 0), ey = m.Extent(c, 1);
      const int ez = (DIM == 2) ? 1 : m.Extent(c, 2);
      const int off = m.Offset(c);
      for (int dz = 0; dz < ez; dz++)
      {
         MFEM_FOREACH_THREAD(dy, y, ey)
         MFEM_FOREACH_THREAD(dx, x, ex)
         {
            if constexpr (DIM == 2)
            {
               YE(off + dx + ex * dy, 0, 0, 0, e) += dofs[dy][dx];
            }
            else
            {
               YE(off + dx + ex * (dy + ey * dz), 0, 0, 0, e) +=
                  dofs[dz][dy][dx];
            }
         }
      }
      MFEM_SYNC_THREAD;
   }

   /// Interpolate a vector element at the quadrature points under field
   /// operator @a FOP.
   template<typename FOP, typename XE_t, typename ArgReg>
   static MFEM_HOST_DEVICE void load_vector(
      Shared &s, const int e, const DofToQuadMap &m,
      const XE_t &XE, ArgReg &rarg)
   {
      zero_vector_components(rarg, vector_num_slots<FOP>(m.range_dim), m.Q1D());

      const int nterms = vector_num_terms<FOP>(m.range_dim);
      for (int c = 0; c < m.range_dim; c++)
      {
         // Gathered once and reused by every sweep of this component.
         vector_dofs_t dofs;
         load_vector_dofs(e, m, c, XE, dofs);
         for (int t = 0; t < nterms; t++)
         {
            contract_vector_component<FOP>(m, c, t, s, dofs, rarg);
         }
      }
   }

   /// Integrate quadrature-point data against a vector element under field
   /// operator @a FOP and add the result into the element vector.
   template<typename FOP, typename YE_t, typename ArgReg>
   static MFEM_HOST_DEVICE void write_vector(
      Shared &s, const int e, const DofToQuadMap &m,
      const YE_t &YE, ArgReg &rarg)
   {
      const int nterms = vector_num_terms<FOP>(m.range_dim);
      for (int c = 0; c < m.range_dim; c++)
      {
         for (int t = 0; t < nterms; t++)
         {
            // write_vector_dofs adds into YE, so several sweeps of one
            // component accumulate rather than overwrite.
            vector_dofs_t dofs;
            contract_vector_component_transpose<FOP>(m, c, t, s, rarg, dofs);
            write_vector_dofs(e, m, c, dofs, YE);
         }
      }
   }
private:
   /// Accumulate one quadrature-point contribution into the slot named by
   /// @a vt. The sign is only ever non-unit for a Curl, so the other
   /// operators keep a bare add in the innermost loop.
   template<typename FOP, typename Reg>
   static MFEM_HOST_DEVICE void vector_accum(
      Reg &reg, int qx, int qy, int qz, const VecTerm &vt, real_t value)
   {
      if constexpr (is_curl_fop_v<FOP>)
      {
         vector_component(reg, qx, qy, qz, vt.slot) += vt.sgn * value;
      }
      else { vector_component(reg, qx, qy, qz, vt.slot) += value; }
   }

   /// Read back the slot named by @a vt, the mirror of vector_accum.
   template<typename FOP, typename Reg>
   static MFEM_HOST_DEVICE real_t vector_src(
      Reg &reg, int qx, int qy, int qz, const VecTerm &vt)
   {
      const real_t value = vector_component(reg, qx, qy, qz, vt.slot);
      if constexpr (is_curl_fop_v<FOP>) { return vt.sgn * value; }
      else { return value; }
   }

   template<typename Reg>
   static MFEM_HOST_DEVICE real_t &vector_component(
      Reg &reg, int qx, int qy, int qz, int c)
   {
      if constexpr (DIM == 2)
      {
         MFEM_CONTRACT_VAR(qz);
         return reg(c, qy, qx);
      }
      else { return reg(c, qz, qy, qx); }
   }

   template<typename Reg>
   static MFEM_HOST_DEVICE void zero_vector_components(
      Reg &reg, const int ncomp, const int q1d)
   {
      MFEM_FOREACH_THREAD(qy, y, q1d)
      MFEM_FOREACH_THREAD(qx, x, q1d)
      for (int qz = 0; qz < ((DIM == 2) ? 1 : q1d); qz++)
         for (int c = 0; c < ncomp; c++)
         {
            vector_component(reg, qx, qy, qz, c) = 0.0;
         }
      MFEM_SYNC_THREAD;
   }

public:

   template<int VDIM, int SDIM, typename XE_t, typename Dofs>
   static MFEM_HOST_DEVICE void
   load_grad_dofs(const int e, const int d, const XE_t &XE, Dofs &dofs)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      load_dofs(e, d, XE, dofs);
   }

   template<typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void
   eval_value(const int d, const int q, Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      if constexpr (DIM == 2) { ker::Eval2d(d, q, s.M, s.B, dofs, rarg); }
      else
      {
         ker::Eval3d(d, q, s.M, s.B, dofs, rarg);
      }
   }

   template<int VDIM, int SDIM, typename Smem, typename Dofs, typename ArgReg>
   static MFEM_HOST_DEVICE void
   grad(const int d, const int q, Smem &s, Dofs &dofs, ArgReg &rarg)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      if constexpr (DIM == 2) { ker::Grad2d(d, q, s.M, s.B, s.G, dofs, rarg); }
      else
      {
         ker::Grad3d(d, q, s.M, s.B, s.G, dofs, rarg);
      }
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_value(const int d,
                                            const int q,
                                            const int e,
                                            Smem &s,
                                            ArgReg &rarg,
                                            Dofs &dofs,
                                            YE_t &YE)
   {
      if constexpr (DIM == 2)
      {
         ker::EvalTranspose2d(d, q, s.M, s.B, rarg, dofs);
         ker::WriteDofs2d(e, d, dofs, YE);
      }
      else
      {
         ker::EvalTranspose3d(d, q, s.M, s.B, rarg, dofs);
         ker::WriteDofs3d(e, d, dofs, YE);
      }
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient_2d(const int d,
                                                  const int q,
                                                  const int e,
                                                  Smem &s,
                                                  ArgReg &rarg,
                                                  Dofs &dofs,
                                                  YE_t &YE)
   {
      ker::GradTranspose2d(d, q, s.M, s.B, s.G, rarg, dofs);
      ker::WriteDofs2d(e, d, dofs, YE);
   }

   template<typename Smem, typename Dofs, typename ArgReg, typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient_3d(const int d,
                                                  const int q,
                                                  const int e,
                                                  Smem &s,
                                                  ArgReg &rarg,
                                                  Dofs &dofs,
                                                  YE_t &YE)
   {
      ker::GradTranspose3d(d, q, s.M, s.B, s.G, rarg, dofs);
      ker::WriteDofs3d(e, d, dofs, YE);
   }

   template<int VDIM,
            int SDIM,
            typename Smem,
            typename Dofs,
            typename ArgReg,
            typename YE_t>
   static MFEM_HOST_DEVICE void write_gradient(const int d,
                                               const int q,
                                               const int e,
                                               Smem &s,
                                               ArgReg &rarg,
                                               Dofs &dofs,
                                               YE_t &YE)
   {
      static_assert(SDIM == DIM, "gradient spatial dim must match kernel DIM");
      if constexpr (DIM == 2) { write_gradient_2d(d, q, e, s, rarg, dofs, YE); }
      else
      {
         write_gradient_3d(d, q, e, s, rarg, dofs, YE);
      }
   }
};

// ────────────────────────────────────────────────────────────────────────────
template<int T_DIM, int T_Q1D = LocalQFHOBackendMQ1()>
struct LocalQFHOBackend
{
   // ─────────────────────────────────────────────────────
   static constexpr int DIM = T_DIM, MQ1 = T_Q1D, Q1D = T_Q1D;
   static_assert(DIM == 2 || DIM == 3);

   // ─────────────────────────────────────────────────────
   static inline ThreadBlocks thread_blocks(const int q1d)
   {
      MFEM_ASSERT(q1d <= Q1D, "q1d must be <= " << Q1D);
      return { q1d, q1d, 1 };
   }

   // ─────────────────────────────────────────────────────
   static inline constexpr int MAX_THREADS_PER_BLOCK() { return Q1D * Q1D; }

   // ─────────────────────────────────────────────────────
   using backend_t = ho_ker_backend<DIM, Q1D>;

   // ─────────────────────────────────────────────────────
   using Shared = typename backend_t::Shared;

   // ─────────────────────────────────────────────────────
   template<typename WT, typename WI, typename Cache, typename AddY>
   static MFEM_HOST_DEVICE inline void DiagContract(Shared &s,
                                                    const int num_dof_1d,
                                                    const int q1d,
                                                    const int nz_dof,
                                                    WT wt,
                                                    WI wi,
                                                    Cache cache,
                                                    AddY add_y)
   {
      MFEM_CONTRACT_VAR(nz_dof);
      const int nqz = (DIM == 3) ? q1d : 1;
      const int ndz = (DIM == 3) ? num_dof_1d : 1;

      ker::s_regs3d_t<MQ1> rz, ry;
      auto &smem = s.M;

      MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
         {
            for (int dz = 0; dz < ndz; dz++)
            {
               real_t u = 0.0;
               for (int qz = 0; qz < nqz; qz++)
               {
                  const int q = qx + (qy + qz * q1d) * q1d;
                  const real_t wz =
                     (DIM == 3) ? (wt(2, qz, dz) * wi(2, qz, dz)) : real_t(1);
                  u += wz * cache(q);
               }
               rz[dz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int dz = 0; dz < ndz; dz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            { smem[qy][qx] = rz[dz][qy][qx]; }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(dy, y, num_dof_1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < q1d; qy++)
               {
                  u += wt(1, qy, dy) * wi(1, qy, dy) * smem[qy][qx];
               }
               ry[dz][dy][qx] = u;
            }
         }
         MFEM_SYNC_THREAD;
      }

      for (int dz = 0; dz < ndz; dz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(dy, y, num_dof_1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, q1d)
            { smem[dy][qx] = ry[dz][dy][qx]; }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD_DIRECT(dy, y, num_dof_1d)
         {
            MFEM_FOREACH_THREAD_DIRECT(dx, x, num_dof_1d)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < q1d; qx++)
               {
                  u += wt(0, qx, dx) * wi(0, qx, dx) * smem[dy][qx];
               }
               add_y(dx, dy, dz, u);
            }
         }
         MFEM_SYNC_THREAD;
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   using QReg = ho_qreg_t<backend_t, T>;

   // ─────────────────────────────────────────────────────
   /// Interpolate a field to quadrature points.
   /// Since we dispatch based on the DofToQuadMap, we can handle both
   /// scalar and vector FE.
   template<typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE void LoadValue(Shared &s,
                                                 const int e,
                                                 const DofToQuadMap &dtq,
                                                 const XE_T &XE,
                                                 ArgRegT &rarg)
   {
      if (dtq.IsVectorFE())
      {
         backend_t::template load_vector<Value<>>(s, e, dtq, XE, rarg);
         return;
      }
      const int d = dtq.D1D(), q = dtq.Q1D();
      ker::LoadMatrix(d, q, dtq.B, s.B);
      std::remove_reference_t<ArgRegT> dofs;
      backend_t::load_dofs(e, d, XE, dofs);
      backend_t::eval_value(d, q, s, dofs, rarg);
   }

   // ─────────────────────────────────────────────────────
   /// Evaluate the reference divergence of a field at quadrature points.
   template<typename ArgRegT, typename XE_T>
   static inline MFEM_HOST_DEVICE void LoadDiv(Shared &s,
                                               const int e,
                                               const DofToQuadMap &dtq,
                                               const XE_T &XE,
                                               ArgRegT &rarg)
   {
      backend_t::template load_vector<Div<>>(s, e, dtq, XE, rarg);
   }

   // ─────────────────────────────────────────────────────
   template<int RNK,
            typename ArgRegT,
            typename XE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE void LoadGradient(Shared &s,
                                                    const int e,
                                                    const DofToQuadMap &dtq,
                                                    const XE_T &XE,
                                                    ArgRegT &rarg)
   {
      const int d = dtq.D1D(), q = dtq.Q1D();
      ker::LoadMatrix(d, q, dtq.B, s.B);
      ker::LoadMatrix(d, q, dtq.G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      static constexpr int VDIM =
         (RNK == 1) ? 1 : qf_param_shape<FieldParamT>::extents[0];
      static constexpr int SDIM = (RNK == 1)
                                  ? qf_param_shape<FieldParamT>::extents[0]
                                  : qf_param_shape<FieldParamT>::extents[1];
      if constexpr (SDIM == DIM)
      {
         typename backend_t::template del_reg_t<VDIM, SDIM> dofs;
         if constexpr (RNK == 1) { backend_t::load_dofs(e, d, XE, dofs); }
         else
         {
            backend_t::template load_grad_dofs<VDIM, SDIM>(e, d, XE, dofs);
         }
         backend_t::template grad<VDIM, SDIM>(d, q, s, dofs, rarg);
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline auto
   qp_pull(QReg<T> &reg, int qx, int qy, int qz)
   { return hok::load_at<DIM, T>(reg, qx, qy, qz); }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline auto qp_pull_directional(
      QReg<T> &preg, QReg<T> &sreg, int qx, int qy, int qz, bool dependent)
   { return hok::pull_directional<DIM, T>(preg, sreg, qx, qy, qz, dependent); }

   // ─────────────────────────────────────────────────────
   template<typename DT, typename XE_T>
   static MFEM_HOST_DEVICE inline DT identity_qp_pull_dual(bool dependent,
                                                           const XE_T &XP,
                                                           const XE_T &XD,
                                                           int qx,
                                                           int qy,
                                                           int qz,
                                                           int e)
   {
      constexpr int RNK = qf_param_shape<DT>::rank;
      if constexpr (RNK == 0)
      {
         DT t{};
         t.value = XP(0, qx, qy, qz, e);
         t.gradient = dependent ? XD(0, qx, qy, qz, e) : 0.0;
         return t;
      }
      else if constexpr (RNK == 1)
      {
         constexpr int e0 = qf_param_shape<DT>::extents[0];
         DT t{};
         MFEM_UNROLL(e0)
         for (int dd = 0; dd < e0; ++dd)
         {
            t(dd).value = XP(dd, qx, qy, qz, e);
            t(dd).gradient = dependent ? XD(dd, qx, qy, qz, e) : 0.0;
         }
         return t;
      }
      else if constexpr (RNK == 2)
      {
         constexpr int e0 = qf_param_shape<DT>::extents[0];
         constexpr int e1 = qf_param_shape<DT>::extents[1];
         DT t{};
         MFEM_UNROLL(e0)
         for (int i = 0; i < e0; ++i)
         {
            MFEM_UNROLL(e1)
            for (int j = 0; j < e1; ++j)
            {
               t(i, j).value = XP(i + e0 * j, qx, qy, qz, e);
               t(i, j).gradient =
                  dependent ? XD(i + e0 * j, qx, qy, qz, e) : 0.0;
            }
         }
         return t;
      }
      else
      {
         static_assert(false, "Unsupported");
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline void
   qp_push(QReg<T> &reg, int qx, int qy, int qz, const T &out)
   { hok::store_at<DIM, T, decltype(reg), false>(reg, qx, qy, qz, out); }

   // ─────────────────────────────────────────────────────
   template<typename T>
   static MFEM_HOST_DEVICE inline void
   qp_push_tangent(QReg<T> &reg, int qx, int qy, int qz, const T &out)
   {
      hok::store_at<DIM, T, decltype(reg), qf_param_uses_dual_v<T>>(
                                                                    reg, qx, qy, qz, out);
   }

   // ─────────────────────────────────────────────────────
   template<typename DT, typename YE_T>
   static MFEM_HOST_DEVICE inline void identity_qp_write_value(
      YE_T &YE, int qx, int qy, int qz, int e, const DT &qout)
   {
      constexpr int RNK = qf_param_shape<DT>::rank;
      if constexpr (qf_param_uses_dual_v<DT>)
      {
         if constexpr (RNK == 0)
         {
            YE(0, qx, qy, qz, e) = qf_store_value(qout);
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<DT>::extents[0];
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd)
            {
               YE(dd, qx, qy, qz, e) = qf_store_value(qout(dd));
            }
         }
         else if constexpr (RNK == 2)
         {
            constexpr int e0 = qf_param_shape<DT>::extents[0];
            constexpr int e1 = qf_param_shape<DT>::extents[1];
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  YE(i + e0 * j, qx, qy, qz, e) = qf_store_value(qout(i, j));
               }
            }
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      }
   }

   // ─────────────────────────────────────────────────────
   template<typename DT, typename YE_T>
   static MFEM_HOST_DEVICE inline void identity_qp_write_tangent(
      YE_T &YE, int qx, int qy, int qz, int e, const DT &qout)
   {
      constexpr int RNK = qf_param_shape<DT>::rank;
      if constexpr (qf_param_uses_dual_v<DT>)
      {
         if constexpr (RNK == 0)
         {
            YE(0, qx, qy, qz, e) = qf_store_gradient(qout);
         }
         else if constexpr (RNK == 1)
         {
            constexpr int e0 = qf_param_shape<DT>::extents[0];
            MFEM_UNROLL(e0)
            for (int dd = 0; dd < e0; ++dd)
            {
               YE(dd, qx, qy, qz, e) = qf_store_gradient(qout(dd));
            }
         }
         else if constexpr (RNK == 2)
         {
            constexpr int e0 = qf_param_shape<DT>::extents[0];
            constexpr int e1 = qf_param_shape<DT>::extents[1];
            MFEM_UNROLL(e0)
            for (int i = 0; i < e0; ++i)
            {
               MFEM_UNROLL(e1)
               for (int j = 0; j < e1; ++j)
               {
                  YE(i + e0 * j, qx, qy, qz, e) = qf_store_gradient(qout(i, j));
               }
            }
         }
         else
         {
            static_assert(false, "Unsupported");
         }
      }
   }

   // ─────────────────────────────────────────────────────
   /// Integrate qp scalars against test basis functions.
   /// Since we dispatch based on the DofToQuadMap, we can deal with both
   /// scalar and vector FE.
   template<typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE void WriteValue(Shared &s,
                                                  const int e,
                                                  const DofToQuadMap &dtq,
                                                  YE_T &YE,
                                                  ArgRegT &rarg)
   {
      if (dtq.IsVectorFE())
      {
         backend_t::template write_vector<Value<>>(s, e, dtq, YE, rarg);
         return;
      }
      const int d = dtq.D1D(), q = dtq.Q1D();
      ker::LoadMatrix(d, q, dtq.B, s.B);
      std::remove_reference_t<ArgRegT> dofs;
      backend_t::write_value(d, q, e, s, rarg, dofs, YE);
   }

   // ─────────────────────────────────────────────────────
   /// Integrate qp scalars against divergence of test basis.
   template<typename ArgRegT, typename YE_T>
   static inline MFEM_HOST_DEVICE void WriteDiv(Shared &s,
                                                const int e,
                                                const DofToQuadMap &dtq,
                                                YE_T &YE,
                                                ArgRegT &rarg)
   {
      backend_t::template write_vector<Div<>>(s, e, dtq, YE, rarg);
   }

   // ─────────────────────────────────────────────────────
   template<int RNK,
            typename ArgRegT,
            typename YE_T,
            typename FieldParamT = ArgRegT>
   static inline MFEM_HOST_DEVICE void WriteGradient(Shared &s,
                                                     const int e,
                                                     const DofToQuadMap &dtq,
                                                     YE_T &YE,
                                                     ArgRegT &rarg)
   {
      const int d = dtq.D1D(), q = dtq.Q1D();
      ker::LoadMatrix(d, q, dtq.B, s.B);
      ker::LoadMatrix(d, q, dtq.G, s.G);
      static_assert(RNK == 1 || RNK == 2);
      static constexpr int VDIM =
         (RNK == 1) ? 1 : qf_param_shape<FieldParamT>::extents[0];
      static constexpr int SDIM = (RNK == 1)
                                  ? qf_param_shape<FieldParamT>::extents[0]
                                  : qf_param_shape<FieldParamT>::extents[1];
      if constexpr (SDIM == DIM)
      {
         typename backend_t::template del_reg_t<VDIM, SDIM> dofs;
         backend_t::template write_gradient<VDIM, SDIM>(
            d, q, e, s, rarg, dofs, YE);
      }
   }
};

/// @brief Dispatch to a compile-time HO kernel with MQ1 >= runtime @a q1d.
template <typename HOKernelTable, int DIM, int MQ1 = LocalQFHOBackendMQ1()>
inline typename HOKernelTable::KernelSignature
DispatchHOKernelByQ1D(int q1d)
{
   MFEM_VERIFY(q1d >= 2 && q1d <= MQ1,
               "Unsupported HO quadrature order: " << q1d);
   if (q1d <= 8) { return HOKernelTable::template Kernel<DIM, 8>(); }
   if constexpr (MQ1 > 8)
   {
      if (q1d <= 10) { return HOKernelTable::template Kernel<DIM, 10>(); }
   }
   if constexpr (MQ1 > 10)
   {
      if (q1d <= 12) { return HOKernelTable::template Kernel<DIM, 12>(); }
   }
   if constexpr (MQ1 > 12)
   {
      return HOKernelTable::template Kernel<DIM, 16>();
   }
   return nullptr;
}

/// @brief Select the compile-time HO kernel for runtime @a dim.
///
/// QFDIM is deduced at compile-time from the q-function signature.
/// If it is not possible to deduce the dimension (QFDIM=0), we fallback
/// to the original runtime dispatching, which means we emit both 2D and 3D
/// branches.
template <typename HOKernelTable, int QFDIM, int MQ1 = LocalQFHOBackendMQ1()>
inline typename HOKernelTable::KernelSignature
DispatchHOKernelByDim(int dim, int q1d)
{
   if constexpr (QFDIM == 2 || QFDIM == 3)
   {
      MFEM_VERIFY(dim == QFDIM,
                  "mesh dimension " << dim << " does not match the " << QFDIM
                  << "D q-function signature this integrator was built from");
      return DispatchHOKernelByQ1D<HOKernelTable, QFDIM, MQ1>(q1d);
   }
   else
   {
      // Couldn't deduce the dimension from the q-function signature,
      // we fall back to original runtime dispatching.
      if (dim == 2)
      {
         return DispatchHOKernelByQ1D<HOKernelTable, 2, MQ1>(q1d);
      }
      if (dim == 3)
      {
         return DispatchHOKernelByQ1D<HOKernelTable, 3, MQ1>(q1d);
      }
      MFEM_ABORT("Unsupported dimension " << dim);
      return nullptr;
   }
}

} // namespace mfem::future
