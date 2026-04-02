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

#include "../mesh/mesh_headers.hpp"
#include "eltrans/eltrans_basis.hpp"
#include "fem.hpp"

#include "eltrans.hpp"

#include "../general/forall.hpp"

#include <cmath>

namespace mfem
{
BatchInverseElementTransformation::BatchInverseElementTransformation()
{
   static Kernels kernels;
}
BatchInverseElementTransformation::BatchInverseElementTransformation(
   const GridFunction &gf, MemoryType d_mt)
   : BatchInverseElementTransformation()
{
   UpdateNodes(gf, d_mt);
}

BatchInverseElementTransformation::BatchInverseElementTransformation(
   const Mesh &mesh, MemoryType d_mt)
   : BatchInverseElementTransformation(*mesh.GetNodes(), d_mt) {}

BatchInverseElementTransformation::~BatchInverseElementTransformation() {}

void BatchInverseElementTransformation::UpdateNodes(const Mesh &mesh,
                                                    MemoryType d_mt)
{
   UpdateNodes(*mesh.GetNodes(), d_mt);
}

void BatchInverseElementTransformation::UpdateNodes(const GridFunction &gf,
                                                    MemoryType d_mt)
{
   MemoryType my_d_mt =
      (d_mt != MemoryType::DEFAULT) ? d_mt : Device::GetDeviceMemoryType();

   gf_ = &gf;
   const FiniteElementSpace *fespace = gf.FESpace();
   const int max_order = fespace->GetMaxElementOrder();
   const int ndof1d = max_order + 1;
   int ND = ndof1d;
   const int dim = fespace->GetMesh()->Dimension();
   MFEM_VERIFY(fespace->GetMesh()->GetNumGeometries(dim) <= 1,
               "Mixed meshes are not supported.");
   for (int d = 1; d < dim; ++d)
   {
      ND *= ndof1d;
   }
   const int vdim = fespace->GetVDim();
   const int NE = fespace->GetNE();
   node_pos.SetSize(vdim * ND * NE, my_d_mt);

   const FiniteElement *fe = fespace->GetTypicalFE();
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement *>(fe);
   if (fespace->IsVariableOrder() || tfe == nullptr)
   {
      MFEM_VERIFY(fe->GetGeomType() == Geometry::SEGMENT ||
                  fe->GetGeomType() == Geometry::SQUARE ||
                  fe->GetGeomType() == Geometry::CUBE,
                  "unsupported geometry type");
      // project onto GLL nodes
      basis_type = BasisType::GaussLobatto;
      points1d = poly1d.GetPointsArray(max_order, BasisType::GaussLobatto);
      points1d->HostRead();
      // either mixed order, or not a tensor basis
      node_pos.HostWrite();
      real_t tmp[3];
      int sdim = vdim;
      Vector pos(tmp, sdim);
      IntegrationPoint ip;
      int idcs[3];
      for (int e = 0; e < NE; ++e)
      {
         fe = fespace->GetFE(e);
         for (int i = 0; i < ND; ++i)
         {
            idcs[0] = i % ndof1d;
            idcs[1] = i / ndof1d;
            idcs[2] = idcs[1] / ndof1d;
            idcs[1] = idcs[1] % ndof1d;
            ip.x = (*points1d)[idcs[0]];
            ip.y = (*points1d)[idcs[1]];
            ip.z = (*points1d)[idcs[2]];
            gf.GetVectorValue(e, ip, pos);
            for (int d = 0; d < sdim; ++d)
            {
               node_pos[i + (d + e * sdim) * ND] = pos[d];
            }
         }
      }
   }
   else
   {
      const Operator *elem_restr =
         fespace->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      elem_restr->Mult(gf, node_pos);
      basis_type = tfe->GetBasisType();
      points1d = poly1d.GetPointsArray(max_order, basis_type);
   }
}

void BatchInverseElementTransformation::Transform(const Vector &pts,
                                                  const Array<int> &elems,
                                                  Array<int> &types,
                                                  Vector &refs, bool use_device,
                                                  Array<int> *iters) const
{
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      // no devices available
      use_device = false;
   }
   const FiniteElementSpace *fespace = gf_->FESpace();
   const FiniteElement *fe = fespace->GetTypicalFE();
   const int dim = fe->GetDim();
   const int vdim = fespace->GetVDim();
   const int NE = fespace->GetNE();
   const int order = fe->GetOrder();
   int npts = elems.Size();
   auto geom = fe->GetGeomType();

   types.SetSize(npts);
   refs.SetSize(npts * dim);

   auto pptr = pts.Read(use_device);
   auto eptr = elems.Read(use_device);
   auto mptr = node_pos.Read(use_device);
   auto tptr = types.Write(use_device);
   auto xptr = refs.ReadWrite(use_device);
   int* iter_ptr = nullptr;
   if (iters != nullptr)
   {
      iters->SetSize(npts);
      iter_ptr = iters->Write(use_device);
   }
   auto nptr = points1d->Read(use_device);
   int ndof1d = points1d->Size();

   switch (init_guess_type)
   {
      case InverseElementTransformation::Center:
      {
         real_t cx, cy, cz;
         auto ip0 = Geometries.GetCenter(geom);
         cx = ip0.x;
         cy = ip0.y;
         cz = ip0.z;
         switch (dim)
         {
            case 1:
               forall_switch(use_device, npts,
               [=] MFEM_HOST_DEVICE(int i) { xptr[i] = cx; });
               break;
            case 2:
               forall_switch(use_device, npts, [=] MFEM_HOST_DEVICE(int i)
               {
                  xptr[i] = cx;
                  xptr[i + npts] = cy;
               });
               break;
            case 3:
               forall_switch(use_device, npts, [=] MFEM_HOST_DEVICE(int i)
               {
                  xptr[i] = cx;
                  xptr[i + npts] = cy;
                  xptr[i + 2 * npts] = cz;
               });
               break;
         }
      } break;
      case InverseElementTransformation::ClosestRefNode:
      case InverseElementTransformation::ClosestPhysNode:
      {
         int nq1d = (qpts_order >= 0 ? qpts_order
                     : std::max(order + rel_qpts_order, 0)) +
                    1;
         int btype = BasisType::GetNodalBasis(guess_points_type);

         if ((btype == BasisType::Invalid || btype == basis_type) &&
             nq1d == ndof1d)
         {
            // special case: test points are basis nodal points
            if (init_guess_type ==
                InverseElementTransformation::ClosestPhysNode)
            {
               FindClosestPhysDof::Run(geom, vdim, use_device, npts, NE, ndof1d,
                                       mptr, pptr, eptr, nptr, xptr);
            }
            else
            {
               FindClosestRefDof::Run(geom, vdim, use_device, npts, NE, ndof1d, mptr,
                                      pptr, eptr, nptr, xptr);
            }
         }
         else
         {
            BasisType::Check(btype);
            auto qpoints = poly1d.GetPointsArray(nq1d - 1, btype);
            auto qptr = qpoints->Read(use_device);
            if (init_guess_type ==
                InverseElementTransformation::ClosestPhysNode)
            {
               FindClosestPhysPoint::Run(geom, vdim, use_device, npts, NE, ndof1d,
                                         nq1d, mptr, pptr, eptr, nptr, qptr,
                                         xptr);
            }
            else
            {
               FindClosestRefPoint::Run(geom, vdim, use_device, npts, NE, ndof1d,
                                        nq1d, mptr, pptr, eptr, nptr, qptr, xptr);
            }
         }
      } break;
      case InverseElementTransformation::GivenPoint:
         // nothing to do here
         break;
      case InverseElementTransformation::EdgeScan:
      {
         int nq1d = (qpts_order >= 0 ? qpts_order
                     : std::max(order + rel_qpts_order, 0)) +
                    1;
         int btype = BasisType::GetNodalBasis(guess_points_type);
         if (btype == BasisType::Invalid)
         {
            // default to closed uniform points
            btype = BasisType::ClosedUniform;
         }
         auto qpoints = poly1d.GetPointsArray(nq1d - 1, btype);
         auto qptr = qpoints->Read(use_device);
         NewtonEdgeScan::Run(geom, vdim, solver_type, use_device, ref_tol,
                             phys_rtol, max_iter, npts, NE, ndof1d, mptr, pptr,
                             eptr, nptr, qptr, nq1d, tptr, iter_ptr,
                             xptr);
      }
      return;
   }
   // general case: for each point, use guess inside refs
   NewtonSolve::Run(geom, vdim, solver_type, use_device, ref_tol, phys_rtol,
                    max_iter, npts, NE, ndof1d, mptr, pptr, eptr, nptr, tptr,
                    iter_ptr, xptr);
}

/// \cond DO_NOT_DOCUMENT
namespace internal
{
// data for batch inverse transform newton solvers
struct InvTNewtonSolverBase
{
   real_t ref_tol;
   real_t phys_rtol;
   // physical space coordinates of mesh element nodes
   const real_t *mptr;
   // physical space point coordinates to find
   const real_t *pptr;
   // element indices
   const int *eptr;
   // newton solve result code
   int *tptr;
   // number of iterations taken
   int *iter_ptr;
   // result ref coords
   real_t *xptr;
   eltrans::Lagrange basis1d;

   int max_iter;
   // number of points in pptr
   int npts;
};

// helper for computing dx = (pseudo)-inverse jac * [pt - F(x)]
template <int Dim, int SDim> struct InvTLinSolve;

template <> struct InvTLinSolve<1, 1>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      dx[0] = rhs[0] / jac[0];
   }
};

template <> struct InvTLinSolve<1, 2>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      auto a00 = jac[0];
      auto a10 = jac[1 * MFEM_THREAD_SIZE(x)];
      real_t den = a00 * a00 +
                   a10 * a10;
      dx[0] = (a00 * rhs[0] + a10 * rhs[1 * MFEM_THREAD_SIZE(x)]) / den;
   }
};

template <> struct InvTLinSolve<1, 3>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      auto a00 = jac[0];
      auto a10 = jac[1 * MFEM_THREAD_SIZE(x)];
      auto a20 = jac[2 * MFEM_THREAD_SIZE(x)];
      real_t den = a00 * a00 + a10 * a10 + a20 * a20;
      dx[0] = (a00 * rhs[0] + a10 * rhs[1 * MFEM_THREAD_SIZE(x)] +
               a20 * rhs[2 * MFEM_THREAD_SIZE(x)]) /
              den;
   }
};

template <> struct InvTLinSolve<2, 2>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      auto a00 = jac[(0 + 0 * 2) * MFEM_THREAD_SIZE(x)];
      auto a10 = jac[(1 + 0 * 2) * MFEM_THREAD_SIZE(x)];
      auto a01 = jac[(0 + 1 * 2) * MFEM_THREAD_SIZE(x)];
      auto a11 = jac[(1 + 1 * 2) * MFEM_THREAD_SIZE(x)];
      real_t den = 1 / (a00 * a11 - a01 * a10);
      dx[0] = (a11 * rhs[0 * MFEM_THREAD_SIZE(x)] -
               a01 * rhs[1 * MFEM_THREAD_SIZE(x)]) *
              den;
      dx[1] = (a00 * rhs[1 * MFEM_THREAD_SIZE(x)] -
               a10 * rhs[0 * MFEM_THREAD_SIZE(x)]) *
              den;
   }
};

template <> struct InvTLinSolve<2, 3>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      auto a00 = jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)];
      auto a01 = jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)];
      auto a10 = jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)];
      auto a11 = jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)];
      auto a20 = jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)];
      auto a21 = jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)];
      // a00**2*a11**2 + a00**2*a21**2 - 2*a00*a01*a10*a11 - 2*a00*a01*a20*a21 +
      // a01**2*a10**2 + a01**2*a20**2 + a10**2*a21**2 - 2*a10*a11*a20*a21 +
      // a11**2*a20**2
      real_t den = 1 / (a00 * a00 * a11 * a11 + a00 * a00 * a21 * a21 -
                        2 * a00 * a01 * a10 * a11 - 2 * a00 * a01 * a20 * a21 +
                        a01 * a01 * a10 * a10 + a01 * a01 * a20 * a20 +
                        a10 * a10 * a21 * a21 - 2 * a10 * a11 * a20 * a21 +
                        a11 * a11 * a20 * a20);
      //   x0*(a00*(a01**2 + a11**2 + a21**2) - a01*(a00*a01 + a10*a11 + a20*a21))
      // + x1*(a10*(a01**2 + a11**2 + a21**2) - a11*(a00*a01 + a10*a11 + a20*a21))
      // + x2*(a20*(a01**2 + a11**2 + a21**2) - a21*(a00*a01 + a10*a11 + a20*a21))
      dx[0] = (rhs[0 * MFEM_THREAD_SIZE(x)] *
               (a00 * (a01 * a01 + a11 * a11 + a21 * a21) -
                a01 * (a00 * a01 + a10 * a11 + a20 * a21)) +
               rhs[1 * MFEM_THREAD_SIZE(x)] *
               (a10 * (a01 * a01 + a11 * a11 + a21 * a21) -
                a11 * (a00 * a01 + a10 * a11 + a20 * a21)) +
               rhs[2 * MFEM_THREAD_SIZE(x)] *
               (a20 * (a01 * a01 + a11 * a11 + a21 * a21) -
                a21 * (a00 * a01 + a10 * a11 + a20 * a21))) *
              den;
      //  x0*(a01*(a00**2 + a10**2 + a20**2)-a00*(a00*a01 + a10*a11 + a20*a21))
      // +x1*(a11*(a00**2 + a10**2 + a20**2)-a10*(a00*a01 + a10*a11 + a20*a21))
      // +x2*(a21*(a00**2 + a10**2 + a20**2)-a20*(a00*a01 + a10*a11 + a20*a21))
      dx[1] = (rhs[0 * MFEM_THREAD_SIZE(x)] *
               (a01 * (a00 * a00 + a10 * a10 + a20 * a20) -
                a00 * (a00 * a01 + a10 * a11 + a20 * a21)) +
               rhs[1 * MFEM_THREAD_SIZE(x)] *
               (a11 * (a00 * a00 + a10 * a10 + a20 * a20) -
                a10 * (a00 * a01 + a10 * a11 + a20 * a21)) +
               rhs[2 * MFEM_THREAD_SIZE(x)] *
               (a21 * (a00 * a00 + a10 * a10 + a20 * a20) -
                a20 * (a00 * a01 + a10 * a11 + a20 * a21))) *
              den;
   }
};

template <> struct InvTLinSolve<3, 3>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      auto a00 = jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)];
      auto a01 = jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)];
      auto a02 = jac[(0 + 2 * 3) * MFEM_THREAD_SIZE(x)];
      auto a10 = jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)];
      auto a11 = jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)];
      auto a12 = jac[(1 + 2 * 3) * MFEM_THREAD_SIZE(x)];
      auto a20 = jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)];
      auto a21 = jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)];
      auto a22 = jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)];

      real_t den = 1 / (a00 * a11 * a22 - a00 * a12 * a22 - a01 * a10 * a22 +
                        a01 * a12 * a20 + a02 * a10 * a21 - a02 * a11 * a20);
      dx[0] = (rhs[0 * MFEM_THREAD_SIZE(x)] * (a11 * a22 - a12 * a21) -
               rhs[1 * MFEM_THREAD_SIZE(x)] * (a01 * a22 - a02 * a21) +
               rhs[2 * MFEM_THREAD_SIZE(x)] * (a01 * a12 - a02 * a11)) *
              den;
      dx[1] = (rhs[0 * MFEM_THREAD_SIZE(x)] * (a12 * a20 - a10 * a22) +
               rhs[1 * MFEM_THREAD_SIZE(x)] * (a00 * a22 - a02 * a20) -
               rhs[2 * MFEM_THREAD_SIZE(x)] * (a00 * a12 - a02 * a10)) *
              den;
      dx[2] = (rhs[0 * MFEM_THREAD_SIZE(x)] * (a10 * a21 - a11 * a20) -
               rhs[1 * MFEM_THREAD_SIZE(x)] * (a00 * a21 - a01 * a20) +
               rhs[2 * MFEM_THREAD_SIZE(x)] * (a00 * a11 - a01 * a10)) *
              den;
   }
};

template <int Geom, InverseElementTransformation::SolverType SolverType>
struct ProjectType;

template <>
struct ProjectType<Geometry::SEGMENT, InverseElementTransformation::Newton>
{
   static MFEM_HOST_DEVICE bool project(real_t &x, real_t &dx)
   {
      x += dx;
      return false;
   }
};

template <>
struct ProjectType<Geometry::SQUARE, InverseElementTransformation::Newton>
{
   static MFEM_HOST_DEVICE bool project(real_t &x, real_t &y, real_t &dx,
                                        real_t &dy)
   {
      x += dx;
      y += dy;
      return false;
   }
};

template <>
struct ProjectType<Geometry::CUBE, InverseElementTransformation::Newton>
{
   static MFEM_HOST_DEVICE bool project(real_t &x, real_t &y, real_t &z,
                                        real_t &dx, real_t &dy, real_t &dz)
   {
      x += dx;
      y += dy;
      z += dz;
      return false;
   }
};

template <int Geom>
struct ProjectType<Geom, InverseElementTransformation::NewtonElementProject>
{
   template <class... Ts> static MFEM_HOST_DEVICE bool project(Ts &&...args)
   {
      return eltrans::GeometryUtils<Geom>::project(args...);
   }
};

template <int Geom, int SDim,
          InverseElementTransformation::SolverType SolverType, int max_team_x>
struct InvTNewtonSolver;

template <int SDim, InverseElementTransformation::SolverType SType,
          int max_team_x>
struct InvTNewtonSolver<Geometry::SEGMENT, SDim, SType, max_team_x>
   : public InvTNewtonSolverBase
{
   static int ndofs(int ndof1d) { return ndof1d; }

   // theoretically unbounded
   static constexpr MFEM_HOST_DEVICE int max_dof1d() { return 0x1000; }

   int MFEM_HOST_DEVICE operator()(int idx) const
   {
      // parallelize one team per pt
      constexpr int Dim = 1;
      int iter = 0;
      MFEM_SHARED real_t ref_coord[Dim];
      // contiguous in team_x, then SDim
      MFEM_SHARED real_t phys_coord[SDim * max_team_x];
      // contiguous in team_x, SDim, then Dim
      MFEM_SHARED real_t jac[SDim * Dim * max_team_x];
      MFEM_SHARED bool term_flag[1];
      MFEM_SHARED int res[1];
      MFEM_SHARED real_t dx[Dim];
      MFEM_SHARED real_t prev_dx[Dim];
      MFEM_SHARED bool hit_bdr[1];
      MFEM_SHARED bool prev_hit_bdr[1];
      real_t phys_tol = 0;
      if (MFEM_THREAD_ID(x) == 0)
      {
         term_flag[0] = false;
         res[0] = InverseElementTransformation::Unknown;
         for (int d = 0; d < Dim; ++d)
         {
            ref_coord[d] = xptr[idx + d * npts];
            dx[d] = 0;
            prev_dx[d] = 0;
         }
         for (int d = 0; d < SDim; ++d)
         {
            phys_tol += pptr[idx + d * npts] * pptr[idx + d * npts];
         }
         phys_tol = fmax(phys_rtol * phys_rtol, phys_tol * phys_rtol * phys_rtol);
         hit_bdr[0] = prev_hit_bdr[0] = false;
      }
      // for each iteration
      while (true)
      {
         MFEM_SYNC_THREAD;
         // compute phys_coord and jacobian at the same time
         for (int d = 0; d < SDim; ++d)
         {
            phys_coord[MFEM_THREAD_ID(x) + d * MFEM_THREAD_SIZE(x)] = 0;
         }
         for (int i = 0; i < SDim * Dim; ++i)
         {
            jac[MFEM_THREAD_ID(x) + i * MFEM_THREAD_SIZE(x)] = 0;
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(j0, x, basis1d.pN)
         {
            real_t b0, db0;
            basis1d.eval_d1(b0, db0, ref_coord[0], j0);
            for (int d = 0; d < SDim; ++d)
            {
               phys_coord[MFEM_THREAD_ID(x) + d * MFEM_THREAD_SIZE(x)] +=
                  mptr[j0 + (eptr[idx] * SDim + d) * basis1d.pN] * b0;
               jac[MFEM_THREAD_ID(x) + (d + 0 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[j0 + (eptr[idx] * SDim + d) * basis1d.pN] * db0;
            }
         }

         for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
         {
            MFEM_SYNC_THREAD;
            int a = MFEM_THREAD_ID(x);
            int b = a + i;
            if (a < i && b < basis1d.pN)
            {
               for (int d = 0; d < SDim; ++d)
               {
                  phys_coord[a + d * MFEM_THREAD_SIZE(x)] +=
                     phys_coord[b + d * MFEM_THREAD_SIZE(x)];
               }
               for (int j = 0; j < SDim * Dim; ++j)
               {
                  jac[a + j * MFEM_THREAD_SIZE(x)] +=
                     jac[b + j * MFEM_THREAD_SIZE(x)];
               }
            }
         }
         MFEM_SYNC_THREAD;

         // rest of newton solve logic is serial, have thread 0 solve for it
         if (MFEM_THREAD_ID(x) == 0)
         {
            // compute objective function
            // f(x) = 1/2 |pt - F(x)|^2
            real_t dist = 0;
            for (int d = 0; d < SDim; ++d)
            {
               real_t tmp =
                  pptr[idx + d * npts] - phys_coord[d * MFEM_THREAD_SIZE(x)];
               phys_coord[d * MFEM_THREAD_SIZE(x)] = tmp;
               dist += tmp * tmp;
            }
            // phys_coord now contains pt - F(x)
            // check for phys_tol convergence
            if (dist <= phys_tol)
            {
               // found solution
               res[0] = eltrans::GeometryUtils<Geometry::SEGMENT>::inside(
                           ref_coord[0])
                        ? InverseElementTransformation::Inside
                        : InverseElementTransformation::Outside;
               tptr[idx] = res[0];
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               if (iter_ptr)
               {
                  iter_ptr[idx] = iter + 1;
               }
               term_flag[0] = true;
            }
            else if (iter >= max_iter)
            {
               // terminate on max iterations
               tptr[idx] = InverseElementTransformation::Unknown;
               res[0] = InverseElementTransformation::Unknown;
               // might as well save where we failed at
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               if (iter_ptr)
               {
                  iter_ptr[idx] = iter + 1;
               }
               term_flag[0] = true;
            }
            else
            {
               // compute dx = (pseudo)-inverse jac * [pt - F(x)]
               InvTLinSolve<Dim, SDim>::solve(jac, phys_coord, dx);

               hit_bdr[0] = ProjectType<Geometry::SEGMENT, SType>::project(
                               ref_coord[0], dx[0]);

               // check for ref coord convergence or stagnation on boundary
               if (hit_bdr[0])
               {
                  if (prev_hit_bdr[0])
                  {
                     real_t dx_change = 0;
                     for (int d = 0; d < Dim; ++d)
                     {
                        real_t tmp = dx[d] - prev_dx[d];
                        dx_change += tmp * tmp;
                     }
                     if (dx_change <= ref_tol * ref_tol)
                     {
                        // stuck on the boundary
                        tptr[idx] = InverseElementTransformation::Outside;
                        res[0] = InverseElementTransformation::Outside;
                        for (int d = 0; d < Dim; ++d)
                        {
                           xptr[idx + d * npts] = ref_coord[d];
                        }
                        if (iter_ptr)
                        {
                           iter_ptr[idx] = iter + 1;
                        }
                        term_flag[0] = true;
                     }
                  }
               }

               prev_hit_bdr[0] = hit_bdr[0];
            }
         }

         MFEM_SYNC_THREAD;
         if (term_flag[0])
         {
            return res[0];
         }
         MFEM_FOREACH_THREAD(d, x, Dim) { prev_dx[d] = dx[d]; }
         ++iter;
      }
   }
};

template <int SDim, InverseElementTransformation::SolverType SType,
          int max_team_x>
struct InvTNewtonSolver<Geometry::SQUARE, SDim, SType, max_team_x>
   : public InvTNewtonSolverBase
{
   static int ndofs(int ndof1d) { return ndof1d * ndof1d; }

   static constexpr MFEM_HOST_DEVICE int max_dof1d() { return 32; }

   int MFEM_HOST_DEVICE operator()(int idx) const
   {
      // parallelize one team per pt
      constexpr int Dim = 2;
      int iter = 0;
      MFEM_SHARED real_t ref_coord[Dim];
      // contiguous in team_x, then SDim
      MFEM_SHARED real_t phys_coord[SDim * max_team_x];
      MFEM_SHARED real_t basis0[max_dof1d()];
      MFEM_SHARED real_t dbasis0[max_dof1d()];
      MFEM_SHARED real_t basis1[max_dof1d()];
      MFEM_SHARED real_t dbasis1[max_dof1d()];
      // contiguous in team_x, SDim, then Dim
      MFEM_SHARED real_t jac[SDim * Dim * max_team_x];
      MFEM_SHARED bool term_flag[1];
      MFEM_SHARED int res[1];
      MFEM_SHARED real_t dx[Dim];
      MFEM_SHARED real_t prev_dx[Dim];
      MFEM_SHARED bool hit_bdr[1];
      MFEM_SHARED bool prev_hit_bdr[1];
      real_t phys_tol = 0;
      if (MFEM_THREAD_ID(x) == 0)
      {
         term_flag[0] = false;
         res[0] = InverseElementTransformation::Unknown;
         hit_bdr[0] = false;
         prev_hit_bdr[0] = false;
         for (int d = 0; d < Dim; ++d)
         {
            ref_coord[d] = xptr[idx + d * npts];
            dx[d] = 0;
            prev_dx[d] = 0;
         }
         for (int d = 0; d < SDim; ++d)
         {
            phys_tol += pptr[idx + d * npts] * pptr[idx + d * npts];
         }
         phys_tol = fmax(phys_rtol * phys_rtol, phys_tol * phys_rtol * phys_rtol);
      }
      // for each iteration
      while (true)
      {
         MFEM_SYNC_THREAD;
         // compute phys_coord and jacobian at the same time
         for (int d = 0; d < SDim; ++d)
         {
            phys_coord[MFEM_THREAD_ID(x) + d * MFEM_THREAD_SIZE(x)] = 0;
         }
         for (int i = 0; i < SDim * Dim; ++i)
         {
            jac[MFEM_THREAD_ID(x) + i * MFEM_THREAD_SIZE(x)] = 0;
         }
         MFEM_FOREACH_THREAD(j0, x, basis1d.pN)
         {
            basis1d.eval_d1(basis0[j0], dbasis0[j0], ref_coord[0], j0);
            basis1d.eval_d1(basis1[j0], dbasis1[j0], ref_coord[1], j0);
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(jidx, x, basis1d.pN * basis1d.pN)
         {
            int idcs[Dim];
            idcs[0] = jidx % basis1d.pN;
            idcs[1] = jidx / basis1d.pN;
            for (int d = 0; d < SDim; ++d)
            {
               phys_coord[MFEM_THREAD_ID(x) + d * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + (eptr[idx] * SDim + d) * basis1d.pN) *
                               basis1d.pN] *
                  basis0[idcs[0]] * basis1[idcs[1]];
               jac[MFEM_THREAD_ID(x) + (d + 0 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + (eptr[idx] * SDim + d) * basis1d.pN) *
                               basis1d.pN] *
                  dbasis0[idcs[0]] * basis1[idcs[1]];
               jac[MFEM_THREAD_ID(x) + (d + 1 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + (eptr[idx] * SDim + d) * basis1d.pN) *
                               basis1d.pN] *
                  basis0[idcs[0]] * dbasis1[idcs[1]];
            }
         }

         for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
         {
            MFEM_SYNC_THREAD;
            int a = MFEM_THREAD_ID(x);
            int b = a + i;
            if (a < i && b < basis1d.pN * basis1d.pN)
            {
               for (int d = 0; d < SDim; ++d)
               {
                  phys_coord[a + d * MFEM_THREAD_SIZE(x)] +=
                     phys_coord[b + d * MFEM_THREAD_SIZE(x)];
               }
               for (int j = 0; j < SDim * Dim; ++j)
               {
                  jac[a + j * MFEM_THREAD_SIZE(x)] +=
                     jac[b + j * MFEM_THREAD_SIZE(x)];
               }
            }
         }
         MFEM_SYNC_THREAD;

         // rest of newton solve logic is serial, have thread 0 solve for it
         if (MFEM_THREAD_ID(x) == 0)
         {
            // compute objective function
            // f(x) = 1/2 |pt - F(x)|^2
            real_t dist = 0;
            for (int d = 0; d < SDim; ++d)
            {
               real_t tmp =
                  pptr[idx + d * npts] - phys_coord[d * MFEM_THREAD_SIZE(x)];
               phys_coord[d * MFEM_THREAD_SIZE(x)] = tmp;
               dist += tmp * tmp;
            }
            // phys_coord now contains pt - F(x)
            // check for phys_tol convergence
            if (dist <= phys_tol)
            {
               // found solution
               res[0] = eltrans::GeometryUtils<Geometry::SQUARE>::inside(
                           ref_coord[0], ref_coord[1])
                        ? InverseElementTransformation::Inside
                        : InverseElementTransformation::Outside;
               tptr[idx] = res[0];
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               if (iter_ptr)
               {
                  iter_ptr[idx] = iter + 1;
               }
               term_flag[0] = true;
            }
            else if (iter >= max_iter)
            {
               // terminate on max iterations
               tptr[idx] = InverseElementTransformation::Unknown;
               res[0] = InverseElementTransformation::Unknown;
               // might as well save where we failed at
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               if (iter_ptr)
               {
                  iter_ptr[idx] = iter + 1;
               }
               term_flag[0] = true;
            }
            else
            {
               // compute dx = (pseudo)-inverse jac * [pt - F(x)]
               InvTLinSolve<Dim, SDim>::solve(jac, phys_coord, dx);

               hit_bdr[0] = ProjectType<Geometry::SQUARE, SType>::project(
                               ref_coord[0], ref_coord[1], dx[0], dx[1]);

               // check for ref coord convergence or stagnation on boundary
               if (hit_bdr[0])
               {
                  if (prev_hit_bdr[0])
                  {
                     real_t dx_change = 0;
                     for (int d = 0; d < Dim; ++d)
                     {
                        real_t tmp = dx[d] - prev_dx[d];
                        dx_change += tmp * tmp;
                     }
                     if (dx_change <= ref_tol * ref_tol)
                     {
                        // stuck on the boundary
                        tptr[idx] = InverseElementTransformation::Outside;
                        res[0] = InverseElementTransformation::Outside;
                        for (int d = 0; d < Dim; ++d)
                        {
                           xptr[idx + d * npts] = ref_coord[d];
                        }
                        if (iter_ptr)
                        {
                           iter_ptr[idx] = iter + 1;
                        }
                        term_flag[0] = true;
                     }
                  }
               }

               prev_hit_bdr[0] = hit_bdr[0];
            }
         }

         MFEM_SYNC_THREAD;
         if (term_flag[0])
         {
            return res[0];
         }
         MFEM_FOREACH_THREAD(d, x, Dim) { prev_dx[d] = dx[d]; }
         MFEM_SYNC_THREAD;
         ++iter;
      }
   }
};

template <int SDim, InverseElementTransformation::SolverType SType,
          int max_team_x>
struct InvTNewtonSolver<Geometry::CUBE, SDim, SType, max_team_x>
   : public InvTNewtonSolverBase
{
   static int ndofs(int ndof1d) { return ndof1d * ndof1d * ndof1d; }

   static constexpr MFEM_HOST_DEVICE int max_dof1d() { return 32; }

   int MFEM_HOST_DEVICE operator()(int idx) const
   {
      // parallelize one team per pt
      constexpr int Dim = 3;
      int iter = 0;
      MFEM_SHARED real_t ref_coord[Dim];
      // contiguous in team_x, then SDim
      MFEM_SHARED real_t phys_coord[SDim * max_team_x];
      MFEM_SHARED real_t basis0[max_dof1d()];
      MFEM_SHARED real_t dbasis0[max_dof1d()];
      MFEM_SHARED real_t basis1[max_dof1d()];
      MFEM_SHARED real_t dbasis1[max_dof1d()];
      MFEM_SHARED real_t basis2[max_dof1d()];
      MFEM_SHARED real_t dbasis2[max_dof1d()];
      // contiguous in team_x, SDim, then Dim
      MFEM_SHARED real_t jac[SDim * Dim * max_team_x];
      MFEM_SHARED bool term_flag[1];
      MFEM_SHARED int res[1];
      MFEM_SHARED real_t dx[Dim];
      MFEM_SHARED real_t prev_dx[Dim];
      MFEM_SHARED bool hit_bdr[1];
      MFEM_SHARED bool prev_hit_bdr[1];
      real_t phys_tol = 0;
      if (MFEM_THREAD_ID(x) == 0)
      {
         term_flag[0] = false;
         res[0] = InverseElementTransformation::Unknown;
         hit_bdr[0] = false;
         prev_hit_bdr[0] = false;
         for (int d = 0; d < Dim; ++d)
         {
            ref_coord[d] = xptr[idx + d * npts];
            dx[d] = 0;
            prev_dx[d] = 0;
         }
         for (int d = 0; d < SDim; ++d)
         {
            phys_tol += pptr[idx + d * npts] * pptr[idx + d * npts];
         }
         phys_tol = fmax(phys_rtol * phys_rtol, phys_tol * phys_rtol * phys_rtol);
      }
      // for each iteration
      while (true)
      {
         MFEM_SYNC_THREAD;
         // compute phys_coord and jacobian at the same time
         for (int d = 0; d < SDim; ++d)
         {
            phys_coord[MFEM_THREAD_ID(x) + d * MFEM_THREAD_SIZE(x)] = 0;
         }
         for (int i = 0; i < SDim * Dim; ++i)
         {
            jac[MFEM_THREAD_ID(x) + i * MFEM_THREAD_SIZE(x)] = 0;
         }
         MFEM_FOREACH_THREAD(j0, x, basis1d.pN)
         {
            basis1d.eval_d1(basis0[j0], dbasis0[j0], ref_coord[0], j0);
            basis1d.eval_d1(basis1[j0], dbasis1[j0], ref_coord[1], j0);
            basis1d.eval_d1(basis2[j0], dbasis2[j0], ref_coord[2], j0);
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(jidx, x, basis1d.pN * basis1d.pN * basis1d.pN)
         {
            int idcs[Dim];
            idcs[0] = jidx % basis1d.pN;
            idcs[1] = jidx / basis1d.pN;
            idcs[2] = idcs[1] / basis1d.pN;
            idcs[1] %= basis1d.pN;
            for (int d = 0; d < SDim; ++d)
            {
               phys_coord[MFEM_THREAD_ID(x) + d * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] + (idcs[1] + (idcs[2] + (eptr[idx] * SDim + d) *
                                             basis1d.pN) *
                                  basis1d.pN) *
                               basis1d.pN] *
                  basis0[idcs[0]] * basis1[idcs[1]] * basis2[idcs[2]];
               jac[MFEM_THREAD_ID(x) + (d + 0 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] + (idcs[1] + (idcs[2] + (eptr[idx] * SDim + d) *
                                             basis1d.pN) *
                                  basis1d.pN) *
                               basis1d.pN] *
                  dbasis0[idcs[0]] * basis1[idcs[1]] * basis2[idcs[2]];
               jac[MFEM_THREAD_ID(x) + (d + 1 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] + (idcs[1] + (idcs[2] + (eptr[idx] * SDim + d) *
                                             basis1d.pN) *
                                  basis1d.pN) *
                               basis1d.pN] *
                  basis0[idcs[0]] * dbasis1[idcs[1]] * basis2[idcs[2]];
               jac[MFEM_THREAD_ID(x) + (d + 2 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] + (idcs[1] + (idcs[2] + (eptr[idx] * SDim + d) *
                                             basis1d.pN) *
                                  basis1d.pN) *
                               basis1d.pN] *
                  basis0[idcs[0]] * basis1[idcs[1]] * dbasis2[idcs[2]];
            }
         }

         for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
         {
            MFEM_SYNC_THREAD;
            int a = MFEM_THREAD_ID(x);
            int b = a + i;
            if (a < i && b < basis1d.pN * basis1d.pN * basis1d.pN)
            {
               for (int d = 0; d < SDim; ++d)
               {
                  phys_coord[a + d * MFEM_THREAD_SIZE(x)] +=
                     phys_coord[b + d * MFEM_THREAD_SIZE(x)];
               }
               for (int j = 0; j < SDim * Dim; ++j)
               {
                  jac[a + j * MFEM_THREAD_SIZE(x)] +=
                     jac[b + j * MFEM_THREAD_SIZE(x)];
               }
            }
         }
         MFEM_SYNC_THREAD;

         // rest of newton solve logic is serial, have thread 0 solve for it
         if (MFEM_THREAD_ID(x) == 0)
         {
            // compute objective function
            // f(x) = 1/2 |pt - F(x)|^2
            real_t dist = 0;
            for (int d = 0; d < SDim; ++d)
            {
               real_t tmp =
                  pptr[idx + d * npts] - phys_coord[d * MFEM_THREAD_SIZE(x)];
               phys_coord[d * MFEM_THREAD_SIZE(x)] = tmp;
               dist += tmp * tmp;
            }
            // phys_coord now contains pt - F(x)
            // check for phys_tol convergence
            if (dist <= phys_tol)
            {
               // found solution
               res[0] = eltrans::GeometryUtils<Geometry::CUBE>::inside(
                           ref_coord[0], ref_coord[1], ref_coord[2])
                        ? InverseElementTransformation::Inside
                        : InverseElementTransformation::Outside;
               tptr[idx] = res[0];
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               if (iter_ptr)
               {
                  iter_ptr[idx] = iter + 1;
               }
               term_flag[0] = true;
            }
            else if (iter >= max_iter)
            {
               // terminate on max iterations
               tptr[idx] = InverseElementTransformation::Unknown;
               res[0] = InverseElementTransformation::Unknown;
               // might as well save where we failed at
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               if (iter_ptr)
               {
                  iter_ptr[idx] = iter + 1;
               }
               term_flag[0] = true;
            }
            else
            {
               // compute dx = (pseudo)-inverse jac * [pt - F(x)]
               InvTLinSolve<Dim, SDim>::solve(jac, phys_coord, dx);

               hit_bdr[0] = ProjectType<Geometry::CUBE, SType>::project(
                               ref_coord[0], ref_coord[1], ref_coord[2], dx[0], dx[1],
                               dx[2]);

               // check for ref coord convergence or stagnation on boundary
               if (hit_bdr[0])
               {
                  if (prev_hit_bdr[0])
                  {
                     real_t dx_change = 0;
                     for (int d = 0; d < Dim; ++d)
                     {
                        real_t tmp = dx[d] - prev_dx[d];
                        dx_change += tmp * tmp;
                     }
                     if (dx_change <= ref_tol * ref_tol)
                     {
                        // stuck on the boundary
                        tptr[idx] = InverseElementTransformation::Outside;
                        res[0] = InverseElementTransformation::Outside;
                        for (int d = 0; d < Dim; ++d)
                        {
                           xptr[idx + d * npts] = ref_coord[d];
                        }
                        if (iter_ptr)
                        {
                           iter_ptr[idx] = iter + 1;
                        }
                        term_flag[0] = true;
                     }
                  }
               }
            }
         }

         MFEM_SYNC_THREAD;
         if (term_flag[0])
         {
            return res[0];
         }
         MFEM_FOREACH_THREAD(d, x, Dim) { prev_dx[d] = dx[d]; }
         ++iter;
      }
   }
};

// data for finding the batch transform initial guess co-located at dofs
struct DofFinderBase
{
   // physical space coordinates of mesh element nodes
   const real_t *mptr;
   // physical space point coordinates to find
   const real_t *pptr;
   // element indices
   const int *eptr;
   // reference space nodes to test
   const real_t *qptr;
   // initial guess results
   real_t *xptr;
   eltrans::Lagrange basis1d;

   // number of points in pptr
   int npts;
};

template <int Geom, int SDim, int max_team_x>
struct PhysDofFinder;

template <int SDim, int max_team_x>
struct PhysDofFinder<Geometry::SEGMENT, SDim, max_team_x>
   : public DofFinderBase
{
   static int ndofs(int ndofs1d) { return ndofs1d; }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 1;
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      MFEM_FOREACH_THREAD(i, x, max_team_x)
      {
#ifdef MFEM_USE_DOUBLE
         dists[i] = HUGE_VAL;
#else
         dists[i] = HUGE_VALF;
#endif
      }
      MFEM_SYNC_THREAD;
      // team serial portion
      MFEM_FOREACH_THREAD(i, x, basis1d.pN)
      {
         real_t phys_coord[SDim];
         for (int d = 0; d < SDim; ++d)
         {
            phys_coord[d] = mptr[i + (d + eptr[idx] * SDim) * basis1d.pN];
         }
         real_t dist = 0;
         // L-2 norm squared
         for (int d = 0; d < SDim; ++d)
         {
            real_t tmp = phys_coord[d] - pptr[idx + d * npts];
            dist += tmp * tmp;
         }
         if (dist < dists[MFEM_THREAD_ID(x)])
         {
            // closer guess in physical space
            dists[MFEM_THREAD_ID(x)] = dist;
            ref_buf[MFEM_THREAD_ID(x)] = basis1d.z[i];
         }
      }
      // now do tree reduce
      for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
      {
         MFEM_SYNC_THREAD;
         if (MFEM_THREAD_ID(x) < i)
         {
            if (dists[MFEM_THREAD_ID(x) + i] < dists[MFEM_THREAD_ID(x)])
            {
               dists[MFEM_THREAD_ID(x)] = dists[MFEM_THREAD_ID(x) + i];
               ref_buf[MFEM_THREAD_ID(x)] = ref_buf[MFEM_THREAD_ID(x) + i];
            }
         }
      }
      // write results out
      // not needed in 1D
      // MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(x) == 0)
      {
         xptr[idx] = ref_buf[0];
      }
   }
};

template <int SDim, int max_team_x>
struct PhysDofFinder<Geometry::SQUARE, SDim, max_team_x>
   : public DofFinderBase
{
   static int ndofs(int ndofs1d) { return ndofs1d * ndofs1d; }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 2;
      int n = basis1d.pN * basis1d.pN;
      if (n > max_team_x)
      {
         n = max_team_x;
      }
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      MFEM_FOREACH_THREAD(i, x, max_team_x)
      {
#ifdef MFEM_USE_DOUBLE
         dists[i] = HUGE_VAL;
#else
         dists[i] = HUGE_VALF;
#endif
      }
      // team serial portion
      MFEM_FOREACH_THREAD(j, x, basis1d.pN * basis1d.pN)
      {
         real_t phys_coord[SDim] = {0};
         int idcs[Dim];
         idcs[0] = j % basis1d.pN;
         idcs[1] = j / basis1d.pN;
         for (int d = 0; d < SDim; ++d)
         {
            phys_coord[d] =
               mptr[idcs[0] + (idcs[1] + (d + eptr[idx] * SDim) * basis1d.pN) *
                            basis1d.pN];
         }
         real_t dist = 0;
         // L-2 norm squared
         for (int d = 0; d < SDim; ++d)
         {
            real_t tmp = pptr[idx + d * npts] - phys_coord[d];
            dist += tmp * tmp;
         }
         if (dist < dists[MFEM_THREAD_ID(x)])
         {
            // closer guess in physical space
            dists[MFEM_THREAD_ID(x)] = dist;
            for (int d = 0; d < Dim; ++d)
            {
               ref_buf[MFEM_THREAD_ID(x) + d * n] = basis1d.z[idcs[d]];
            }
         }
      }
      // now do tree reduce
      for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
      {
         MFEM_SYNC_THREAD;
         if (MFEM_THREAD_ID(x) < i)
         {
            if (dists[MFEM_THREAD_ID(x) + i] < dists[MFEM_THREAD_ID(x)])
            {
               dists[MFEM_THREAD_ID(x)] = dists[MFEM_THREAD_ID(x) + i];
               for (int d = 0; d < Dim; ++d)
               {
                  ref_buf[MFEM_THREAD_ID(x) + d * n] =
                     ref_buf[MFEM_THREAD_ID(x) + i + d * n];
               }
            }
         }
      }
      // write results out
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d, x, Dim) { xptr[idx + d * npts] = ref_buf[d * n]; }
   }
};

template <int SDim, int max_team_x>
struct PhysDofFinder<Geometry::CUBE, SDim, max_team_x>
   : public DofFinderBase
{
   static int ndofs(int ndofs1d) { return ndofs1d * ndofs1d * ndofs1d; }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 3;
      int n = basis1d.pN * basis1d.pN * basis1d.pN;
      if (n > max_team_x)
      {
         n = max_team_x;
      }
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      MFEM_FOREACH_THREAD(i, x, max_team_x)
      {
#ifdef MFEM_USE_DOUBLE
         dists[i] = HUGE_VAL;
#else
         dists[i] = HUGE_VALF;
#endif
      }
      // team serial portion
      MFEM_FOREACH_THREAD(j, x, basis1d.pN * basis1d.pN * basis1d.pN)
      {
         real_t phys_coord[SDim] = {0};
         int idcs[Dim];
         idcs[0] = j % basis1d.pN;
         idcs[1] = j / basis1d.pN;
         idcs[2] = idcs[1] / basis1d.pN;
         idcs[1] = idcs[1] % basis1d.pN;
         for (int d = 0; d < SDim; ++d)
         {
            phys_coord[d] =
               mptr[idcs[0] + (idcs[1] + (idcs[2] + (d + eptr[idx] * SDim) *
                                          basis1d.pN) *
                               basis1d.pN) *
                            basis1d.pN];
         }
         real_t dist = 0;
         // L-2 norm squared
         for (int d = 0; d < SDim; ++d)
         {
            real_t tmp = pptr[idx + d * npts] - phys_coord[d];
            dist += tmp * tmp;
         }
         if (dist < dists[MFEM_THREAD_ID(x)])
         {
            // closer guess in physical space
            dists[MFEM_THREAD_ID(x)] = dist;
            for (int d = 0; d < Dim; ++d)
            {
               ref_buf[MFEM_THREAD_ID(x) + d * n] = basis1d.z[idcs[d]];
            }
         }
      }
      // now do tree reduce
      for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
      {
         MFEM_SYNC_THREAD;
         if (MFEM_THREAD_ID(x) < i)
         {
            if (dists[MFEM_THREAD_ID(x) + i] < dists[MFEM_THREAD_ID(x)])
            {
               dists[MFEM_THREAD_ID(x)] = dists[MFEM_THREAD_ID(x) + i];
               for (int d = 0; d < Dim; ++d)
               {
                  ref_buf[MFEM_THREAD_ID(x) + d * n] =
                     ref_buf[MFEM_THREAD_ID(x) + i + d * n];
               }
            }
         }
      }
      // write results out
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d, x, Dim) { xptr[idx + d * npts] = ref_buf[d * n]; }
   }
};

// data for finding the batch transform initial guess
struct NodeFinderBase
{
   // physical space coordinates of mesh element nodes
   const real_t *mptr;
   // physical space point coordinates to find
   const real_t *pptr;
   // element indices
   const int *eptr;
   // reference space nodes to test
   const real_t *qptr;
   // initial guess results
   real_t *xptr;
   eltrans::Lagrange basis1d;

   // number of points in pptr
   int npts;
   // number of points per element along each dimension to test
   int nq1d;
   // total number of points to test
   int nq;
};

template <int Geom, int SDim, int max_team_x, int max_q1d>
struct PhysNodeFinder;

template <int SDim, int max_team_x, int max_q1d>
struct PhysNodeFinder<Geometry::SEGMENT, SDim, max_team_x, max_q1d>
   : public NodeFinderBase
{

   static int compute_nq(int nq1d) { return nq1d; }

   static int ndofs(int ndof1d) { return ndof1d; }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 1;
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      MFEM_FOREACH_THREAD(i, x, max_team_x)
      {
#ifdef MFEM_USE_DOUBLE
         dists[i] = HUGE_VAL;
#else
         dists[i] = HUGE_VALF;
#endif
      }
      MFEM_SYNC_THREAD;
      // team serial portion
      MFEM_FOREACH_THREAD(i, x, nq)
      {
         real_t phys_coord[SDim] = {0};
         for (int j0 = 0; j0 < basis1d.pN; ++j0)
         {
            real_t b = basis1d.eval(qptr[i], j0);
            for (int d = 0; d < SDim; ++d)
            {
               phys_coord[d] +=
                  mptr[j0 + (d + eptr[idx] * SDim) * basis1d.pN] * b;
            }
         }
         real_t dist = 0;
         // L-2 norm squared
         for (int d = 0; d < SDim; ++d)
         {
            real_t tmp = phys_coord[d] - pptr[idx + d * npts];
            dist += tmp * tmp;
         }
         if (dist < dists[MFEM_THREAD_ID(x)])
         {
            // closer guess in physical space
            dists[MFEM_THREAD_ID(x)] = dist;
            ref_buf[MFEM_THREAD_ID(x)] = qptr[i];
         }
      }
      // now do tree reduce
      for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
      {
         MFEM_SYNC_THREAD;
         if (MFEM_THREAD_ID(x) < i)
         {
            if (dists[MFEM_THREAD_ID(x) + i] < dists[MFEM_THREAD_ID(x)])
            {
               dists[MFEM_THREAD_ID(x)] = dists[MFEM_THREAD_ID(x) + i];
               ref_buf[MFEM_THREAD_ID(x)] = ref_buf[MFEM_THREAD_ID(x) + i];
            }
         }
      }
      // write results out
      // not needed in 1D
      // MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(x) == 0)
      {
         xptr[idx] = ref_buf[0];
      }
   }
};

template <int SDim, int max_team_x, int max_q1d>
struct PhysNodeFinder<Geometry::SQUARE, SDim, max_team_x, max_q1d>
   : public NodeFinderBase
{

   static int compute_nq(int nq1d) { return nq1d * nq1d; }

   static int ndofs(int ndof1d)
   {
      return ndof1d * ndof1d;
   }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 2;
      constexpr int max_dof1d = 32;
      int n = (nq < max_team_x) ? nq : max_team_x;
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      MFEM_SHARED real_t basis[max_dof1d * max_q1d];
      MFEM_FOREACH_THREAD(i, x, max_team_x)
      {
#ifdef MFEM_USE_DOUBLE
         dists[i] = HUGE_VAL;
#else
         dists[i] = HUGE_VALF;
#endif
      }
      MFEM_FOREACH_THREAD(j0, x, nq1d)
      {
         for (int i0 = 0; i0 < basis1d.pN; ++i0)
         {
            basis[j0 + i0 * nq1d] = basis1d.eval(qptr[j0], i0);
         }
      }
      MFEM_SYNC_THREAD;
      // team serial portion
      MFEM_FOREACH_THREAD(j, x, nq)
      {
         real_t phys_coord[SDim] = {0};
         int idcs[Dim];
         idcs[0] = j % nq1d;
         idcs[1] = j / nq1d;
         for (int i1 = 0; i1 < basis1d.pN; ++i1)
         {
            for (int i0 = 0; i0 < basis1d.pN; ++i0)
            {
               real_t b = basis[idcs[0] + i0 * nq1d] * basis[idcs[1] + i1 * nq1d];
               for (int d = 0; d < SDim; ++d)
               {
                  phys_coord[d] +=
                     mptr[i0 + (i1 + (d + eptr[idx] * SDim) * basis1d.pN) *
                             basis1d.pN] *
                     b;
               }
            }
         }
         real_t dist = 0;
         // L-2 norm squared
         for (int d = 0; d < SDim; ++d)
         {
            real_t tmp = pptr[idx + d * npts] - phys_coord[d];
            dist += tmp * tmp;
         }
         if (dist < dists[MFEM_THREAD_ID(x)])
         {
            // closer guess in physical space
            dists[MFEM_THREAD_ID(x)] = dist;
            for (int d = 0; d < Dim; ++d)
            {
               ref_buf[MFEM_THREAD_ID(x) + d * n] = qptr[idcs[d]];
            }
         }
      }
      // now do tree reduce
      for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
      {
         MFEM_SYNC_THREAD;
         if (MFEM_THREAD_ID(x) < i)
         {
            if (dists[MFEM_THREAD_ID(x) + i] < dists[MFEM_THREAD_ID(x)])
            {
               dists[MFEM_THREAD_ID(x)] = dists[MFEM_THREAD_ID(x) + i];
               for (int d = 0; d < Dim; ++d)
               {
                  ref_buf[MFEM_THREAD_ID(x) + d * n] =
                     ref_buf[MFEM_THREAD_ID(x) + i + d * n];
               }
            }
         }
      }
      // write results out
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d, x, Dim) { xptr[idx + d * npts] = ref_buf[d * n]; }
   }
};

template <int SDim, int max_team_x, int max_q1d>
struct PhysNodeFinder<Geometry::CUBE, SDim, max_team_x, max_q1d>
   : public NodeFinderBase
{

   static int compute_nq(int nq1d) { return nq1d * nq1d * nq1d; }

   static int ndofs(int ndof1d)
   {
      return ndof1d * ndof1d * ndof1d;
   }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 3;
      constexpr int max_dof1d = 32;
      int n = (nq < max_team_x) ? nq : max_team_x;
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      // contiguous in quad
      MFEM_SHARED real_t basis[max_dof1d * max_q1d];
      MFEM_FOREACH_THREAD(i, x, max_team_x)
      {
#ifdef MFEM_USE_DOUBLE
         dists[i] = HUGE_VAL;
#else
         dists[i] = HUGE_VALF;
#endif
      }
      MFEM_FOREACH_THREAD(j0, x, nq1d)
      {
         for (int i0 = 0; i0 < basis1d.pN; ++i0)
         {
            basis[j0 + i0 * nq1d] = basis1d.eval(qptr[j0], i0);
         }
      }
      MFEM_SYNC_THREAD;
      // team serial portion
      MFEM_FOREACH_THREAD(j, x, nq)
      {
         real_t phys_coord[SDim] = {0};
         int idcs[Dim];
         idcs[0] = j % nq1d;
         idcs[1] = j / nq1d;
         idcs[2] = idcs[1] / nq1d;
         idcs[1] = idcs[1] % nq1d;
         for (int i2 = 0; i2 < basis1d.pN; ++i2)
         {
            for (int i1 = 0; i1 < basis1d.pN; ++i1)
            {
               for (int i0 = 0; i0 < basis1d.pN; ++i0)
               {
                  real_t b = basis[idcs[0] + i0 * nq1d] * basis[idcs[1] + i1 * nq1d] *
                             basis[idcs[2] + i2 * nq1d];
                  for (int d = 0; d < SDim; ++d)
                  {
                     phys_coord[d] +=
                        mptr[i0 +
                                (i1 + (i2 + (d + eptr[idx] * SDim) * basis1d.pN) *
                                 basis1d.pN) *
                                basis1d.pN] *
                        b;
                  }
               }
            }
         }
         real_t dist = 0;
         // L-2 norm squared
         for (int d = 0; d < SDim; ++d)
         {
            real_t tmp = pptr[idx + d * npts] - phys_coord[d];
            dist += tmp * tmp;
         }
         if (dist < dists[MFEM_THREAD_ID(x)])
         {
            // closer guess in physical space
            dists[MFEM_THREAD_ID(x)] = dist;
            for (int d = 0; d < Dim; ++d)
            {
               ref_buf[MFEM_THREAD_ID(x) + d * n] = qptr[idcs[d]];
            }
         }
      }
      // now do tree reduce
      for (int i = (MFEM_THREAD_SIZE(x) >> 1); i > 0; i >>= 1)
      {
         MFEM_SYNC_THREAD;
         if (MFEM_THREAD_ID(x) < i)
         {
            if (dists[MFEM_THREAD_ID(x) + i] < dists[MFEM_THREAD_ID(x)])
            {
               dists[MFEM_THREAD_ID(x)] = dists[MFEM_THREAD_ID(x) + i];
               for (int d = 0; d < Dim; ++d)
               {
                  ref_buf[MFEM_THREAD_ID(x) + d * n] =
                     ref_buf[MFEM_THREAD_ID(x) + i + d * n];
               }
            }
         }
      }
      // write results out
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d, x, Dim) { xptr[idx + d * npts] = ref_buf[d * n]; }
   }
};

template <int Geom, int SDim, bool use_device>
static void ClosestPhysNodeImpl(int npts, int nelems, int ndof1d, int nq1d,
                                const real_t *mptr, const real_t *pptr,
                                const int *eptr, const real_t *nptr,
                                const real_t *qptr, real_t *xptr)
{
   constexpr int max_team_x = 64;
   constexpr int max_q1d = 128;
   PhysNodeFinder<Geom, SDim, max_team_x, max_q1d> func;
   MFEM_VERIFY(ndof1d <= 32, "maximum of 32 dofs per dim is allowed");
   MFEM_VERIFY(nq1d <= max_q1d, "maximum of 128 test points per dim is allowed");
   func.basis1d.z = nptr;
   func.basis1d.pN = ndof1d;
   func.mptr = mptr;
   func.pptr = pptr;
   func.eptr = eptr;
   func.qptr = qptr;
   func.xptr = xptr;
   func.npts = npts;
   func.nq1d = nq1d;
   func.nq = func.compute_nq(nq1d);
   if (use_device)
   {
      // team_x must be a power of 2
      int team_x = max_team_x;
      while (true)
      {
         if (team_x <= func.nq)
         {
            break;
         }
         team_x >>= 1;
      }
      team_x = std::min<int>(max_team_x, 2 * team_x);
      forall_2D(npts, team_x, 1, func);
   }
   else
   {
      forall_switch(false, npts, func);
   }
}

template <int Geom, int SDim, bool use_device>
static void ClosestPhysDofImpl(int npts, int nelems, int ndof1d,
                               const real_t *mptr, const real_t *pptr,
                               const int *eptr, const real_t *nptr,
                               real_t *xptr)
{
   constexpr int max_team_x = 64;
   PhysDofFinder<Geom, SDim, max_team_x> func;
   func.basis1d.z = nptr;
   func.basis1d.pN = ndof1d;
   func.mptr = mptr;
   func.pptr = pptr;
   func.eptr = eptr;
   func.xptr = xptr;
   func.npts = npts;
   if (use_device)
   {
      int team_x = max_team_x;
      int ndof = func.ndofs(ndof1d);
      while (true)
      {
         if (team_x <= ndof)
         {
            break;
         }
         team_x >>= 1;
      }
      team_x = std::min<int>(max_team_x, 2 * team_x);
      forall_2D(npts, team_x, 1, func);
   }
   else
   {
      forall_switch(false, npts, func);
   }
}

template <int Geom, int SDim, bool use_device>
static void ClosestRefDofImpl(int npts, int nelems, int ndof1d,
                              const real_t *mptr, const real_t *pptr,
                              const int *eptr, const real_t *nptr,
                              real_t *xptr)
{
   // TODO
   MFEM_ABORT("ClostestRefDofImpl not implemented yet");
}

template <int Geom, int SDim, bool use_device>
static void ClosestRefNodeImpl(int npts, int nelems, int ndof1d, int nq1d,
                               const real_t *mptr, const real_t *pptr,
                               const int *eptr, const real_t *nptr,
                               const real_t *qptr, real_t *xptr)
{
   // TODO
   MFEM_ABORT("ClostestRefNodeImpl not implemented yet");
}

template <int Geom, int SDim, InverseElementTransformation::SolverType SType,
          bool use_device>
static void NewtonSolveImpl(real_t ref_tol, real_t phys_rtol, int max_iter,
                            int npts, int nelems, int ndof1d,
                            const real_t *mptr, const real_t *pptr,
                            const int *eptr, const real_t *nptr, int *tptr,
                            int *iter_ptr, real_t *xptr)
{
   constexpr int max_team_x = use_device ? 64 : 1;
   InvTNewtonSolver<Geom, SDim, SType, max_team_x> func;
   MFEM_VERIFY(ndof1d <= func.max_dof1d(),
               "exceeded max_dof1d limit (32 for 2D/3D)");
   func.ref_tol = ref_tol;
   func.phys_rtol = phys_rtol;
   func.max_iter = max_iter;
   func.basis1d.z = nptr;
   func.basis1d.pN = ndof1d;
   func.mptr = mptr;
   func.pptr = pptr;
   func.eptr = eptr;
   func.xptr = xptr;
   func.iter_ptr = iter_ptr;
   func.tptr = tptr;
   func.npts = npts;
   if (use_device)
   {
      int team_x = max_team_x;
      int ndof = func.ndofs(ndof1d);
      while (true)
      {
         if (team_x <= ndof)
         {
            break;
         }
         team_x >>= 1;
      }
      team_x = std::min<int>(max_team_x, 2 * team_x);
      forall_2D(npts, team_x, 1, func);
   }
   else
   {
      forall_switch(false, npts, func);
   }
}

template <int Geom, int SDim,
          InverseElementTransformation::SolverType SolverType, int max_team_x>
struct InvTNewtonEdgeScanner
{
   InvTNewtonSolver<Geom, SDim, SolverType, max_team_x> solver;
   // 1D ref space initial guesses
   const real_t *qptr;
   // num 1D points in qptr
   int nq1d;

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      // can only be outside if all test points report outside
      int res = InverseElementTransformation::Outside;
      for (int i = 0; i < nq1d; ++i)
      {
         for (int d = 0; d < eltrans::GeometryUtils<Geom>::Dimension();
              ++d)
         {
            if (MFEM_THREAD_ID(x) == 0)
            {
               for (int d2 = 0;
                    d2 < eltrans::GeometryUtils<Geom>::Dimension();
                    ++d2)
               {
                  solver.xptr[idx + d2 * solver.npts] = 0;
               }
               solver.xptr[idx + d * solver.npts] = qptr[i];
            }
            MFEM_SYNC_THREAD;
            int res_tmp = solver(idx);
            switch (res_tmp)
            {
               case InverseElementTransformation::Inside:
                  return;
               case InverseElementTransformation::Outside:
                  break;
               case InverseElementTransformation::Unknown:
                  res = InverseElementTransformation::Unknown;
                  break;
            }
            if (qptr[i] == 0)
            {
               // don't repeat test the origin
               break;
            }
         }
      }
      if (MFEM_THREAD_ID(x) == 0)
      {
         solver.tptr[idx] = res;
      }
   }
};

template <int Geom, int SDim, InverseElementTransformation::SolverType SType,
          bool use_device>
static void NewtonEdgeScanImpl(real_t ref_tol, real_t phys_rtol, int max_iter,
                               int npts, int nelems, int ndof1d,
                               const real_t *mptr, const real_t *pptr,
                               const int *eptr, const real_t *nptr,
                               const real_t *qptr, int nq1d, int *tptr,
                               int *iter_ptr, real_t *xptr)
{
   constexpr int max_team_x = use_device ? 64 : 1;
   InvTNewtonEdgeScanner<Geom, SDim, SType, max_team_x> func;
   MFEM_VERIFY(ndof1d <= func.solver.max_dof1d(),
               "exceeded max_dof1d limit (32 for 2D/3D)");
   func.solver.ref_tol = ref_tol;
   func.solver.phys_rtol = phys_rtol;
   func.solver.max_iter = max_iter;
   func.solver.basis1d.z = nptr;
   func.solver.basis1d.pN = ndof1d;
   func.solver.mptr = mptr;
   func.solver.pptr = pptr;
   func.solver.eptr = eptr;
   func.solver.xptr = xptr;
   func.solver.iter_ptr = iter_ptr;
   func.solver.tptr = tptr;
   func.solver.npts = npts;
   func.nq1d = nq1d;
   func.qptr = qptr;
   if (use_device)
   {
      int team_x = max_team_x;
      int ndof = func.solver.ndofs(ndof1d);
      while (true)
      {
         if (team_x <= ndof)
         {
            break;
         }
         team_x >>= 1;
      }
      team_x = std::min<int>(max_team_x, 2 * team_x);
      forall_2D(npts, team_x, 1, func);
   }
   else
   {
      forall_switch(false, npts, func);
   }
}

} // namespace internal

template <int Geom, int SDim, bool use_device>
BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Kernel()
{
   return internal::ClosestPhysNodeImpl<Geom, SDim, use_device>;
}

template <int Geom, int SDim, bool use_device>
BatchInverseElementTransformation::ClosestPhysDofKernelType
BatchInverseElementTransformation::FindClosestPhysDof::Kernel()
{
   return internal::ClosestPhysDofImpl<Geom, SDim, use_device>;
}

template <int Geom, int SDim, bool use_device>
BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Kernel()
{
   return internal::ClosestRefNodeImpl<Geom, SDim, use_device>;
}

template <int Geom, int SDim, bool use_device>
BatchInverseElementTransformation::ClosestRefDofKernelType
BatchInverseElementTransformation::FindClosestRefDof::Kernel()
{
   return internal::ClosestRefDofImpl<Geom, SDim, use_device>;
}

template <int Geom, int SDim, InverseElementTransformation::SolverType SType,
          bool use_device>
BatchInverseElementTransformation::NewtonKernelType
BatchInverseElementTransformation::NewtonSolve::Kernel()
{
   return internal::NewtonSolveImpl<Geom, SDim, SType, use_device>;
}

template <int Geom, int SDim, InverseElementTransformation::SolverType SType,
          bool use_device>
BatchInverseElementTransformation::NewtonEdgeScanKernelType
BatchInverseElementTransformation::NewtonEdgeScan::Kernel()
{
   return internal::NewtonEdgeScanImpl<Geom, SDim, SType, use_device>;
}

BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Fallback(int, int,
                                                                  bool)
{
   MFEM_ABORT("Invalid Geom/SDim combination");
}

BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Fallback(int, int,
                                                                 bool)
{
   MFEM_ABORT("Invalid Geom/SDim combination");
}

BatchInverseElementTransformation::NewtonKernelType
BatchInverseElementTransformation::NewtonSolve::Fallback(
   int, int, InverseElementTransformation::SolverType, bool)
{
   MFEM_ABORT("Invalid Geom/SDim/SolverType combination");
}

BatchInverseElementTransformation::NewtonEdgeScanKernelType
BatchInverseElementTransformation::NewtonEdgeScan::Fallback(
   int, int, InverseElementTransformation::SolverType, bool)
{
   MFEM_ABORT("Invalid Geom/SDim/SolverType combination");
}

BatchInverseElementTransformation::ClosestPhysDofKernelType
BatchInverseElementTransformation::FindClosestPhysDof::Fallback(int, int, bool)
{
   MFEM_ABORT("Invalid Geom/SDim combination");
}

BatchInverseElementTransformation::ClosestRefDofKernelType
BatchInverseElementTransformation::FindClosestRefDof::Fallback(int, int, bool)
{
   MFEM_ABORT("Invalid Geom/SDim combination");
}

BatchInverseElementTransformation::Kernels::Kernels()
{
   using BatchInvTr = BatchInverseElementTransformation;

   constexpr auto SEGMENT = Geometry::SEGMENT;
   constexpr auto SQUARE = Geometry::SQUARE;
   constexpr auto CUBE = Geometry::CUBE;
   constexpr auto Newton = InverseElementTransformation::Newton;
   constexpr auto NewtonElementProject =
      InverseElementTransformation::NewtonElementProject;

   BatchInvTr::AddFindClosestSpecialization<SEGMENT, 1>();
   BatchInvTr::AddFindClosestSpecialization<SEGMENT, 2>();
   BatchInvTr::AddFindClosestSpecialization<SEGMENT, 3>();

   BatchInvTr::AddFindClosestSpecialization<SQUARE, 2>();
   BatchInvTr::AddFindClosestSpecialization<SQUARE, 3>();

   BatchInvTr::AddFindClosestSpecialization<CUBE, 3>();

   // NewtonSolve
   BatchInvTr::AddNewtonSolveSpecialization<SEGMENT, 1, Newton>();
   BatchInvTr::AddNewtonSolveSpecialization<SEGMENT, 2, Newton>();
   BatchInvTr::AddNewtonSolveSpecialization<SEGMENT, 3, Newton>();

   BatchInvTr::AddNewtonSolveSpecialization<SEGMENT, 1, NewtonElementProject>();
   BatchInvTr::AddNewtonSolveSpecialization<SEGMENT, 2, NewtonElementProject>();
   BatchInvTr::AddNewtonSolveSpecialization<SEGMENT, 3, NewtonElementProject>();

   BatchInvTr::AddNewtonSolveSpecialization<SQUARE, 2, Newton>();
   BatchInvTr::AddNewtonSolveSpecialization<SQUARE, 3, Newton>();
   BatchInvTr::AddNewtonSolveSpecialization<SQUARE, 2, NewtonElementProject>();
   BatchInvTr::AddNewtonSolveSpecialization<SQUARE, 3, NewtonElementProject>();

   BatchInvTr::AddNewtonSolveSpecialization<CUBE, 3, Newton>();
   BatchInvTr::AddNewtonSolveSpecialization<CUBE, 3, NewtonElementProject>();
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
