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

BatchInverseElementTransformation::~BatchInverseElementTransformation() {}

void BatchInverseElementTransformation::Setup(Mesh &m, MemoryType d_mt)
{
   static Kernels kernels;
   MemoryType my_d_mt =
      (d_mt != MemoryType::DEFAULT) ? d_mt : Device::GetDeviceMemoryType();

   mesh = &m;
   MFEM_VERIFY(mesh->GetNodes(), "the provided mesh must have valid nodes.");
   const FiniteElementSpace *fespace = mesh->GetNodalFESpace();
   const int max_order = fespace->GetMaxElementOrder();
   const int ndof1d = max_order + 1;
   int ND = ndof1d;
   const int dim = mesh->Dimension();
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "Mixed meshes are not swupported.");
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
      points1d = poly1d.GetPointsArray(max_order, BasisType::GaussLobatto);
      points1d->HostRead();
      // either mixed order, or not a tensor basis
      node_pos.HostWrite();
      real_t tmp[3];
      int sdim = mesh->SpaceDimension();
      MFEM_VERIFY(sdim == vdim,
                  "mesh.SpaceDimension and fespace.GetVDim mismatch");
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
            mesh->GetNodes()->GetVectorValue(e, ip, pos);
            for (int d = 0; d < sdim; ++d)
            {
               node_pos[i + (d * NE + e) * ND] = pos[d];
            }
         }
      }
   }
   else
   {
      const Operator *elem_restr =
         fespace->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      elem_restr->Mult(*mesh->GetNodes(), node_pos);
      points1d = poly1d.GetPointsArray(max_order, tfe->GetBasisType());
   }
}


void BatchInverseElementTransformation::Transform(const Vector &pts,
                                                  const Array<int> &elems,
                                                  Array<int> &types,
                                                  Vector &refs, bool use_dev)
{
   if (!Device::Allows(Backend::DEVICE_MASK))
   {
      // no devices available
      use_dev = false;
   }
   const FiniteElementSpace *fespace = mesh->GetNodalFESpace();
   const FiniteElement *fe = fespace->GetTypicalFE();
   const int dim = fe->GetDim();
   const int vdim = fespace->GetVDim();
   const int NE = fespace->GetNE();
   // const int ND = fe->GetDof();
   const int order = fe->GetOrder();
   int npts = elems.Size();
   auto geom = fe->GetGeomType();

   types.SetSize(npts);
   refs.SetSize(npts * dim);

   auto pptr = pts.Read(use_dev);
   auto eptr = elems.Read(use_dev);
   auto mptr = node_pos.Read(use_dev);
   auto tptr = types.Write(use_dev);
   auto xptr = refs.ReadWrite(use_dev);
   auto nptr = points1d->Read(use_dev);
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
               forall_switch(use_dev, npts,
               [=] MFEM_HOST_DEVICE(int i) { xptr[i] = cx; });
               break;
            case 2:
               forall_switch(use_dev, npts, [=] MFEM_HOST_DEVICE(int i)
               {
                  xptr[i] = cx;
                  xptr[i + npts] = cy;
               });
               break;
            case 3:
               forall_switch(use_dev, npts, [=] MFEM_HOST_DEVICE(int i)
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
         int nq1d = std::max(order + rel_qpts_order, 0) + 1;
         int btype = BasisType::GetNodalBasis(guess_points_type);
         auto qpoints = poly1d.GetPointsArray(nq1d - 1, btype);
         auto qptr = qpoints->Read(use_dev);
         if (init_guess_type == InverseElementTransformation::ClosestPhysNode)
         {
            FindClosestPhysPoint::Run(geom, vdim, use_dev, npts, NE, ndof1d, nq1d,
                                      mptr, pptr, eptr, nptr, qptr, xptr);
         }
         else
         {
            FindClosestRefPoint::Run(geom, vdim, use_dev, npts, NE, ndof1d, nq1d,
                                     mptr, pptr, eptr, nptr, qptr, xptr);
         }
      } break;
      case InverseElementTransformation::GivenPoint:
         // nothing to do here
         break;
      case InverseElementTransformation::EdgeScan:
      {
         // TODO
      }
      return;
   }
   // general case: for each point, use guess inside refs
   NewtonSolve::Run(geom, vdim, solver_type, use_dev, ref_tol, phys_rtol,
                    max_iter, npts, NE, ndof1d, mptr, pptr, eptr, nptr, tptr,
                    xptr);
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
   // result ref coords
   real_t *xptr;
   eltrans::Lagrange basis1d;

   int max_iter;
   // ndof * nelems
   int stride_sdim;
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
      real_t den = jac[0] * jac[0] +
                   jac[1 * MFEM_THREAD_SIZE(x)] * jac[1 * MFEM_THREAD_SIZE(x)];
      dx[0] = (jac[0] * rhs[0] +
               jac[1 * MFEM_THREAD_SIZE(x)] * rhs[1 * MFEM_THREAD_SIZE(x)]) /
              den;
   }
};

template <> struct InvTLinSolve<1, 3>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      real_t den = jac[0] * jac[0] +
                   jac[1 * MFEM_THREAD_SIZE(x)] * jac[1 * MFEM_THREAD_SIZE(x)] +
                   jac[2 * MFEM_THREAD_SIZE(x)] * jac[2 * MFEM_THREAD_SIZE(x)];
      dx[0] = (jac[0] * rhs[0] +
               jac[1 * MFEM_THREAD_SIZE(x)] * rhs[1 * MFEM_THREAD_SIZE(x)] +
               jac[2 * MFEM_THREAD_SIZE(x)] * rhs[2 * MFEM_THREAD_SIZE(x)]) /
              den;
   }
};

template <> struct InvTLinSolve<2, 2>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      real_t den = 1 / (jac[(0 + 0 * 2) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 2) * MFEM_THREAD_SIZE(x)] -
                        jac[(0 + 1 * 2) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 0 * 2) * MFEM_THREAD_SIZE(x)]);
      dx[0] = (jac[(1 + 1 * 2) * MFEM_THREAD_SIZE(x)] *
               rhs[0 * MFEM_THREAD_SIZE(x)] -
               jac[(0 + 1 * 2) * MFEM_THREAD_SIZE(x)] *
               rhs[1 * MFEM_THREAD_SIZE(x)]) *
              den;
      dx[1] = (jac[(0 + 0 * 2) * MFEM_THREAD_SIZE(x)] *
               rhs[1 * MFEM_THREAD_SIZE(x)] -
               jac[(1 + 0 * 2) * MFEM_THREAD_SIZE(x)] *
               rhs[0 * MFEM_THREAD_SIZE(x)]) *
              den;
   }
};

template <> struct InvTLinSolve<2, 3>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      // a00**2*a11**2 + a00**2*a21**2 - 2*a00*a01*a10*a11 - 2*a00*a01*a20*a21 +
      // a01**2*a10**2 + a01**2*a20**2 + a10**2*a21**2 - 2*a10*a11*a20*a21 +
      // a11**2*a20**2
      real_t den = 1 / (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                        jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] -
                        2 * jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] -
                        2 * jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                        jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] -
                        2 * jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]);
      //   x0*(a00*(a01**2 + a11**2 + a21**2) - a01*(a00*a01 + a10*a11 + a20*a21))
      // + x1*(a10*(a01**2 + a11**2 + a21**2) - a11*(a00*a01 + a10*a11 + a20*a21))
      // + x2*(a20*(a01**2 + a11**2 + a21**2) - a21*(a00*a01 + a10*a11 + a20*a21))
      dx[0] = (rhs[0 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)]) -
                jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)])) +
               rhs[1 * MFEM_THREAD_SIZE(x)] *
               (jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)]) -
                jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)])) +
               rhs[2 * MFEM_THREAD_SIZE(x)] *
               (jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3)]) -
                jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)]))) *
              den;
      //  x0*(a01*(a00**2 + a10**2 + a20**2)-a00*(a00*a01 + a10*a11 + a20*a21))
      // +x1*(a11*(a00**2 + a10**2 + a20**2)-a10*(a00*a01 + a10*a11 + a20*a21))
      // +x2*(a21*(a00**2 + a10**2 + a20**2)-a20*(a00*a01 + a10*a11 + a20*a21))
      dx[1] = (rhs[0 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]) -
                jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)])) +
               rhs[1 * MFEM_THREAD_SIZE(x)] *
               (jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]) -
                jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)])) +
               rhs[2 * MFEM_THREAD_SIZE(x)] *
               (jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]) -
                jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] +
                 jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                 jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)]))) *
              den;
   }
};

template <> struct InvTLinSolve<3, 3>
{
   static void MFEM_HOST_DEVICE solve(const real_t *jac, const real_t *rhs,
                                      real_t *dx)
   {
      real_t den = 1 / (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)] -
                        jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)] -
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)] +
                        jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)] +
                        jac[(0 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] -
                        jac[(0 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                        jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]);
      dx[0] = (rhs[0 * MFEM_THREAD_SIZE(x)] *
               (jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(1 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)]) -
               rhs[1 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(0 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)]) +
               rhs[2 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(1 + 2 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(0 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)])) *
              den;
      dx[1] = (rhs[0 * MFEM_THREAD_SIZE(x)] *
               (jac[(1 + 2 * 3) * MFEM_THREAD_SIZE(x)] * jac[2 + 0 * 3] -
                jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)]) +
               rhs[1 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 2 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(0 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]) -
               rhs[2 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(1 + 2 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(0 + 2 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)])) *
              den;
      dx[2] = (rhs[0 * MFEM_THREAD_SIZE(x)] *
               (jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]) -
               rhs[1 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 1 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(2 + 0 * 3) * MFEM_THREAD_SIZE(x)]) +
               rhs[2 * MFEM_THREAD_SIZE(x)] *
               (jac[(0 + 0 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(1 + 1 * 3) * MFEM_THREAD_SIZE(x)] -
                jac[(0 + 1 * 3) * MFEM_THREAD_SIZE(x)] *
                jac[(1 + 0 * 3) * MFEM_THREAD_SIZE(x)])) *
              den;
   }
};

template <int Geom, InverseElementTransformation::SolverType SolverType>
struct ProjectType;

template <>
struct ProjectType<Geometry::SEGMENT, InverseElementTransformation::Newton>
{
   static MFEM_HOST_DEVICE bool project(real_t& x, real_t& dx)
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
   template <class... Ts> static MFEM_HOST_DEVICE bool project(Ts&&... args)
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
   static int compute_stride_sdim(int ndof1d, int nelems)
   {
      return ndof1d * nelems;
   }

   static int ndofs(int ndof1d) { return ndof1d; }

   // theoretically unbounded
   static constexpr MFEM_HOST_DEVICE int max_dof1d() { return 0x1000; }

   void MFEM_HOST_DEVICE operator()(int idx) const
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
      real_t phys_tol = 0;
      if (MFEM_THREAD_ID(x) == 0)
      {
         term_flag[0] = false;
         for (int d = 0; d < Dim; ++d)
         {
            ref_coord[d] = xptr[idx + d * npts];
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
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(j0, x, basis1d.pN)
         {
            real_t b0, db0;
            basis1d.eval_d1(b0, db0, ref_coord[0], j0);
            for (int d = 0; d < SDim; ++d)
            {
               phys_coord[MFEM_THREAD_ID(x) + d * MFEM_THREAD_SIZE(x)] +=
                  mptr[j0 + eptr[idx] * basis1d.pN + d * stride_sdim] * b0;
               jac[MFEM_THREAD_ID(x) + (d + 0 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[j0 + eptr[idx] * basis1d.pN + d * stride_sdim] * db0;
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
               tptr[idx] =
                  eltrans::GeometryUtils<Geometry::SEGMENT>::inside(ref_coord[0])
                  ? InverseElementTransformation::Inside
                  : InverseElementTransformation::Outside;
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               term_flag[0] = true;
            }
            else if (iter >= max_iter)
            {
               // terminate on max iterations
               tptr[idx] = InverseElementTransformation::Unknown;
               // might as well save where we failed at
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               term_flag[0] = true;
            }
            else
            {
               // compute dx = (pseudo)-inverse jac * [pt - F(x)]
               real_t dx[Dim];
               InvTLinSolve<Dim, SDim>::solve(jac, phys_coord, dx);

               bool hit_bdr = ProjectType<Geometry::SEGMENT, SType>::project(
                                 ref_coord[0], dx[0]);

               // check for ref coord convergence or stagnation on boundary
               if (fabs(dx[0]) <= ref_tol)
               {
                  tptr[idx] = hit_bdr ? InverseElementTransformation::Outside
                              : InverseElementTransformation::Inside;
                  for (int d = 0; d < Dim; ++d)
                  {
                     xptr[idx + d * npts] = ref_coord[d];
                  }
                  term_flag[0] = true;
               }
            }
         }

         MFEM_SYNC_THREAD;
         if (term_flag[0])
         {
            return;
         }
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
   static int compute_stride_sdim(int ndof1d, int nelems)
   {
      return ndof1d * ndof1d * nelems;
   }

   static constexpr MFEM_HOST_DEVICE int max_dof1d() { return 32; }

   void MFEM_HOST_DEVICE operator()(int idx) const
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
      real_t phys_tol = 0;
      if (MFEM_THREAD_ID(x) == 0)
      {
         term_flag[0] = false;
         for (int d = 0; d < Dim; ++d)
         {
            ref_coord[d] = xptr[idx + d * npts];
         }
         for (int d = 0; d < SDim; ++d)
         {
            phys_tol += pptr[idx + d * npts] * pptr[idx + d * npts];
         }
         phys_tol =
            fmax(phys_rtol * phys_rtol, phys_tol * phys_rtol * phys_rtol);
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
                               (idcs[1] + eptr[idx] * basis1d.pN) * basis1d.pN +
                               d * stride_sdim] *
                  basis0[idcs[0]] * basis1[idcs[1]];
               jac[MFEM_THREAD_ID(x) + (d + 0 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + eptr[idx] * basis1d.pN) * basis1d.pN +
                               d * stride_sdim] *
                  dbasis0[idcs[0]] * basis1[idcs[1]];
               jac[MFEM_THREAD_ID(x) + (d + 1 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + eptr[idx] * basis1d.pN) * basis1d.pN +
                               d * stride_sdim] *
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
               tptr[idx] = eltrans::GeometryUtils<Geometry::SQUARE>::inside(
                              ref_coord[0], ref_coord[1])
                           ? InverseElementTransformation::Inside
                           : InverseElementTransformation::Outside;
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               term_flag[0] = true;
            }
            else if (iter >= max_iter)
            {
               // terminate on max iterations
               tptr[idx] = InverseElementTransformation::Unknown;
               // might as well save where we failed at
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               term_flag[0] = true;
            }
            else
            {
               // compute dx = (pseudo)-inverse jac * [pt - F(x)]
               real_t dx[Dim];
               InvTLinSolve<Dim, SDim>::solve(jac, phys_coord, dx);

               bool hit_bdr = ProjectType<Geometry::SQUARE, SType>::project(
                                 ref_coord[0], ref_coord[1], dx[0], dx[1]);

               // check for ref coord convergence or stagnation on boundary
               if (dx[0] * dx[0] + dx[1] * dx[1] <= ref_tol * ref_tol)
               {
                  tptr[idx] = hit_bdr ? InverseElementTransformation::Outside
                              : InverseElementTransformation::Inside;
                  for (int d = 0; d < Dim; ++d)
                  {
                     xptr[idx + d * npts] = ref_coord[d];
                  }
                  term_flag[0] = true;
               }
            }
         }

         MFEM_SYNC_THREAD;
         if (term_flag[0])
         {
            return;
         }
         ++iter;
      }
   }
};

template <int SDim, InverseElementTransformation::SolverType SType,
          int max_team_x>
struct InvTNewtonSolver<Geometry::CUBE, SDim, SType, max_team_x>
   : public InvTNewtonSolverBase
{
   static int compute_stride_sdim(int ndof1d, int nelems)
   {
      return ndof1d * ndof1d * ndof1d * nelems;
   }

   static int ndofs(int ndof1d) { return ndof1d * ndof1d * ndof1d; }

   static constexpr MFEM_HOST_DEVICE int max_dof1d() { return 32; }

   void MFEM_HOST_DEVICE operator()(int idx) const
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
      real_t phys_tol = 0;
      if (MFEM_THREAD_ID(x) == 0)
      {
         term_flag[0] = false;
         for (int d = 0; d < Dim; ++d)
         {
            ref_coord[d] = xptr[idx + d * npts];
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
                  mptr[idcs[0] +
                               (idcs[1] + (idcs[2] + eptr[idx] * basis1d.pN) * basis1d.pN) *
                               basis1d.pN +
                               d * stride_sdim] *
                  basis0[idcs[0]] * basis1[idcs[1]] * basis2[idcs[2]];
               jac[MFEM_THREAD_ID(x) + (d + 0 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + (idcs[2] + eptr[idx] * basis1d.pN) * basis1d.pN) *
                               basis1d.pN +
                               d * stride_sdim] *
                  dbasis0[idcs[0]] * basis1[idcs[1]] * basis2[idcs[2]];
               jac[MFEM_THREAD_ID(x) + (d + 1 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + (idcs[2] + eptr[idx] * basis1d.pN) * basis1d.pN) *
                               basis1d.pN +
                               d * stride_sdim] *
                  basis0[idcs[0]] * dbasis1[idcs[1]] * basis2[idcs[2]];
               jac[MFEM_THREAD_ID(x) + (d + 2 * SDim) * MFEM_THREAD_SIZE(x)] +=
                  mptr[idcs[0] +
                               (idcs[1] + (idcs[2] + eptr[idx] * basis1d.pN) * basis1d.pN) *
                               basis1d.pN +
                               d * stride_sdim] *
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
               tptr[idx] = eltrans::GeometryUtils<Geometry::CUBE>::inside(
                              ref_coord[0], ref_coord[1], ref_coord[2])
                           ? InverseElementTransformation::Inside
                           : InverseElementTransformation::Outside;
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               term_flag[0] = true;
            }
            else if (iter >= max_iter)
            {
               // terminate on max iterations
               tptr[idx] = InverseElementTransformation::Unknown;
               // might as well save where we failed at
               for (int d = 0; d < Dim; ++d)
               {
                  xptr[idx + d * npts] = ref_coord[d];
               }
               term_flag[0] = true;
            }
            else
            {
               // compute dx = (pseudo)-inverse jac * [pt - F(x)]
               real_t dx[Dim];
               InvTLinSolve<Dim, SDim>::solve(jac, phys_coord, dx);

               bool hit_bdr = ProjectType<Geometry::CUBE, SType>::project(
                                 ref_coord[0], ref_coord[1], ref_coord[2], dx[0], dx[1], dx[2]);

               // check for ref coord convergence or stagnation on boundary
               if (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] <=
                   ref_tol * ref_tol)
               {
                  tptr[idx] = hit_bdr ? InverseElementTransformation::Outside
                              : InverseElementTransformation::Inside;
                  for (int d = 0; d < Dim; ++d)
                  {
                     xptr[idx + d * npts] = ref_coord[d];
                  }
                  term_flag[0] = true;
               }
            }
         }

         MFEM_SYNC_THREAD;
         if (term_flag[0])
         {
            return;
         }
         ++iter;
      }
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

   // ndof * nelems
   int stride_sdim;
   // number of points in pptr
   int npts;
   // number of points per element along each dimension to test
   int nq1d;
   // total number of points to test
   int nq;
};

template <int Geom, int SDim, int max_team_x> struct PhysNodeFinder;

template <int SDim, int max_team_x>
struct PhysNodeFinder<Geometry::SEGMENT, SDim, max_team_x>
   : public NodeFinderBase
{

   static int compute_nq(int nq1d) { return nq1d; }

   static int compute_stride_sdim(int ndof1d, int nelems)
   {
      return ndof1d * nelems;
   }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 1;
      // int n = (nq < max_team_x) ? nq : max_team_x;
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
                  mptr[j0 + eptr[idx] * basis1d.pN + d * stride_sdim] * b;
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

template <int SDim, int max_team_x>
struct PhysNodeFinder<Geometry::SQUARE, SDim, max_team_x>
   : public NodeFinderBase
{

   static int compute_nq(int nq1d) { return nq1d * nq1d; }

   static int compute_stride_sdim(int ndof1d, int nelems)
   {
      return ndof1d * ndof1d * nelems;
   }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      constexpr int Dim = 2;
      constexpr int max_dof1d = 32;
      int n = (nq < max_team_x) ? nq : max_team_x;
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      MFEM_SHARED real_t basis_buf[max_dof1d * max_team_x];
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
         int idcs[Dim];
         idcs[0] = i % nq1d;
         idcs[1] = i / nq1d;
         for (int j1 = 0; j1 < basis1d.pN; ++j1)
         {
            basis_buf[MFEM_THREAD_ID(x) + j1 * MFEM_THREAD_SIZE(x)] =
               basis1d.eval(qptr[idcs[1]], j1);
         }
         for (int j0 = 0; j0 < basis1d.pN; ++j0)
         {
            real_t b0 = basis1d.eval(qptr[idcs[0]], j0);
            for (int j1 = 0; j1 < basis1d.pN; ++j1)
            {
               real_t b =
                  b0 * basis_buf[MFEM_THREAD_ID(x) + j1 * MFEM_THREAD_SIZE(x)];
               for (int d = 0; d < SDim; ++d)
               {
                  phys_coord[d] +=
                     mptr[j0 + (j1 + eptr[idx] * basis1d.pN) * basis1d.pN +
                             d * stride_sdim] *
                     b;
               }
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

template <int SDim, int max_team_x>
struct PhysNodeFinder<Geometry::CUBE, SDim, max_team_x>
   : public NodeFinderBase
{

   static int compute_nq(int nq1d) { return nq1d * nq1d * nq1d; }

   static int compute_stride_sdim(int ndof1d, int nelems)
   {
      return ndof1d * ndof1d * ndof1d * nelems;
   }

   void MFEM_HOST_DEVICE operator()(int idx) const
   {
      // TODO: for some reason this is extremely slow?
      constexpr int Dim = 3;
      constexpr int max_dof1d = 32;
      int n = (nq < max_team_x) ? nq : max_team_x;
      // L-2 norm squared
      MFEM_SHARED real_t dists[max_team_x];
      MFEM_SHARED real_t ref_buf[Dim * max_team_x];
      MFEM_SHARED real_t basis1_buf[max_dof1d * max_team_x];
      MFEM_SHARED real_t basis2_buf[max_dof1d * max_team_x];
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
         int idcs[Dim];
         idcs[0] = i % nq1d;
         idcs[1] = i / nq1d;
         idcs[2] = idcs[1] / nq1d;
         idcs[1] = idcs[1] % nq1d;
         for (int j1 = 0; j1 < basis1d.pN; ++j1)
         {
            basis1_buf[MFEM_THREAD_ID(x) + j1 * MFEM_THREAD_SIZE(x)] =
               basis1d.eval(qptr[idcs[1]], j1);
            basis2_buf[MFEM_THREAD_ID(x) + j1 * MFEM_THREAD_SIZE(x)] =
               basis1d.eval(qptr[idcs[2]], j1);
         }
         for (int j0 = 0; j0 < basis1d.pN; ++j0)
         {
            real_t b0 = basis1d.eval(qptr[idcs[0]], j0);
            for (int j1 = 0; j1 < basis1d.pN; ++j1)
            {
               real_t b1 =
                  b0 * basis1_buf[MFEM_THREAD_ID(x) + j1 * MFEM_THREAD_SIZE(x)];
               for (int j2 = 0; j2 < basis1d.pN; ++j2)
               {
                  real_t b =
                     b1 * basis2_buf[MFEM_THREAD_ID(x) + j2 * MFEM_THREAD_SIZE(x)];
                  for (int d = 0; d < SDim; ++d)
                  {
                     phys_coord[d] +=
                        mptr[j0 +
                                (j1 + (j2 + eptr[idx] * basis1d.pN) * basis1d.pN) *
                                basis1d.pN +
                                d * stride_sdim] *
                        b;
                  }
               }
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

template <int Geom, int SDim, bool use_dev>
static void ClosestPhysNodeImpl(int npts, int nelems, int ndof1d, int nq1d,
                                const real_t *mptr, const real_t *pptr,
                                const int *eptr, const real_t *nptr,
                                const real_t *qptr, real_t *xptr)
{
   constexpr int max_team_x = use_dev ? 64 : 1;
   PhysNodeFinder<Geom, SDim, max_team_x> func;
   // constexpr int max_dof1d = 32;
   MFEM_ASSERT(ndof1d <= 32, "maximum of 32 dofs per dim is allowed");
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
   func.stride_sdim = func.compute_stride_sdim(ndof1d, nelems);
   // TODO: any batching of npts?
   if (use_dev)
   {
      int team_x = std::min<int>(max_team_x, func.nq);
      forall_2D(npts, team_x, 1, func);
      MFEM_DEVICE_SYNC;
   }
   else
   {
      forall_switch(false, npts, func);
   }
}

template <int Geom, int SDim, bool use_dev>
static void ClosestRefNodeImpl(int npts, int nelems, int ndof1d, int nq1d,
                               const real_t *mptr, const real_t *pptr,
                               const int *eptr, const real_t *nptr,
                               const real_t *qptr, real_t *xptr)
{
   // TODO
   MFEM_ABORT("ClostestRefNodeImpl not implemented yet");
}

template <int Geom, int SDim, InverseElementTransformation::SolverType SType,
          bool use_dev>
static void
NewtonSolveImpl(real_t ref_tol, real_t phys_rtol, int max_iter, int npts,
                int nelems, int ndof1d, const real_t *mptr, const real_t *pptr,
                const int *eptr, const real_t *nptr, int *tptr, real_t *xptr)
{
   constexpr int max_team_x = use_dev ? 64 : 1;
   InvTNewtonSolver<Geom, SDim, SType, max_team_x> func;
   // constexpr int max_dof1d = 32;
   MFEM_ASSERT(ndof1d <= func.max_dof1d(),
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
   func.tptr = tptr;
   func.npts = npts;
   func.stride_sdim = func.compute_stride_sdim(ndof1d, nelems);
   if (use_dev)
   {
      int team_x = std::min<int>(max_team_x, func.ndofs(ndof1d));
      forall_2D(npts, team_x, 1, func);
   }
   else
   {
      forall_switch(false, npts, func);
   }
}

} // namespace internal

template <int Geom, int SDim, bool use_dev>
BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Kernel()
{
   return internal::ClosestPhysNodeImpl<Geom, SDim, use_dev>;
}

BatchInverseElementTransformation::ClosestPhysPointKernelType
BatchInverseElementTransformation::FindClosestPhysPoint::Fallback(int, int,
                                                                  bool)
{
   MFEM_ABORT("Invalid Geom/SDim combination");
}

template <int Geom, int SDim, bool use_dev>
BatchInverseElementTransformation::ClosestRefPointKernelType
BatchInverseElementTransformation::FindClosestRefPoint::Kernel()
{
   return internal::ClosestRefNodeImpl<Geom, SDim, use_dev>;
}

template <int Geom, int SDim, InverseElementTransformation::SolverType SType,
          bool use_dev>
BatchInverseElementTransformation::NewtonKernelType
BatchInverseElementTransformation::NewtonSolve::Kernel()
{
   return internal::NewtonSolveImpl<Geom, SDim, SType, use_dev>;
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

BatchInverseElementTransformation::Kernels::Kernels()
{
   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SEGMENT, 1, true>();
   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SEGMENT, 2, true>();
   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SEGMENT, 3, true>();

   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SQUARE, 2, true>();
   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SQUARE, 3, true>();

   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::CUBE, 3, true>();

   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SEGMENT, 1, false>();
   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SEGMENT, 2, false>();
   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SEGMENT, 3, false>();

   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SQUARE, 2, false>();
   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::SQUARE, 3, false>();

   BatchInverseElementTransformation::AddFindClosestSpecialization<
   Geometry::CUBE, 3, false>();

   // NewtonSolve
#if 1
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 1, InverseElementTransformation::Newton, true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 1, InverseElementTransformation::Newton, false>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 2, InverseElementTransformation::Newton, true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 2, InverseElementTransformation::Newton, false>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 3, InverseElementTransformation::Newton, true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 3, InverseElementTransformation::Newton, false>();
#endif

   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 1, InverseElementTransformation::NewtonElementProject,
            true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 1, InverseElementTransformation::NewtonElementProject,
            false>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 2, InverseElementTransformation::NewtonElementProject,
            true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 2, InverseElementTransformation::NewtonElementProject,
            false>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 3, InverseElementTransformation::NewtonElementProject,
            true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SEGMENT, 3, InverseElementTransformation::NewtonElementProject,
            false>();

#if 1
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 2, InverseElementTransformation::Newton, true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 2, InverseElementTransformation::Newton, false>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 3, InverseElementTransformation::Newton, true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 3, InverseElementTransformation::Newton, false>();
#endif
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 2, InverseElementTransformation::NewtonElementProject,
            true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 2, InverseElementTransformation::NewtonElementProject,
            false>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 3, InverseElementTransformation::NewtonElementProject,
            true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::SQUARE, 3, InverseElementTransformation::NewtonElementProject,
            false>();

#if 1
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::CUBE, 3, InverseElementTransformation::Newton, true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::CUBE, 3, InverseElementTransformation::Newton, false>();
#endif
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::CUBE, 3, InverseElementTransformation::NewtonElementProject,
            true>();
   BatchInverseElementTransformation::AddNewtonSolveSpecialization<
   Geometry::CUBE, 3, InverseElementTransformation::NewtonElementProject,
            false>();
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
