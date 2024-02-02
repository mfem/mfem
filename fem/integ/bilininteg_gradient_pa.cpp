// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"

namespace mfem
{

/* Description of the *SetupND functions
   Inputs are as follows
   \b Q1D number of quadrature points in one dimension.
   \b NE number of elements.
   \b MAP_TYPE map type of test fe.
   \b w quadrature weights.
   \b j element Jacobians.
   \b detj element Jacobian determinants.
   \b c coefficient at quadrature points.

   The function is used to precompute data needed at quadrature points during
   the action. */

/* Description of the *ApplyND functions
   The template parameters are
   \b T_D1D number of degrees of freedom in one dimension,
   \b T_Q1D number of quadrature points in one dimension,
   and are necessary to allow for compiler optimizations inside the kernel.

   Inputs are as follows
   \b NE number of elements.
   \b B matrix of basis functions.
   \b G matrix of derivatives of the basis functions.
   \b Bt transpose of matrix of basis functions.
   \b Gt transpose matrix of derivatives of the basis functions.
   \b op data used during action of the element matrix in the tensor
   product application.

   \b x input vector of degrees of freedom on the element.
   \b y output vector of degrees of freedom on the element.

   The function computes the kernel for one dimension that is suitable for
   tensor product action to form ND operators.
   Most of the ND inputs are reshaped as NQ*(ND*ND)*NE data structure, i.e.
   to allow indexing such as op(qpt,i,j,el).

   The output data structure is dependent on the kernel and layout of the
   dimension ND and element number, but in general resembles the action of the
   element matrix in the tensor product application. */

/* Description of the Smem*ApplyND functions
   The shared memory (Smem) versions of the kernels differ from the regular
   versions in the following properties.

   \b mfem::forall is using only one level of parallelism.
   \b mfem::forall_ND uses an additional level of parallelism
   \b MFEM_FOREACH_THREAD

   These macros allow automatic mapping of manually defined blocks to
   underlying hardware threads. These threads can share memory by using
   the \b MFEM_SHARED keyword for local arrays. */

// PA Gradient Assemble 2D kernel
static void PAGradientSetup2D(const int Q1D,
                              const int NE,
                              const int MAP_TYPE,
                              const Array<double> &w,
                              const Vector &j,
                              const Vector &detj,
                              const Vector &c,
                              Vector &op)
{
   const bool by_val = (MAP_TYPE == FiniteElement::VALUE);
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto DETJ = Reshape(detj.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, 2, 2, NE);

   const bool const_c = c.Size() == 1;
   const auto C = const_c ? Reshape(c.Read(), 1,1) :
                  Reshape(c.Read(), NQ, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J12 = J(q,0,1,e);
         const double J21 = J(q,1,0,e);
         const double J22 = J(q,1,1,e);
         // Coefficient and weight
         const double Co = const_c ? C(0,0) : C(q,e);
         const double cw = W[q] * Co * (by_val ? 1.0 : 1.0/DETJ(q,e));
         // Store wq * Q * adj(J)
         y(q,0,0,e) = cw *  J22; // 1,1
         y(q,0,1,e) = cw * -J12; // 1,2
         y(q,1,0,e) = cw * -J21; // 2,1
         y(q,1,1,e) = cw *  J11; // 2,2
      }
   });
}

// PA Gradient Assemble 3D kernel
static void PAGradientSetup3D(const int Q1D,
                              const int NE,
                              const int MAP_TYPE,
                              const Array<double> &w,
                              const Vector &j,
                              const Vector &detj,
                              const Vector &c,
                              Vector &op)
{
   const bool by_val = (MAP_TYPE == FiniteElement::VALUE);
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto DETJ = Reshape(detj.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, 3, 3, NE);

   const bool const_c = c.Size() == 1;
   const auto C = const_c ? Reshape(c.Read(), 1,1) :
                  Reshape(c.Read(), NQ,NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // Coefficient and weight
         const double Co = const_c ? C(0,0) : C(q,e);
         const double cw = W[q] * Co * (by_val ? 1.0 : 1.0/DETJ(q,e));
         // Store wq * Q * adj(J)
         y(q,0,0,e) = cw * A11; // 1,1
         y(q,0,1,e) = cw * A12; // 1,2
         y(q,0,2,e) = cw * A13; // 1,3
         y(q,1,0,e) = cw * A21; // 2,1
         y(q,1,1,e) = cw * A22; // 2,2
         y(q,1,2,e) = cw * A23; // 2,3
         y(q,2,0,e) = cw * A31; // 3,1
         y(q,2,1,e) = cw * A32; // 3,2
         y(q,2,2,e) = cw * A33; // 3,3
      }
   });
}

static void PAGradientSetup(const int dim,
                            const int TR_D1D,
                            const int TE_D1D,
                            const int Q1D,
                            const int NE,
                            const int MAP_TYPE,
                            const Array<double> &W,
                            const Vector &J,
                            const Vector &DET_J,
                            const Vector &COEFF,
                            Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PAGradientSetup"); }
   if (dim == 2)
   {
      PAGradientSetup2D(Q1D, NE, MAP_TYPE, W, J, DET_J, COEFF, op);
   }
   if (dim == 3)
   {
      PAGradientSetup3D(Q1D, NE, MAP_TYPE, W, J, DET_J, COEFF, op);
   }
}

void GradientIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                    const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements ordered by nodes
   MFEM_ASSERT(trial_fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   // Assuming the same element type
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement &trial_fe = *trial_fes.GetFE(0); // H1
   const FiniteElement &test_fe = *test_fes.GetFE(0); // H1^d or L2^d
   ElementTransformation *trans = mesh->GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
                                                            *trans);
   const int dims = trial_fe.GetDim();
   const int dimsToStore = dims * dims;
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS |
                                    GeometricFactors::DETERMINANTS);
   trial_maps = &trial_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   trial_dofs1D = trial_maps->ndof;
   quad1D = trial_maps->nqpt;
   test_maps  = &test_fe.GetDofToQuad(*ir, DofToQuad::TENSOR);
   test_dofs1D = test_maps->ndof;
   MFEM_ASSERT(quad1D == test_maps->nqpt,
               "PA requires test and trial space to have same number of quadrature points!");
   pa_data.SetSize(nq * dimsToStore * ne, Device::GetMemoryType());

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   PAGradientSetup(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                   test_fe.GetMapType(), ir->GetWeights(), geom->J, geom->detJ,
                   coeff, pa_data);
}

// PA Gradient Apply 2D kernel
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void PAGradientApply2D(const int NE,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Vector &op_,
                              const Vector &x_,
                              Vector &y_,
                              const int tr_d1d = 0,
                              const int te_d1d = 0,
                              const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, TR_D1D);
   auto G = Reshape(g.Read(), Q1D, TR_D1D);
   auto Bt = Reshape(bt.Read(), TE_D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D*Q1D, 2,2, NE);
   auto x = Reshape(x_.Read(), TR_D1D, TR_D1D, NE);
   auto y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, 2, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
      const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = 2;
      // the following variables are evaluated at compile time
      constexpr int max_TE_D1D = T_TE_D1D ? T_TE_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      double grad[max_Q1D][max_Q1D][VDIM];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < TR_D1D; ++dy)
      {
         double gradX[max_Q1D][VDIM];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < TR_D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * G(qx,dx);
               gradX[qx][1] += s * B(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][0] * wy;
               grad[qy][qx][1] += gradX[qx][1] * wDy;
            }
         }
      }
      // We've now calculated grad(p) = [Dxy, xDy] in plane
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;
            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = gradX*op(q,0,0,e) + gradY*op(q,1,0,e);
            grad[qy][qx][1] = gradX*op(q,0,1,e) + gradY*op(q,1,1,e);
         }
      }
      // We've now calculated grad = grad p * op
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double opX[max_TE_D1D][VDIM];
         for (int dx = 0; dx < TE_D1D; ++dx)
         {
            opX[dx][0] = 0.0;
            opX[dx][1] = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               opX[dx][0] += Bt(dx,qx)*grad[qy][qx][0];
               opX[dx][1] += Bt(dx,qx)*grad[qy][qx][1];
            }
         }
         for (int dy = 0; dy < TE_D1D; ++dy)
         {
            const double wy = Bt(dy,qy);
            for (int dx = 0; dx < TE_D1D; ++dx)
            {
               y(dx,dy,0,e) += wy*opX[dx][0];
               y(dx,dy,1,e) += wy*opX[dx][1];
            }
         }
      }
      // We've now calculated y = u * grad
   });
}

// PA Gradient Apply 2D kernel transpose
template<int T_TR_D1D = 0, int T_TE_D1D = 0, int T_Q1D = 0>
static void PAGradientApplyTranspose2D(const int NE,
                                       const Array<double> &bt,
                                       const Array<double> &gt,
                                       const Array<double> &b,
                                       const Vector &op_,
                                       const Vector &x_,
                                       Vector &y_,
                                       const int tr_d1d = 0,
                                       const int te_d1d = 0,
                                       const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto Bt = Reshape(bt.Read(), TR_D1D, Q1D);
   auto Gt = Reshape(gt.Read(), TR_D1D, Q1D);
   auto B = Reshape(b.Read(), Q1D, TE_D1D);
   auto op = Reshape(op_.Read(), Q1D*Q1D, 2,2, NE);
   auto x = Reshape(x_.Read(), TE_D1D, TE_D1D, 2, NE);
   auto y = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
      const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = 2;
      // the following variables are evaluated at compile time
      constexpr int max_TR_D1D = T_TR_D1D ? T_TR_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      // B: testdof-to-quad
      double Bxy[max_Q1D][max_Q1D][VDIM];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            Bxy[qy][qx][0] = 0.0;
            Bxy[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < TE_D1D; ++dy)
      {
         double Bx[max_Q1D][VDIM];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            Bx[qx][0] = 0.0;
            Bx[qx][1] = 0.0;
         }
         for (int dx = 0; dx < TE_D1D; ++dx)
         {
            const double s0 = x(dx,dy,0,e);
            const double s1 = x(dx,dy,1,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Bx[qx][0] += s0 * B(qx,dx);
               Bx[qx][1] += s1 * B(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Bxy[qy][qx][0] += Bx[qx][0] * wy;
               Bxy[qy][qx][1] += Bx[qx][1] * wy;
            }
         }
      }
      // D: quad-to-quad
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;
            const double Bxy0 = Bxy[qy][qx][0];
            const double Bxy1 = Bxy[qy][qx][1];

            Bxy[qy][qx][0] = Bxy0*op(q,0,0,e) + Bxy1*op(q,0,1,e);
            Bxy[qy][qx][1] = Bxy0*op(q,1,0,e) + Bxy1*op(q,1,1,e);
         }
      }
      // Bt: quad-to-trialdof
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double Btx[max_TR_D1D][VDIM];
         for (int dx = 0; dx < TR_D1D; ++dx)
         {
            Btx[dx][0] = 0.0;
            Btx[dx][1] = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Btx[dx][0] += Gt(dx,qx)*Bxy[qy][qx][0];
               Btx[dx][1] += Bt(dx,qx)*Bxy[qy][qx][1];
            }
         }
         for (int dy = 0; dy < TR_D1D; ++dy)
         {
            const double wy = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < TR_D1D; ++dx)
            {
               y(dx,dy,e) += wy*Btx[dx][0] + wDy*Btx[dx][1];
            }
         }
      }
   });
}

// PA Gradient Apply 3D kernel
template<const int T_TR_D1D = 0, const int T_TE_D1D = 0, const int T_Q1D = 0>
static void PAGradientApply3D(const int NE,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Vector &op_,
                              const Vector &x_,
                              Vector &y_,
                              int tr_d1d = 0,
                              int te_d1d = 0,
                              int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, TR_D1D);
   auto G = Reshape(g.Read(), Q1D, TR_D1D);
   auto Bt = Reshape(bt.Read(), TE_D1D, Q1D);
   auto op = Reshape(op_.Read(), Q1D*Q1D*Q1D, 3,3, NE);
   auto x = Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, NE);
   auto y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 3, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
      const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = 3;
      // the following variables are evaluated at compile time
      constexpr int max_TE_D1D = T_TE_D1D ? T_TE_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      double grad[max_Q1D][max_Q1D][max_Q1D][VDIM];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < TR_D1D; ++dz)
      {
         double gradXY[max_Q1D][max_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < TR_D1D; ++dy)
         {
            double gradX[max_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < TR_D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }
      // We've now calculated grad(p) = [Dxyz, xDyz, xyDz] in plane
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + (qy + qz * Q1D) * Q1D;
               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];

               grad[qz][qy][qx][0] = gradX*op(q,0,0,e) + gradY*op(q,1,0,e) + gradZ*op(q,2,0,e);
               grad[qz][qy][qx][1] = gradX*op(q,0,1,e) + gradY*op(q,1,1,e) + gradZ*op(q,2,1,e);
               grad[qz][qy][qx][2] = gradX*op(q,0,2,e) + gradY*op(q,1,2,e) + gradZ*op(q,2,2,e);
            }
         }
      }
      // We've now calculated grad = grad p * op
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double opXY[max_TE_D1D][max_TE_D1D][VDIM];
         for (int dy = 0; dy < TE_D1D; ++dy)
         {
            for (int dx = 0; dx < TE_D1D; ++dx)
            {
               opXY[dy][dx][0] = 0.0;
               opXY[dy][dx][1] = 0.0;
               opXY[dy][dx][2] = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double opX[max_TE_D1D][VDIM];
            for (int dx = 0; dx < TE_D1D; ++dx)
            {
               opX[dx][0] = 0.0;
               opX[dx][1] = 0.0;
               opX[dx][2] = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  opX[dx][0] += Bt(dx,qx)*grad[qz][qy][qx][0];
                  opX[dx][1] += Bt(dx,qx)*grad[qz][qy][qx][1];
                  opX[dx][2] += Bt(dx,qx)*grad[qz][qy][qx][2];
               }
            }
            for (int dy = 0; dy < TE_D1D; ++dy)
            {
               for (int dx = 0; dx < TE_D1D; ++dx)
               {
                  opXY[dy][dx][0] += Bt(dy,qy)*opX[dx][0];
                  opXY[dy][dx][1] += Bt(dy,qy)*opX[dx][1];
                  opXY[dy][dx][2] += Bt(dy,qy)*opX[dx][2];
               }
            }
         }
         for (int dz = 0; dz < TE_D1D; ++dz)
         {
            for (int dy = 0; dy < TE_D1D; ++dy)
            {
               for (int dx = 0; dx < TE_D1D; ++dx)
               {
                  y(dx,dy,dz,0,e) += Bt(dz,qz)*opXY[dy][dx][0];
                  y(dx,dy,dz,1,e) += Bt(dz,qz)*opXY[dy][dx][1];
                  y(dx,dy,dz,2,e) += Bt(dz,qz)*opXY[dy][dx][2];
               }
            }
         }
      }
      // We've now calculated y = u * grad
   });
}

// PA Gradient Apply 3D kernel
template<const int T_TR_D1D = 0, const int T_TE_D1D = 0, const int T_Q1D = 0>
static void PAGradientApplyTranspose3D(const int NE,
                                       const Array<double> &bt,
                                       const Array<double> &gt,
                                       const Array<double> &b,
                                       const Vector &op_,
                                       const Vector &x_,
                                       Vector &y_,
                                       int tr_d1d = 0,
                                       int te_d1d = 0,
                                       int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto Bt = Reshape(bt.Read(), TR_D1D, Q1D);
   auto Gt = Reshape(gt.Read(), TR_D1D, Q1D);
   auto B = Reshape(b.Read(), Q1D, TE_D1D);
   auto op = Reshape(op_.Read(), Q1D*Q1D*Q1D, 3,3, NE);
   auto x = Reshape(x_.Read(), TE_D1D, TE_D1D, TE_D1D, 3, NE);
   auto y = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, TR_D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
      const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = 3;
      // the following variables are evaluated at compile time
      constexpr int max_TR_D1D = T_TR_D1D ? T_TR_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      // B: testdof-to-quad
      double Bxyz[max_Q1D][max_Q1D][max_Q1D][VDIM];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Bxyz[qz][qy][qx][0] = 0.0;
               Bxyz[qz][qy][qx][1] = 0.0;
               Bxyz[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < TE_D1D; ++dz)
      {
         double Bxy[max_Q1D][max_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Bxy[qy][qx][0] = 0.0;
               Bxy[qy][qx][1] = 0.0;
               Bxy[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < TE_D1D; ++dy)
         {
            double Bx[max_Q1D][3];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Bx[qx][0] = 0.0;
               Bx[qx][1] = 0.0;
               Bx[qx][2] = 0.0;
            }
            for (int dx = 0; dx < TE_D1D; ++dx)
            {
               const double s0 = x(dx,dy,dz,0,e);
               const double s1 = x(dx,dy,dz,1,e);
               const double s2 = x(dx,dy,dz,2,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  Bx[qx][0] += s0 * B(qx,dx);
                  Bx[qx][1] += s1 * B(qx,dx);
                  Bx[qx][2] += s2 * B(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  Bxy[qy][qx][0] += Bx[qx][0] * wy;
                  Bxy[qy][qx][1] += Bx[qx][1] * wy;
                  Bxy[qy][qx][2] += Bx[qx][2] * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz = B(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  Bxyz[qz][qy][qx][0] += Bxy[qy][qx][0] * wz;
                  Bxyz[qz][qy][qx][1] += Bxy[qy][qx][1] * wz;
                  Bxyz[qz][qy][qx][2] += Bxy[qy][qx][2] * wz;
               }
            }
         }
      }
      // D: quad-to-quad
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + (qy + qz * Q1D) * Q1D;
               const double Bxyz0 = Bxyz[qz][qy][qx][0];
               const double Bxyz1 = Bxyz[qz][qy][qx][1];
               const double Bxyz2 = Bxyz[qz][qy][qx][2];

               Bxyz[qz][qy][qx][0] = Bxyz0*op(q,0,0,e) + Bxyz1*op(q,0,1,e) + Bxyz2*op(q,0,2,e);
               Bxyz[qz][qy][qx][1] = Bxyz0*op(q,1,0,e) + Bxyz1*op(q,1,1,e) + Bxyz2*op(q,1,2,e);
               Bxyz[qz][qy][qx][2] = Bxyz0*op(q,2,0,e) + Bxyz1*op(q,2,1,e) + Bxyz2*op(q,2,2,e);
            }
         }
      }
      // Bt: quad-to-trialdof
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double Btxy[max_TR_D1D][max_TR_D1D][VDIM];
         for (int dy = 0; dy < TR_D1D; ++dy)
         {
            for (int dx = 0; dx < TR_D1D; ++dx)
            {
               Btxy[dy][dx][0] = 0.0;
               Btxy[dy][dx][1] = 0.0;
               Btxy[dy][dx][2] = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double Btx[max_TR_D1D][VDIM];
            for (int dx = 0; dx < TR_D1D; ++dx)
            {
               Btx[dx][0] = 0.0;
               Btx[dx][1] = 0.0;
               Btx[dx][2] = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  Btx[dx][0] += Gt(dx,qx)*Bxyz[qz][qy][qx][0];
                  Btx[dx][1] += Bt(dx,qx)*Bxyz[qz][qy][qx][1];
                  Btx[dx][2] += Bt(dx,qx)*Bxyz[qz][qy][qx][2];
               }
            }
            for (int dy = 0; dy < TR_D1D; ++dy)
            {
               const double wy = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
               for (int dx = 0; dx < TR_D1D; ++dx)
               {
                  Btxy[dy][dx][0] += wy *Btx[dx][0];
                  Btxy[dy][dx][1] += wDy*Btx[dx][1];
                  Btxy[dy][dx][2] += wy *Btx[dx][2];
               }
            }
         }
         for (int dz = 0; dz < TR_D1D; ++dz)
         {
            const double wz = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
            for (int dy = 0; dy < TR_D1D; ++dy)
            {
               for (int dx = 0; dx < TR_D1D; ++dx)
               {
                  y(dx,dy,dz,e) += wz *Btxy[dy][dx][0] +
                                   wz *Btxy[dy][dx][1] +
                                   wDz*Btxy[dy][dx][2];
               }
            }
         }
      }
   });
}

// Shared memory PA Gradient Apply 3D kernel
template<const int T_TR_D1D = 0, const int T_TE_D1D = 0, const int T_Q1D = 0>
static void SmemPAGradientApply3D(const int NE,
                                  const Array<double> &b_,
                                  const Array<double> &g_,
                                  const Array<double> &bt_,
                                  const Vector &d_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int tr_d1d = 0,
                                  const int te_d1d = 0,
                                  const int q1d = 0)
{
   const int TR_D1D = T_TR_D1D ? T_TR_D1D : tr_d1d;
   const int TE_D1D = T_TE_D1D ? T_TE_D1D : te_d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(TR_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TE_D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(TR_D1D <= Q1D, "");
   MFEM_VERIFY(TE_D1D <= Q1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   auto b = Reshape(b_.Read(), Q1D, TR_D1D);
   auto g = Reshape(g_.Read(), Q1D, TR_D1D);
   auto bt = Reshape(bt_.Read(), TE_D1D, Q1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D*Q1D, 3, 3, NE);
   auto x = Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, NE);
   auto y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 3, NE);

   mfem::forall_3D(NE, (Q1D>8)?8:Q1D, (Q1D>8)?8:Q1D, (Q1D>8)?8:Q1D,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1DR = T_TR_D1D ? T_TR_D1D : tr_d1d;
      const int D1DE = T_TE_D1D ? T_TE_D1D : te_d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD1R = T_TR_D1D ? T_TR_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MD1E = T_TE_D1D ? T_TE_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MD1 = MD1E > MD1R ? MD1E : MD1R;
      constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      MFEM_SHARED double sm0[3][MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[3][MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);

      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);

      double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
      double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
      double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);

      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
      double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1);
      double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2);

      double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
      double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1);
      double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);
      MFEM_FOREACH_THREAD(dz,z,D1DR)
      {
         MFEM_FOREACH_THREAD(dy,y,D1DR)
         {
            MFEM_FOREACH_THREAD(dx,x,D1DR)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1DR)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1DR)
      {
         MFEM_FOREACH_THREAD(dy,y,D1DR)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1DR; ++dx)
               {
                  const double coord = X[dz][dy][dx];
                  u += coord * B[qx][dx];
                  v += coord * G[qx][dx];
               }
               DDQ0[dz][dy][qx] = u;
               DDQ1[dz][dy][qx] = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1DR)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dy = 0; dy < D1DR; ++dy)
               {
                  u += DDQ1[dz][dy][qx] * B[qy][dy];
                  v += DDQ0[dz][dy][qx] * G[qy][dy];
                  w += DDQ0[dz][dy][qx] * B[qy][dy];
               }
               DQQ0[dz][qy][qx] = u;
               DQQ1[dz][qy][qx] = v;
               DQQ2[dz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dz = 0; dz < D1DR; ++dz)
               {
                  u += DQQ0[dz][qy][qx] * B[qz][dz];
                  v += DQQ1[dz][qy][qx] * B[qz][dz];
                  w += DQQ2[dz][qy][qx] * G[qz][dz];
               }
               QQQ0[qz][qy][qx] = u;
               QQQ1[qz][qy][qx] = v;
               QQQ2[qz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const int q = qx + (qy + qz * Q1D) * Q1D;
               const double gX = QQQ0[qz][qy][qx];
               const double gY = QQQ1[qz][qy][qx];
               const double gZ = QQQ2[qz][qy][qx];
               QQQ0[qz][qy][qx] = (D(q,0,0,e)*gX) + (D(q,1,0,e)*gY) + (D(q,2,0,e)*gZ);
               QQQ1[qz][qy][qx] = (D(q,0,1,e)*gX) + (D(q,1,1,e)*gY) + (D(q,2,1,e)*gZ);
               QQQ2[qz][qy][qx] = (D(q,0,2,e)*gX) + (D(q,1,2,e)*gY) + (D(q,2,2,e)*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1DE)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[d][q] = bt(d,q);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1DE)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ0[qz][qy][qx] * Bt[dx][qx];
                  v += QQQ1[qz][qy][qx] * Bt[dx][qx];
                  w += QQQ2[qz][qy][qx] * Bt[dx][qx];
               }
               QQD0[qz][qy][dx] = u;
               QQD1[qz][qy][dx] = v;
               QQD2[qz][qy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1DE)
         {
            MFEM_FOREACH_THREAD(dx,x,D1DE)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += QQD0[qz][qy][dx] * Bt[dy][qy];
                  v += QQD1[qz][qy][dx] * Bt[dy][qy];
                  w += QQD2[qz][qy][dx] * Bt[dy][qy];
               }
               QDD0[qz][dy][dx] = u;
               QDD1[qz][dy][dx] = v;
               QDD2[qz][dy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1DE)
      {
         MFEM_FOREACH_THREAD(dy,y,D1DE)
         {
            MFEM_FOREACH_THREAD(dx,x,D1DE)
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += QDD0[qz][dy][dx] * Bt[dz][qz];
                  v += QDD1[qz][dy][dx] * Bt[dz][qz];
                  w += QDD2[qz][dy][dx] * Bt[dz][qz];
               }
               y(dx,dy,dz,0,e) += u;
               y(dx,dy,dz,1,e) += v;
               y(dx,dy,dz,2,e) += w;
            }
         }
      }
   });
}

static void PAGradientApply(const int dim,
                            const int TR_D1D,
                            const int TE_D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &B,
                            const Array<double> &G,
                            const Array<double> &Bt,
                            const Vector &op,
                            const Vector &x,
                            Vector &y)
{
   if (dim == 2)
   {
      return PAGradientApply2D(NE,B,G,Bt,op,x,y,TR_D1D,TE_D1D,Q1D);
   }
   if (dim == 3)
   {
      return PAGradientApply3D(NE,B,G,Bt,op,x,y,TR_D1D,TE_D1D,Q1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

static void PAGradientApplyTranspose(const int dim,
                                     const int TR_D1D,
                                     const int TE_D1D,
                                     const int Q1D,
                                     const int NE,
                                     const Array<double> &Bt,
                                     const Array<double> &Gt,
                                     const Array<double> &B,
                                     const Vector &op,
                                     const Vector &x,
                                     Vector &y)
{
   if (dim == 2)
   {
      return PAGradientApplyTranspose2D(NE,Bt,Gt,B,op,x,y,TR_D1D,TE_D1D,Q1D);
   }
   if (dim == 3)
   {
      return PAGradientApplyTranspose3D(NE,Bt,Gt,B,op,x,y,TR_D1D,TE_D1D,Q1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA Gradient Apply kernel
void GradientIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PAGradientApply(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                   trial_maps->B, trial_maps->G, test_maps->Bt, pa_data,
                   x, y);
}

// PA Gradient Apply transpose kernel
void GradientIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   PAGradientApplyTranspose(dim, trial_dofs1D, test_dofs1D, quad1D, ne,
                            trial_maps->Bt, trial_maps->Gt, test_maps->B, pa_data,
                            x, y);
}

} // namespace mfem
