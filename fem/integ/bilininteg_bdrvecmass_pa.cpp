// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include <unordered_map>

namespace mfem
{

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PABdrVectorMassApply2D(const int NE,
                                   const Array<double> &B_,
                                   const Array<double> &Bt_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 2;
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B  = Reshape(B_.Read(), Q1D, D1D);
   auto Bt = Reshape(Bt_.Read(), D1D, Q1D);
   auto D  = Reshape(d_.Read(), Q1D, VDIM*VDIM, NE);
   auto x  = Reshape(x_.Read(), D1D, VDIM, NE);
   auto y  = Reshape(y_.ReadWrite(), D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      double sol_q[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            sol_q[q][c] = 0.0;
         }
      }

      // dof -> quad.
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            const double s = x(d,c,e);
            for (int q = 0; q < Q1D; ++q)
            {
               sol_q[q][c] += B(q,d) * s;
            }
         }
      }

      // quad data.
      double mass_q[max_Q1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            mass_q[q][c] = 0.0;
            for (int cc = 0; cc < VDIM; ++cc)
            {
               mass_q[q][c] += D(q,cc + c*VDIM,e) * sol_q[q][cc];
            }
         }
      }

      // quad -> dof.
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            double s = 0.0;
            for (int q = 0; q < Q1D; ++q)
            {
               s += Bt(d,q) * mass_q[q][c];
            }
            y(d,c,e) += s;
         }
      }
   });
}

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PABdrVectorMassApply3D(const int NE,
                                   const Array<double> &B_,
                                   const Array<double> &Bt_,
                                   const Vector &q_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 3;
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B  = Reshape(B_.Read(), Q1D, D1D);
   auto Bt = Reshape(Bt_.Read(), D1D, Q1D);
   auto D  = Reshape(q_.Read(), Q1D, Q1D, VDIM*VDIM, NE);
   auto x  = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y  = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      double sol_xy[max_Q1D][max_Q1D][VDIM];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               sol_xy[qy][qx][c] = 0.0;
            }
         }
      }

      // dof -> quad.
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[max_Q1D][VDIM];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               sol_x[qx][c] = 0.0;
            }
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               const double s = x(dx,dy,c,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx][c] += B(qx,dx) * s;
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double d2q = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  sol_xy[qy][qx][c] += d2q * sol_x[qx][c];
               }
            }
         }
      }

      // quad data.
      double mass_xy[max_Q1D][max_Q1D][VDIM];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass_xy[qy][qx][c] = 0.0;
               for (int cc = 0; cc < VDIM; ++cc)
               {
                  mass_xy[qy][qx][c] +=
                     D(qx,qy,cc + c*VDIM,e) * sol_xy[qy][qx][cc];
               }
            }
         }
      }

      // quad -> dof.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[max_D1D][VDIM];
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               sol_x[dx][c] = 0.0;
            }
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               const double s = mass_xy[qy][qx][c];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx][c] += Bt(dx,qx) * s;
               }
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double q2d = Bt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  y(dx,dy,c,e) += q2d * sol_x[dx][c];
               }
            }
         }
      }
   });
}

static void PABdrVectorMassApplyDense(const int NF,
                                      const int face_dofs,
                                      const int vdim,
                                      const Vector &pa_data,
                                      const Vector &x,
                                      Vector &y)
{
   auto D = Reshape(pa_data.Read(), face_dofs, vdim, face_dofs, vdim, NF);
   auto X = Reshape(x.Read(), face_dofs, vdim, NF);
   auto Y = Reshape(y.ReadWrite(), face_dofs, vdim, NF);
   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      for (int i = 0; i < face_dofs; i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            double val = 0.0;
            for (int j = 0; j < face_dofs; j++)
            {
               for (int cc = 0; cc < vdim; cc++)
               {
                  val += D(i,c,j,cc,f) * X(j,cc,f);
               }
            }
            Y(i,c,f) += val;
         }
      }
   });
}

void BoundaryVectorMassIntegrator::
AssemblePABoundaryFaces(const FiniteElementSpace &fes)
{
   nf = fes.GetNFbyType(FaceType::Boundary);
   if (nf == 0) { return; }
   ne = nf;

   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el =
      *fes.GetTraceElement(0, fes.GetMesh()->GetFaceGeometry(0));

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * fes.FEColl()->GetOrder();
      ir = &IntRules.Get(mesh->GetFaceGeometry(0), order);
   }

   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Dimension not supported.");
   vdim = (vdim == -1) ? dim : vdim;
   MFEM_VERIFY(vdim == fes.GetVDim(), "vdim != fes.GetVDim()");
   MFEM_VERIFY(vdim == dim, "vdim != dim");

   const MemoryType mt = pa_mt == MemoryType::DEFAULT
                         ? Device::GetDeviceMemoryType()
                         : pa_mt;
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;

   const int nq = ir->GetNPoints();
   const int face_dofs = el.GetDof();
   pa_data.SetSize(face_dofs*vdim * face_dofs*vdim * nf, mt);

   std::unordered_map<int, int> face_to_be;
   for (int be = 0; be < mesh->GetNBE(); be++)
   {
      face_to_be[mesh->GetBdrElementFaceIndex(be)] = be;
   }
   const Array<int> &face_indices = mesh->GetFaceIndices(FaceType::Boundary);

   DenseMatrix mcoeff(vdim);
   Vector shape(face_dofs), lex_shape(face_dofs);
   const auto *tensor_el = dynamic_cast<const TensorBasisElement*>(&el);
   MFEM_VERIFY(tensor_el, "Boundary PA requires a tensor basis element.");
   const Array<int> &dof_map = tensor_el->GetDofMap();
   auto D = Reshape(pa_data.HostWrite(), face_dofs, vdim, face_dofs, vdim, nf);
   for (int f = 0; f < nf; f++)
   {
      for (int i = 0; i < face_dofs; i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            for (int j = 0; j < face_dofs; j++)
            {
               for (int cc = 0; cc < vdim; cc++) { D(i,c,j,cc,f) = 0.0; }
            }
         }
      }
      const auto face_to_be_it = face_to_be.find(face_indices[f]);
      auto b_face_tr = (face_to_be_it == face_to_be.end()) ? nullptr :
                       mesh->GetBdrFaceTransformations(face_to_be_it->second);
      if (b_face_tr == nullptr)
      {
         continue;
      }
      for (int q = 0; q < nq; q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         b_face_tr->SetAllIntPoints(&ip);
         MQ->Eval(mcoeff, *b_face_tr, ip);
         el.CalcShape(ip, shape);
         if (dof_map.Size())
         {
            for (int i = 0; i < face_dofs; i++)
            {
               lex_shape(i) = shape(dof_map[i]);
            }
         }
         else
         {
            lex_shape = shape;
         }
         for (int i = 0; i < vdim; i++)
         {
            for (int j = 0; j < vdim; j++)
            {
               for (int di = 0; di < face_dofs; di++)
               {
                  for (int dj = 0; dj < face_dofs; dj++)
                  {
                     D(di,i,dj,j,f) += mcoeff(i,j) * lex_shape(di) * lex_shape(dj);
                  }
               }
            }
         }
      }
   }
}

void BoundaryVectorMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   const int face_dofs = (dim == 2) ? dofs1D : dofs1D*dofs1D;
   return PABdrVectorMassApplyDense(ne, face_dofs, vdim, pa_data, x, y);

   if (dim == 2)
   {
      MFEM_VERIFY(vdim == 2, "Not implemented for genereal vdim");
      return PABdrVectorMassApply2D(ne, maps->B, maps->Bt, pa_data,
                                    x, y, dofs1D, quad1D);
   }
   if (dim == 3)
   {
      MFEM_VERIFY(vdim == 3, "Not implemented for genereal vdim");
      return PABdrVectorMassApply3D(ne, maps->B, maps->Bt, pa_data,
                                    x, y, dofs1D, quad1D);
   }
}

} // namespace mfem
