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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "qfunction.hpp"
#include "restriction.hpp"

using namespace std;

namespace mfem
{
// PA DG Trace Integrator
static void PADGDiffusionSetup2D(const int Q1D,
                                 const int NF,
                                 const Array<double> &w,
                                 const Vector &det,
                                 const Vector &nor,
                                 const Vector &rho,
                                 const Vector &vel,
                                 const double alpha,
                                 const double beta,
                                 Vector &op)
{
   // const int VDIM = 2;

   // auto d = Reshape(det.Read(), Q1D, NF);
   // auto n = Reshape(nor.Read(), Q1D, VDIM, NF);
   // const bool const_r = rho.Size() == 1;
   // auto R =
   //    const_r ? Reshape(rho.Read(), 1,1) : Reshape(rho.Read(), Q1D,NF);
   // const bool const_v = vel.Size() == 2;
   // auto V =
   //    const_v ? Reshape(vel.Read(), 2,1,1) : Reshape(vel.Read(), 2,Q1D,NF);
   // auto W = w.Read();
   // auto qd = Reshape(op.Write(), Q1D, 2, 2, NF);

   // mfem::forall(Q1D*NF, [=] MFEM_HOST_DEVICE (int tid)
   // {
   //    const int f = tid / Q1D;
   //    const int q = tid % Q1D;
   //    {
   //       const double r = const_r ? R(0,0) : R(q,f);
   //       const double v0 = const_v ? V(0,0,0) : V(0,q,f);
   //       const double v1 = const_v ? V(1,0,0) : V(1,q,f);
   //       const double dot = n(q,0,f) * v0 + n(q,1,f) * v1;
   //       const double abs = dot > 0.0 ? dot : -dot;
   //       const double w = W[q]*r*d(q,f);
   //       qd(q,0,0,f) = w*( alpha/2 * dot + beta * abs );
   //       qd(q,1,0,f) = w*( alpha/2 * dot - beta * abs );
   //       qd(q,0,1,f) = w*(-alpha/2 * dot - beta * abs );
   //       qd(q,1,1,f) = w*(-alpha/2 * dot + beta * abs );
   //    }
   // });
}

static void PADGDiffusionSetup(const int dim,
                               const int D1D,
                               const int Q1D,
                               const int NF,
                               const Array<double> &W,
                               const Vector &det,
                               const Vector &nor,
                               const Vector &rho,
                               const Vector &u,
                               const double alpha,
                               const double beta,
                               Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADGTraceSetup"); }
   if (dim == 2)
   {
      PADGDiffusionSetup2D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
   }
   if (dim == 3)
   {
      MFEM_ABORT("Not yet implemented");
   }
}

void DGDiffusionIntegrator::SetupPA(const FiniteElementSpace &fes,
                                    FaceType type)
{
   // const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
   //                       Device::GetDeviceMemoryType() : pa_mt;

   // nf = fes.GetNFbyType(type);
   // if (nf==0) { return; }
   // // Assumes tensor-product elements
   // Mesh *mesh = fes.GetMesh();
   // const FiniteElement &el =
   //    *fes.GetTraceElement(0, fes.GetMesh()->GetFaceGeometry(0));
   // FaceElementTransformations &T0 =
   //    *fes.GetMesh()->GetFaceElementTransformations(0);
   // const IntegrationRule *ir = IntRule?
   //                             IntRule:
   //                             &GetRule(el.GetGeomType(), el.GetOrder(), T0);
   // const int symmDims = 4;
   // nq = ir->GetNPoints();
   // dim = mesh->Dimension();
   // geom = mesh->GetFaceGeometricFactors(
   //           *ir,
   //           FaceGeometricFactors::DETERMINANTS |
   //           FaceGeometricFactors::NORMALS, type, mt);
   // maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   // dofs1D = maps->ndof;
   // quad1D = maps->nqpt;
   // pa_data.SetSize(symmDims * nq * nf, Device::GetMemoryType());

   // FaceQuadratureSpace qs(*mesh, *ir, type);
   // CoefficientVector vel(*u, qs, CoefficientStorage::COMPRESSED);

   // CoefficientVector r(qs, CoefficientStorage::COMPRESSED);
   // if (rho == nullptr)
   // {
   //    r.SetConstant(1.0);
   // }
   // else if (ConstantCoefficient *const_rho = dynamic_cast<ConstantCoefficient*>
   //                                           (rho))
   // {
   //    r.SetConstant(const_rho->constant);
   // }
   // else if (QuadratureFunctionCoefficient* qf_rho =
   //             dynamic_cast<QuadratureFunctionCoefficient*>(rho))
   // {
   //    r.MakeRef(qf_rho->GetQuadFunction());
   // }
   // else
   // {
   //    r.SetSize(nq * nf);
   //    auto C_vel = Reshape(vel.HostRead(), dim, nq, nf);
   //    auto n = Reshape(geom->normal.HostRead(), nq, dim, nf);
   //    auto C = Reshape(r.HostWrite(), nq, nf);
   //    int f_ind = 0;
   //    for (int f = 0; f < mesh->GetNumFacesWithGhost(); ++f)
   //    {
   //       Mesh::FaceInformation face = mesh->GetFaceInformation(f);
   //       if (face.IsNonconformingCoarse() || !face.IsOfFaceType(type))
   //       {
   //          // We skip nonconforming coarse faces as they are treated
   //          // by the corresponding nonconforming fine faces.
   //          continue;
   //       }
   //       FaceElementTransformations &T =
   //          *fes.GetMesh()->GetFaceElementTransformations(f);
   //       for (int q = 0; q < nq; ++q)
   //       {
   //          // Convert to lexicographic ordering
   //          int iq = ToLexOrdering(dim, face.element[0].local_face_id,
   //                                 quad1D, q);

   //          T.SetAllIntPoints(&ir->IntPoint(q));
   //          const IntegrationPoint &eip1 = T.GetElement1IntPoint();
   //          const IntegrationPoint &eip2 = T.GetElement2IntPoint();
   //          double rq;

   //          if (face.IsBoundary())
   //          {
   //             rq = rho->Eval(*T.Elem1, eip1);
   //          }
   //          else
   //          {
   //             double udotn = 0.0;
   //             for (int d=0; d<dim; ++d)
   //             {
   //                udotn += C_vel(d,iq,f_ind)*n(iq,d,f_ind);
   //             }
   //             if (udotn >= 0.0) { rq = rho->Eval(*T.Elem2, eip2); }
   //             else { rq = rho->Eval(*T.Elem1, eip1); }
   //          }
   //          C(iq,f_ind) = rq;
   //       }
   //       f_ind++;
   //    }
   //    MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");
   // }
   // PADGTraceSetup(dim, dofs1D, quad1D, nf, ir->GetWeights(),
   //                geom->detJ, geom->normal, r, vel,
   //                alpha, beta, pa_data);
}

void DGDiffusionIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace&
                                                    fes)
{
   SetupPA(fes, FaceType::Interior);
}

void DGDiffusionIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace&
                                                    fes)
{
   SetupPA(fes, FaceType::Boundary);
}

template<int T_D1D = 0, int T_Q1D = 0> static
void PADGDiffusionApply2D(const int NF,
                          const Array<double> &b,
                          const Array<double> &bt,
                          const Vector &op_,
                          const Vector &x_,
                          const Vector &dxdn_,
                          Vector &y_,
                          Vector &dydn_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   // const int VDIM = 1;
   // const int D1D = T_D1D ? T_D1D : d1d;
   // const int Q1D = T_Q1D ? T_Q1D : q1d;
   // MFEM_VERIFY(D1D <= MAX_D1D, "");
   // MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   // auto B = Reshape(b.Read(), Q1D, D1D);
   // auto Bt = Reshape(bt.Read(), D1D, Q1D);
   // auto op = Reshape(op_.Read(), Q1D, 2, 2, NF);
   // auto x = Reshape(x_.Read(), D1D, VDIM, 2, NF);
   // auto y = Reshape(y_.ReadWrite(), D1D, VDIM, 2, NF);

   // mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   // {
   //    const int VDIM = 1;
   //    const int D1D = T_D1D ? T_D1D : d1d;
   //    const int Q1D = T_Q1D ? T_Q1D : q1d;
   //    // the following variables are evaluated at compile time
   //    constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
   //    constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
   //    double u0[max_D1D][VDIM];
   //    double u1[max_D1D][VDIM];
   //    for (int d = 0; d < D1D; d++)
   //    {
   //       for (int c = 0; c < VDIM; c++)
   //       {
   //          u0[d][c] = x(d,c,0,f);
   //          u1[d][c] = x(d,c,1,f);
   //       }
   //    }
   //    double Bu0[max_Q1D][VDIM];
   //    double Bu1[max_Q1D][VDIM];
   //    for (int q = 0; q < Q1D; ++q)
   //    {
   //       for (int c = 0; c < VDIM; c++)
   //       {
   //          Bu0[q][c] = 0.0;
   //          Bu1[q][c] = 0.0;
   //       }
   //       for (int d = 0; d < D1D; ++d)
   //       {
   //          const double b = B(q,d);
   //          for (int c = 0; c < VDIM; c++)
   //          {
   //             Bu0[q][c] += b*u0[d][c];
   //             Bu1[q][c] += b*u1[d][c];
   //          }
   //       }
   //    }
   //    double DBu[max_Q1D][VDIM];
   //    for (int q = 0; q < Q1D; ++q)
   //    {
   //       for (int c = 0; c < VDIM; c++)
   //       {
   //          DBu[q][c] = op(q,0,0,f)*Bu0[q][c] + op(q,1,0,f)*Bu1[q][c];
   //       }
   //    }
   //    double BDBu[max_D1D][VDIM];
   //    for (int d = 0; d < D1D; ++d)
   //    {
   //       for (int c = 0; c < VDIM; c++)
   //       {
   //          BDBu[d][c] = 0.0;
   //       }
   //       for (int q = 0; q < Q1D; ++q)
   //       {
   //          const double b = Bt(d,q);
   //          for (int c = 0; c < VDIM; c++)
   //          {
   //             BDBu[d][c] += b*DBu[q][c];
   //          }
   //       }
   //       for (int c = 0; c < VDIM; c++)
   //       {
   //          y(d,c,0,f) +=  BDBu[d][c];
   //          y(d,c,1,f) += -BDBu[d][c];
   //       }
   //    }
   // });
}

static void PADGDiffusionApply(const int dim,
                               const int D1D,
                               const int Q1D,
                               const int NF,
                               const Array<double> &B,
                               const Array<double> &Bt,
                               const Vector &op,
                               const Vector &x,
                               const Vector &dxdn,
                               Vector &y,
                               Vector &dydn)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PADGDiffusionApply2D<2,2>(NF,B,Bt,op,x,dxdn,y,dydn);
         case 0x33: return PADGDiffusionApply2D<3,3>(NF,B,Bt,op,x,dxdn,y,dydn);
         case 0x44: return PADGDiffusionApply2D<4,4>(NF,B,Bt,op,x,dxdn,y,dydn);
         case 0x55: return PADGDiffusionApply2D<5,5>(NF,B,Bt,op,x,dxdn,y,dydn);
         case 0x66: return PADGDiffusionApply2D<6,6>(NF,B,Bt,op,x,dxdn,y,dydn);
         case 0x77: return PADGDiffusionApply2D<7,7>(NF,B,Bt,op,x,dxdn,y,dydn);
         case 0x88: return PADGDiffusionApply2D<8,8>(NF,B,Bt,op,x,dxdn,y,dydn);
         case 0x99: return PADGDiffusionApply2D<9,9>(NF,B,Bt,op,x,dxdn,y,dydn);
         default:   return PADGDiffusionApply2D(NF,B,Bt,op,x,dxdn,y,dydn,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      MFEM_ABORT("Not yet implemented");
   }
   MFEM_ABORT("Unknown kernel.");
}

void DGDiffusionIntegrator::AddMultPAFaceNormalDerivatives(const Vector &x,
                                                           const Vector &dxdn, Vector &y, Vector &dydn) const
{
   PADGDiffusionApply(dim, dofs1D, quad1D, nf,
                      maps->B, maps->Bt,
                      pa_data, x, dxdn, y, dydn);
}

} // namespace mfem
