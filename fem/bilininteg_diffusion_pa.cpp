// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "libceed/diffusion.hpp"

using namespace std;

namespace mfem
{

// PA Diffusion Integrator

// OCCA 2D Assemble kernel
#ifdef MFEM_USE_OCCA
static void OccaPADiffusionSetup2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &W,
                                   const Vector &J,
                                   const Vector &C,
                                   Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   const occa::memory o_C = OccaMemoryRead(C.GetMemory(), C.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const bool const_c = C.Size() == 1;
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup2D_ker;
   if (OccaDiffSetup2D_ker.find(id) == OccaDiffSetup2D_ker.end())
   {
      const occa::kernel DiffusionSetup2D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup2D", props);
      OccaDiffSetup2D_ker.emplace(id, DiffusionSetup2D);
   }
   OccaDiffSetup2D_ker.at(id)(NE, o_W, o_J, o_C, o_op, const_c);
}

static void OccaPADiffusionSetup3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &W,
                                   const Vector &J,
                                   const Vector &C,
                                   Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   const occa::memory o_C = OccaMemoryRead(C.GetMemory(), C.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const bool const_c = C.Size() == 1;
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup3D_ker;
   if (OccaDiffSetup3D_ker.find(id) == OccaDiffSetup3D_ker.end())
   {
      const occa::kernel DiffusionSetup3D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup3D", props);
      OccaDiffSetup3D_ker.emplace(id, DiffusionSetup3D);
   }
   OccaDiffSetup3D_ker.at(id)(NE, o_W, o_J, o_C, o_op, const_c);
}
#endif // MFEM_USE_OCCA

template<>
void PADiffusionSetup2D<2>(const int Q1D,
                           const int coeffDim,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           const Vector &c,
                           Vector &d)
{
   const bool symmetric = (coeffDim != 4);
   const bool const_c = c.Size() == 1;
   MFEM_VERIFY(coeffDim < 3 ||
               !const_c, "Constant matrix coefficient not supported");
   const auto W = Reshape(w.Read(), Q1D,Q1D);
   const auto J = Reshape(j.Read(), Q1D,Q1D,2,2,NE);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1,1) :
                  Reshape(c.Read(), coeffDim,Q1D,Q1D,NE);
   auto D = Reshape(d.Write(), Q1D,Q1D, symmetric ? 3 : 4, NE);
   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            const double J11 = J(qx,qy,0,0,e);
            const double J21 = J(qx,qy,1,0,e);
            const double J12 = J(qx,qy,0,1,e);
            const double J22 = J(qx,qy,1,1,e);
            const double w_detJ = W(qx,qy) / ((J11*J22)-(J21*J12));
            if (coeffDim == 3 || coeffDim == 4) // Matrix coefficient
            {
               // First compute entries of R = MJ^{-T}, without det J factor.
               const double M11 = C(0,qx,qy,e);
               const double M12 = C(1,qx,qy,e);
               const double M21 = symmetric ? M12 : C(2,qx,qy,e);
               const double M22 = symmetric ? C(2,qx,qy,e) : C(3,qx,qy,e);
               const double R11 = M11*J22 - M12*J12;
               const double R21 = M21*J22 - M22*J12;
               const double R12 = -M11*J21 + M12*J11;
               const double R22 = -M21*J21 + M22*J11;

               // Now set y to J^{-1}R.
               D(qx,qy,0,e) = w_detJ * ( J22*R11 - J12*R21); // 1,1
               D(qx,qy,1,e) = w_detJ * (-J21*R11 + J11*R21); // 2,1
               D(qx,qy,2,e) = w_detJ * (symmetric ? (-J21*R12 + J11*R22) :
               (J22*R12 - J12*R22)); // 2,2 or 1,2
               if (!symmetric)
               {
                  D(qx,qy,3,e) = w_detJ * (-J21*R12 + J11*R22); // 2,2
               }
            }
            else // Vector or scalar coefficient
            {
               const double C1 = const_c ? C(0,0,0,0) : C(0,qx,qy,e);
               const double C2 = const_c ? C(0,0,0,0) :
                                 (coeffDim == 2 ? C(1,qx,qy,e) : C(0,qx,qy,e));

               D(qx,qy,0,e) =  w_detJ * (C2*J12*J12 + C1*J22*J22); // 1,1
               D(qx,qy,1,e) = -w_detJ * (C2*J12*J11 + C1*J22*J21); // 1,2
               D(qx,qy,2,e) =  w_detJ * (C2*J11*J11 + C1*J21*J21); // 2,2
            }
         }
      }
   });
}

// PA Diffusion Assemble 2D kernel with 3D node coords
template<>
void PADiffusionSetup2D<3>(const int Q1D,
                           const int coeffDim,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           const Vector &c,
                           Vector &d)
{
   MFEM_VERIFY(coeffDim == 1, "Matrix and vector coefficients not supported");
   constexpr int DIM = 2;
   constexpr int SDIM = 3;
   const bool const_c = c.Size() == 1;
   const auto W = Reshape(w.Read(), Q1D,Q1D);
   const auto J = Reshape(j.Read(), Q1D,Q1D,SDIM,DIM,NE);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1) :
                  Reshape(c.Read(), Q1D,Q1D,NE);
   auto D = Reshape(d.Write(), Q1D,Q1D, 3, NE);
   MFEM_FORALL_2D(e, NE, Q1D,Q1D,1,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            const double wq = W(qx,qy);
            const double J11 = J(qx,qy,0,0,e);
            const double J21 = J(qx,qy,1,0,e);
            const double J31 = J(qx,qy,2,0,e);
            const double J12 = J(qx,qy,0,1,e);
            const double J22 = J(qx,qy,1,1,e);
            const double J32 = J(qx,qy,2,1,e);
            const double E = J11*J11 + J21*J21 + J31*J31;
            const double G = J12*J12 + J22*J22 + J32*J32;
            const double F = J11*J12 + J21*J22 + J31*J32;
            const double iw = 1.0 / sqrt(E*G - F*F);
            const double coeff = const_c ? C(0,0,0) : C(qx,qy,e);
            const double alpha = wq * coeff * iw;
            D(qx,qy,0,e) =  alpha * G; // 1,1
            D(qx,qy,1,e) = -alpha * F; // 1,2
            D(qx,qy,2,e) =  alpha * E; // 2,2
         }
      }
   });
}

// PA Diffusion Assemble 3D kernel
void PADiffusionSetup3D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const Array<double> &w,
                        const Vector &j,
                        const Vector &c,
                        Vector &d)
{
   const bool symmetric = (coeffDim != 9);
   const bool const_c = c.Size() == 1;
   MFEM_VERIFY(coeffDim < 6 ||
               !const_c, "Constant matrix coefficient not supported");
   const auto W = Reshape(w.Read(), Q1D,Q1D,Q1D);
   const auto J = Reshape(j.Read(), Q1D,Q1D,Q1D,3,3,NE);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1,1,1) :
                  Reshape(c.Read(), coeffDim,Q1D,Q1D,Q1D,NE);
   auto D = Reshape(d.Write(), Q1D,Q1D,Q1D, symmetric ? 6 : 9, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               const double J11 = J(qx,qy,qz,0,0,e);
               const double J21 = J(qx,qy,qz,1,0,e);
               const double J31 = J(qx,qy,qz,2,0,e);
               const double J12 = J(qx,qy,qz,0,1,e);
               const double J22 = J(qx,qy,qz,1,1,e);
               const double J32 = J(qx,qy,qz,2,1,e);
               const double J13 = J(qx,qy,qz,0,2,e);
               const double J23 = J(qx,qy,qz,1,2,e);
               const double J33 = J(qx,qy,qz,2,2,e);
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
               const double w_detJ = W(qx,qy,qz) / detJ;
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

               if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
               {
                  // Compute entries of R = MJ^{-T} = M adj(J)^T, without det J.
                  const double M11 = C(0, qx,qy,qz, e);
                  const double M12 = C(1, qx,qy,qz, e);
                  const double M13 = C(2, qx,qy,qz, e);
                  const double M21 = (!symmetric) ? C(3, qx,qy,qz, e) : M12;
                  const double M22 = (!symmetric) ? C(4, qx,qy,qz, e) : C(3, qx,qy,qz, e);
                  const double M23 = (!symmetric) ? C(5, qx,qy,qz, e) : C(4, qx,qy,qz, e);
                  const double M31 = (!symmetric) ? C(6, qx,qy,qz, e) : M13;
                  const double M32 = (!symmetric) ? C(7, qx,qy,qz, e) : M23;
                  const double M33 = (!symmetric) ? C(8, qx,qy,qz, e) : C(5, qx,qy,qz, e);

                  const double R11 = M11*A11 + M12*A12 + M13*A13;
                  const double R12 = M11*A21 + M12*A22 + M13*A23;
                  const double R13 = M11*A31 + M12*A32 + M13*A33;
                  const double R21 = M21*A11 + M22*A12 + M23*A13;
                  const double R22 = M21*A21 + M22*A22 + M23*A23;
                  const double R23 = M21*A31 + M22*A32 + M23*A33;
                  const double R31 = M31*A11 + M32*A12 + M33*A13;
                  const double R32 = M31*A21 + M32*A22 + M33*A23;
                  const double R33 = M31*A31 + M32*A32 + M33*A33;

                  // Now set D to J^{-1} R = adj(J) R
                  D(qx,qy,qz,0,e) = w_detJ * (A11*R11 + A12*R21 + A13*R31); // 1,1
                  const double D12 = w_detJ * (A11*R12 + A12*R22 + A13*R32);
                  D(qx,qy,qz,1,e) = D12; // 1,2
                  D(qx,qy,qz,2,e) = w_detJ * (A11*R13 + A12*R23 + A13*R33); // 1,3

                  const double D21 = w_detJ * (A21*R11 + A22*R21 + A23*R31);
                  const double D22 = w_detJ * (A21*R12 + A22*R22 + A23*R32);
                  const double D23 = w_detJ * (A21*R13 + A22*R23 + A23*R33);

                  const double D33 = w_detJ * (A31*R13 + A32*R23 + A33*R33);

                  D(qx,qy,qz,3,e) = symmetric ? D22 : D21; // 2,2 or 2,1
                  D(qx,qy,qz,4,e) = symmetric ? D23 : D22; // 2,3 or 2,2
                  D(qx,qy,qz,5,e) = symmetric ? D33 : D23; // 3,3 or 2,3

                  if (!symmetric)
                  {
                     D(qx,qy,qz,6,e) = w_detJ * (A31*R11 + A32*R21 + A33*R31); // 3,1
                     D(qx,qy,qz,7,e) = w_detJ * (A31*R12 + A32*R22 + A33*R32); // 3,2
                     D(qx,qy,qz,8,e) = D33; // 3,3
                  }
               }
               else  // Vector or scalar coefficient version
               {
                  const double C1 = const_c ? C(0,0,0,0,0) : C(0,qx,qy,qz,e);
                  const double C2 = const_c ? C(0,0,0,0,0) :
                                    (coeffDim == 3 ? C(1,qx,qy,qz,e) : C(0,qx,qy,qz,e));
                  const double C3 = const_c ? C(0,0,0,0,0) :
                                    (coeffDim == 3 ? C(2,qx,qy,qz,e) : C(0,qx,qy,qz,e));

                  // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
                  D(qx,qy,qz,0,e) = w_detJ * (C1*A11*A11 + C2*A12*A12 + C3*A13*A13); // 1,1
                  D(qx,qy,qz,1,e) = w_detJ * (C1*A11*A21 + C2*A12*A22 + C3*A13*A23); // 2,1
                  D(qx,qy,qz,2,e) = w_detJ * (C1*A11*A31 + C2*A12*A32 + C3*A13*A33); // 3,1
                  D(qx,qy,qz,3,e) = w_detJ * (C1*A21*A21 + C2*A22*A22 + C3*A23*A23); // 2,2
                  D(qx,qy,qz,4,e) = w_detJ * (C1*A21*A31 + C2*A22*A32 + C3*A23*A33); // 3,2
                  D(qx,qy,qz,5,e) = w_detJ * (C1*A31*A31 + C2*A32*A32 + C3*A33*A33); // 3,3
               }
            }
         }
      }
   });
}

static void PADiffusionSetup(const int dim,
                             const int sdim,
                             const int D1D,
                             const int Q1D,
                             const int coeffDim,
                             const int NE,
                             const Array<double> &W,
                             const Vector &J,
                             const Vector &C,
                             Vector &D)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADiffusionSetup"); }
   if (dim == 2)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPADiffusionSetup2D(D1D, Q1D, NE, W, J, C, D);
         return;
      }
#else
      MFEM_CONTRACT_VAR(D1D);
#endif // MFEM_USE_OCCA
      if (sdim == 2) { PADiffusionSetup2D<2>(Q1D, coeffDim, NE, W, J, C, D); }
      if (sdim == 3) { PADiffusionSetup2D<3>(Q1D, coeffDim, NE, W, J, C, D); }
   }
   if (dim == 3)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPADiffusionSetup3D(D1D, Q1D, NE, W, J, C, D);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionSetup3D(Q1D, coeffDim, NE, W, J, C, D);
   }
}

void DiffusionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   if (DeviceCanUseCeed())
   {
      delete ceedDataPtr;
      ceedDataPtr = new CeedData;
      InitCeedCoeff(Q, *mesh, *ir, ceedDataPtr);
      return CeedPADiffusionAssemble(fes, *ir, *ceedDataPtr);
   }
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   const int sdim = mesh->SpaceDimension();
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   int coeffDim = 1;
   Vector coeff;
   const int MQfullDim = MQ ? MQ->GetHeight() * MQ->GetWidth() : 0;
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetHeight() == dim && MQ->GetWidth() == dim, "");
      const int MQsymmDim = MQ->GetWidth() * (MQ->GetWidth() + 1) / 2;

      const int MQdim = MQ->IsSymmetric() ? MQsymmDim : MQfullDim;
      coeffDim = MQdim;

      coeff.SetSize(MQdim * nq * ne);
      symmetric = MQ ? MQ->IsSymmetric() : true;

      DenseMatrix M;
      Vector Msymm;
      if (symmetric)
      {
         Msymm.SetSize(MQsymmDim);
      }
      else
      {
         M.SetSize(dim);
      }

      auto C = Reshape(coeff.HostWrite(), MQdim, nq, ne);
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            if (MQ->IsSymmetric())
            {
               MQ->EvalSymmetric(Msymm, *tr, ir->IntPoint(p));

               for (int i=0; i<MQsymmDim; ++i)
               {
                  C(i, p, e) = Msymm[i];
               }
            }
            else
            {
               MQ->Eval(M, *tr, ir->IntPoint(p));

               for (int i=0; i<dim; ++i)
                  for (int j=0; j<dim; ++j)
                  {
                     C(j+(i*dim), p, e) = M(i,j);
                  }
            }
         }
      }
   }
   else if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == dim, "");
      coeffDim = VQ->GetVDim();
      coeff.SetSize(coeffDim * nq * ne);
      auto C = Reshape(coeff.HostWrite(), coeffDim, nq, ne);
      Vector D(coeffDim);
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            VQ->Eval(D, *tr, ir->IntPoint(p));
            for (int i=0; i<coeffDim; ++i)
            {
               C(i, p, e) = D[i];
            }
         }
      }
   }
   else if (Q == nullptr)
   {
      coeff.SetSize(1);
      coeff(0) = 1.0;
   }
   else if (ConstantCoefficient* cQ = dynamic_cast<ConstantCoefficient*>(Q))
   {
      coeff.SetSize(1);
      coeff(0) = cQ->constant;
   }
   else if (QuadratureFunctionCoefficient* cQ =
               dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      const QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == ne*nq,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(ir == &qFun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");
      qFun.Read();
      coeff.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
   }
   else
   {
      coeff.SetSize(nq * ne);
      auto C = Reshape(coeff.HostWrite(), nq, ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C(q,e) = Q->Eval(T, ir->IntPoint(q));
         }
      }
   }
   pa_data.SetSize((symmetric ? symmDims : MQfullDim) * nq * ne,
                   Device::GetDeviceMemoryType());
   PADiffusionSetup(dim, sdim, dofs1D, quad1D, coeffDim, ne, ir->GetWeights(),
                    geom->J, coeff, pa_data);
}

template<int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionDiagonal2D(const int NE,
                                  const bool symmetric,
                                  const Array<double> &b,
                                  const Array<double> &g,
                                  const Vector &d,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   // note the different shape for D, if this is a symmetric matrix we only
   // store necessary entries
   auto D = Reshape(d.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      // gradphi \cdot Q \gradphi has four terms
      double QD0[MQ1][MD1];
      double QD1[MQ1][MD1];
      double QD2[MQ1][MD1];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            QD0[qx][dy] = 0.0;
            QD1[qx][dy] = 0.0;
            QD2[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int q = qx + qy * Q1D;
               const double D00 = D(q,0,e);
               const double D10 = D(q,1,e);
               const double D01 = symmetric ? D10 : D(q,2,e);
               const double D11 = symmetric ? D(q,2,e) : D(q,3,e);
               QD0[qx][dy] += B(qy, dy) * B(qy, dy) * D00;
               QD1[qx][dy] += B(qy, dy) * G(qy, dy) * (D01 + D10);
               QD2[qx][dy] += G(qy, dy) * G(qy, dy) * D11;
            }
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Y(dx,dy,e) += G(qx, dx) * G(qx, dx) * QD0[qx][dy];
               Y(dx,dy,e) += G(qx, dx) * B(qx, dx) * QD1[qx][dy];
               Y(dx,dy,e) += B(qx, dx) * B(qx, dx) * QD2[qx][dy];
            }
         }
      }
   });
}

// Shared memory PA Diffusion Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SmemPADiffusionDiagonal2D(const int NE,
                                      const bool symmetric,
                                      const Array<double> &b_,
                                      const Array<double> &g_,
                                      const Vector &d_,
                                      Vector &y_,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG+0);
      double (*G)[MD1] = (double (*)[MD1]) (BG+1);
      MFEM_SHARED double QD[3][NBZ][MD1][MQ1];
      double (*QD0)[MD1] = (double (*)[MD1])(QD[0] + tidz);
      double (*QD1)[MD1] = (double (*)[MD1])(QD[1] + tidz);
      double (*QD2)[MD1] = (double (*)[MD1])(QD[2] + tidz);
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            QD0[qx][dy] = 0.0;
            QD1[qx][dy] = 0.0;
            QD2[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int q = qx + qy * Q1D;
               const double D00 = D(q,0,e);
               const double D10 = D(q,1,e);
               const double D01 = symmetric ? D10 : D(q,2,e);
               const double D11 = symmetric ? D(q,2,e) : D(q,3,e);
               const double By = B[qy][dy];
               const double Gy = G[qy][dy];
               const double BB = By * By;
               const double BG = By * Gy;
               const double GG = Gy * Gy;
               QD0[qx][dy] += BB * D00;
               QD1[qx][dy] += BG * (D01 + D10);
               QD2[qx][dy] += GG * D11;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Bx = B[qx][dx];
               const double Gx = G[qx][dx];
               const double BB = Bx * Bx;
               const double BG = Bx * Gx;
               const double GG = Gx * Gx;
               Y(dx,dy,e) += GG * QD0[qx][dy];
               Y(dx,dy,e) += BG * QD1[qx][dy];
               Y(dx,dy,e) += BB * QD2[qx][dy];
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionDiagonal3D(const int NE,
                                  const bool symmetric,
                                  const Array<double> &b,
                                  const Array<double> &g,
                                  const Vector &d,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Q = Reshape(d.Read(), Q1D*Q1D*Q1D, symmetric ? 6 : 9, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      double QQD[MQ1][MQ1][MD1];
      double QDD[MQ1][MD1][MD1];
      for (int i = 0; i < DIM; ++i)
      {
         for (int j = 0; j < DIM; ++j)
         {
            // first tensor contraction, along z direction
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     QQD[qx][qy][dz] = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const int q = qx + (qy + qz * Q1D) * Q1D;
                        const int ksym = j >= i ?
                        3 - (3-i)*(2-i)/2 + j:
                        3 - (3-j)*(2-j)/2 + i;
                        const int k = symmetric ? ksym : (i*DIM) + j;
                        const double O = Q(q,k,e);
                        const double Bz = B(qz,dz);
                        const double Gz = G(qz,dz);
                        const double L = i==2 ? Gz : Bz;
                        const double R = j==2 ? Gz : Bz;
                        QQD[qx][qy][dz] += L * O * R;
                     }
                  }
               }
            }
            // second tensor contraction, along y direction
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dz = 0; dz < D1D; ++dz)
               {
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     QDD[qx][dy][dz] = 0.0;
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const double By = B(qy,dy);
                        const double Gy = G(qy,dy);
                        const double L = i==1 ? Gy : By;
                        const double R = j==1 ? Gy : By;
                        QDD[qx][dy][dz] += L * QQD[qx][qy][dz] * R;
                     }
                  }
               }
            }
            // third tensor contraction, along x direction
            for (int dz = 0; dz < D1D; ++dz)
            {
               for (int dy = 0; dy < D1D; ++dy)
               {
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const double Bx = B(qx,dx);
                        const double Gx = G(qx,dx);
                        const double L = i==0 ? Gx : Bx;
                        const double R = j==0 ? Gx : Bx;
                        Y(dx, dy, dz, e) += L * QDD[qx][dy][dz] * R;
                     }
                  }
               }
            }
         }
      }
   });
}

// Shared memory PA Diffusion Diagonal 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPADiffusionDiagonal3D(const int NE,
                                      const bool symmetric,
                                      const Array<double> &b_,
                                      const Array<double> &g_,
                                      const Vector &d_,
                                      Vector &y_,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D*Q1D, symmetric ? 6 : 9, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG+0);
      double (*G)[MD1] = (double (*)[MD1]) (BG+1);
      MFEM_SHARED double QQD[MQ1][MQ1][MD1];
      MFEM_SHARED double QDD[MQ1][MD1][MD1];
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int i = 0; i < DIM; ++i)
      {
         for (int j = 0; j < DIM; ++j)
         {
            // first tensor contraction, along z direction
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
                  MFEM_FOREACH_THREAD(dz,z,D1D)
                  {
                     QQD[qx][qy][dz] = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const int q = qx + (qy + qz * Q1D) * Q1D;
                        const int ksym = j >= i ?
                                         3 - (3-i)*(2-i)/2 + j:
                                         3 - (3-j)*(2-j)/2 + i;
                        const int k = symmetric ? ksym : (i*DIM) + j;
                        const double O = D(q,k,e);
                        const double Bz = B[qz][dz];
                        const double Gz = G[qz][dz];
                        const double L = i==2 ? Gz : Bz;
                        const double R = j==2 ? Gz : Bz;
                        QQD[qx][qy][dz] += L * O * R;
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // second tensor contraction, along y direction
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  MFEM_FOREACH_THREAD(dy,y,D1D)
                  {
                     QDD[qx][dy][dz] = 0.0;
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        const double By = B[qy][dy];
                        const double Gy = G[qy][dy];
                        const double L = i==1 ? Gy : By;
                        const double R = j==1 ? Gy : By;
                        QDD[qx][dy][dz] += L * QQD[qx][qy][dz] * R;
                     }
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // third tensor contraction, along x direction
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               MFEM_FOREACH_THREAD(dy,y,D1D)
               {
                  MFEM_FOREACH_THREAD(dx,x,D1D)
                  {
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const double Bx = B[qx][dx];
                        const double Gx = G[qx][dx];
                        const double L = i==0 ? Gx : Bx;
                        const double R = j==0 ? Gx : Bx;
                        Y(dx, dy, dz, e) += L * QDD[qx][dy][dz] * R;
                     }
                  }
               }
            }
         }
      }
   });
}

static void PADiffusionAssembleDiagonal(const int dim,
                                        const int D1D,
                                        const int Q1D,
                                        const int NE,
                                        const bool symm,
                                        const Array<double> &B,
                                        const Array<double> &G,
                                        const Vector &D,
                                        Vector &Y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPADiffusionDiagonal2D<2,2,8>(NE,symm,B,G,D,Y);
         case 0x33: return SmemPADiffusionDiagonal2D<3,3,8>(NE,symm,B,G,D,Y);
         case 0x44: return SmemPADiffusionDiagonal2D<4,4,4>(NE,symm,B,G,D,Y);
         case 0x55: return SmemPADiffusionDiagonal2D<5,5,4>(NE,symm,B,G,D,Y);
         case 0x66: return SmemPADiffusionDiagonal2D<6,6,2>(NE,symm,B,G,D,Y);
         case 0x77: return SmemPADiffusionDiagonal2D<7,7,2>(NE,symm,B,G,D,Y);
         case 0x88: return SmemPADiffusionDiagonal2D<8,8,1>(NE,symm,B,G,D,Y);
         case 0x99: return SmemPADiffusionDiagonal2D<9,9,1>(NE,symm,B,G,D,Y);
         default: return PADiffusionDiagonal2D(NE,symm,B,G,D,Y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPADiffusionDiagonal3D<2,3>(NE,symm,B,G,D,Y);
         case 0x34: return SmemPADiffusionDiagonal3D<3,4>(NE,symm,B,G,D,Y);
         case 0x45: return SmemPADiffusionDiagonal3D<4,5>(NE,symm,B,G,D,Y);
         case 0x56: return SmemPADiffusionDiagonal3D<5,6>(NE,symm,B,G,D,Y);
         case 0x67: return SmemPADiffusionDiagonal3D<6,7>(NE,symm,B,G,D,Y);
         case 0x78: return SmemPADiffusionDiagonal3D<7,8>(NE,symm,B,G,D,Y);
         case 0x89: return SmemPADiffusionDiagonal3D<8,9>(NE,symm,B,G,D,Y);
         case 0x9A: return SmemPADiffusionDiagonal3D<9,10>(NE,symm,B,G,D,Y);
         default: return PADiffusionDiagonal3D(NE,symm,B,G,D,Y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

void DiffusionIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      CeedAssembleDiagonal(ceedDataPtr, diag);
   }
   else
   {
      if (pa_data.Size()==0) { AssemblePA(*fespace); }
      PADiffusionAssembleDiagonal(dim, dofs1D, quad1D, ne, symmetric,
                                  maps->B, maps->G, pa_data, diag);
   }
}


#ifdef MFEM_USE_OCCA
// OCCA PA Diffusion Apply 2D kernel
static void OccaPADiffusionApply2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Array<double> &Bt,
                                   const Array<double> &Gt,
                                   const Vector &D,
                                   const Vector &X,
                                   Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_G = OccaMemoryRead(G.GetMemory(), G.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_Gt = OccaMemoryRead(Gt.GetMemory(), Gt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply2D_cpu;
      if (OccaDiffApply2D_cpu.find(id) == OccaDiffApply2D_cpu.end())
      {
         const occa::kernel DiffusionApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_CPU", props);
         OccaDiffApply2D_cpu.emplace(id, DiffusionApply2D_CPU);
      }
      OccaDiffApply2D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply2D_gpu;
      if (OccaDiffApply2D_gpu.find(id) == OccaDiffApply2D_gpu.end())
      {
         const occa::kernel DiffusionApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_GPU", props);
         OccaDiffApply2D_gpu.emplace(id, DiffusionApply2D_GPU);
      }
      OccaDiffApply2D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
}

// OCCA PA Diffusion Apply 3D kernel
static void OccaPADiffusionApply3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Array<double> &Bt,
                                   const Array<double> &Gt,
                                   const Vector &D,
                                   const Vector &X,
                                   Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_G = OccaMemoryRead(G.GetMemory(), G.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_Gt = OccaMemoryRead(Gt.GetMemory(), Gt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply3D_cpu;
      if (OccaDiffApply3D_cpu.find(id) == OccaDiffApply3D_cpu.end())
      {
         const occa::kernel DiffusionApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_CPU", props);
         OccaDiffApply3D_cpu.emplace(id, DiffusionApply3D_CPU);
      }
      OccaDiffApply3D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply3D_gpu;
      if (OccaDiffApply3D_gpu.find(id) == OccaDiffApply3D_gpu.end())
      {
         const occa::kernel DiffusionApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_GPU", props);
         OccaDiffApply3D_gpu.emplace(id, DiffusionApply3D_GPU);
      }
      OccaDiffApply3D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
}
#endif // MFEM_USE_OCCA

// PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionApply2D(const int NE,
                               const bool symmetric,
                               const Array<double> &b_,
                               const Array<double> &g_,
                               const Array<double> &bt_,
                               const Array<double> &gt_,
                               const Vector &d_,
                               const Vector &x_,
                               Vector &y_,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b_.Read(), Q1D, D1D);
   auto G = Reshape(g_.Read(), Q1D, D1D);
   auto Bt = Reshape(bt_.Read(), D1D, Q1D);
   auto Gt = Reshape(gt_.Read(), D1D, Q1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto X = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double grad[max_Q1D][max_Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[max_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = X(dx,dy,e);
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
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }
      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;

            const double O11 = D(q,0,e);
            const double O21 = D(q,1,e);
            const double O12 = symmetric ? O21 : D(q,2,e);
            const double O22 = symmetric ? D(q,2,e) : D(q,3,e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O21 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double gradX[max_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double wx  = Bt(dx,qx);
               const double wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy  = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               Y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// Shared memory PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SmemPADiffusionApply2D(const int NE,
                                   const bool symmetric,
                                   const Array<double> &b_,
                                   const Array<double> &g_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D, symmetric ? 3 : 4, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      MFEM_SHARED double GQ[2][NBZ][MD1][MQ1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + tidz);
      double (*DQ0)[MD1] = (double (*)[MD1])(GD[0] + tidz);
      double (*DQ1)[MD1] = (double (*)[MD1])(GD[1] + tidz);
      double (*QQ0)[MD1] = (double (*)[MD1])(GQ[0] + tidz);
      double (*QQ1)[MD1] = (double (*)[MD1])(GQ[1] + tidz);
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X[dy][dx] = x(dx,dy,e);
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][dy] = b(q,dy);
               G[q][dy] = g(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double coords = X[dy][dx];
               u += B[qx][dx] * coords;
               v += G[qx][dx] * coords;
            }
            DQ0[dy][qx] = u;
            DQ1[dy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               u += DQ1[dy][qx] * B[qy][dy];
               v += DQ0[dy][qx] * G[qy][dy];
            }
            QQ0[qy][qx] = u;
            QQ1[qy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int q = (qx + ((qy) * Q1D));
            const double O11 = D(q,0,e);
            const double O21 = D(q,1,e);
            const double O12 = symmetric ? O21 : D(q,2,e);
            const double O22 = symmetric ? D(q,2,e) : D(q,3,e);
            const double gX = QQ0[qy][qx];
            const double gY = QQ1[qy][qx];
            QQ0[qy][qx] = (O11 * gX) + (O12 * gY);
            QQ1[qy][qx] = (O21 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[dy][q] = b(q,dy);
               Gt[dy][q] = g(q,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += Gt[dx][qx] * QQ0[qy][qx];
               v += Bt[dx][qx] * QQ1[qy][qx];
            }
            DQ0[qy][dx] = u;
            DQ1[qy][dx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += DQ0[qy][dx] * Bt[dy][qy];
               v += DQ1[qy][dx] * Gt[dy][qy];
            }
            Y(dx,dy,e) += (u + v);
         }
      }
   });
}

// PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionApply3D(const int NE,
                               const bool symmetric,
                               const Array<double> &b,
                               const Array<double> &g,
                               const Array<double> &bt,
                               const Array<double> &gt,
                               const Vector &d_,
                               const Vector &x_,
                               Vector &y_,
                               int d1d = 0, int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);
   auto D = Reshape(d_.Read(), Q1D*Q1D*Q1D, symmetric ? 6 : 9, NE);
   auto X = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double grad[max_Q1D][max_Q1D][max_Q1D][3];
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
      for (int dz = 0; dz < D1D; ++dz)
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
         for (int dy = 0; dy < D1D; ++dy)
         {
            double gradX[max_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = X(dx,dy,dz,e);
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
      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + (qy + qz * Q1D) * Q1D;
               const double O11 = D(q,0,e);
               const double O12 = D(q,1,e);
               const double O13 = D(q,2,e);
               const double O21 = symmetric ? O12 : D(q,3,e);
               const double O22 = symmetric ? D(q,3,e) : D(q,4,e);
               const double O23 = symmetric ? D(q,4,e) : D(q,5,e);
               const double O31 = symmetric ? O13 : D(q,6,e);
               const double O32 = symmetric ? O23 : D(q,7,e);
               const double O33 = symmetric ? D(q,5,e) : D(q,8,e);
               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];
               grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[qz][qy][qx][1] = (O21*gradX)+(O22*gradY)+(O23*gradZ);
               grad[qz][qy][qx][2] = (O31*gradX)+(O32*gradY)+(O33*gradZ);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double gradXY[max_D1D][max_D1D][3];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double gradX[max_D1D][3];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double gX = grad[qz][qy][qx][0];
               const double gY = grad[qz][qy][qx][1];
               const double gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double wx  = Bt(dx,qx);
                  const double wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy  = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz  = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
}

// Half of B and G are stored in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D.
static MFEM_HOST_DEVICE inline int qi(const int q, const int d, const int Q)
{
   return (q<=d) ? q : Q-1-q;
}

static MFEM_HOST_DEVICE inline int dj(const int q, const int d, const int D)
{
   return (q<=d) ? d : D-1-d;
}

static MFEM_HOST_DEVICE inline int qk(const int q, const int d, const int Q)
{
   return (q<=d) ? Q-1-q : q;
}

static MFEM_HOST_DEVICE inline int dl(const int q, const int d, const int D)
{
   return (q<=d) ? D-1-d : d;
}

static MFEM_HOST_DEVICE inline double sign(const int q, const int d)
{
   return (q<=d) ? -1.0 : 1.0;
}

template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPADiffusionApply3D(const int NE,
                                   const bool symmetric,
                                   const Array<double> &b_,
                                   const Array<double> &g_,
                                   const Vector &d_,
                                   const Vector &x_,
                                   Vector &y_,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int M1Q = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int M1D = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= M1D, "");
   MFEM_VERIFY(Q1D <= M1Q, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto d = Reshape(d_.Read(), Q1D, Q1D, Q1D, symmetric ? 6 : 9, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      MFEM_SHARED double sBG[MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) sBG;
      double (*G)[MD1] = (double (*)[MD1]) sBG;
      double (*Bt)[MQ1] = (double (*)[MQ1]) sBG;
      double (*Gt)[MQ1] = (double (*)[MQ1]) sBG;
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
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int i = qi(qx,dy,Q1D);
            const int j = dj(qx,dy,D1D);
            const int k = qk(qx,dy,Q1D);
            const int l = dl(qx,dy,D1D);
            B[i][j] = b(qx,dy);
            G[k][l] = g(qx,dy) * sign(qx,dy);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[D1D], v[D1D];
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = 0.0; }
            MFEM_UNROLL(MD1)
            for (int dx = 0; dx < D1D; ++dx)
            {
               const int i = qi(qx,dx,Q1D);
               const int j = dj(qx,dx,D1D);
               const int k = qk(qx,dx,Q1D);
               const int l = dl(qx,dx,D1D);
               const double s = sign(qx,dx);
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const double coords = X[dz][dy][dx];
                  u[dz] += coords * B[i][j];
                  v[dz] += coords * G[k][l] * s;
               }
            }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               DDQ0[dz][dy][qx] = u[dz];
               DDQ1[dz][dy][qx] = v[dz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[D1D], v[D1D], w[D1D];
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = w[dz] = 0.0; }
            MFEM_UNROLL(MD1)
            for (int dy = 0; dy < D1D; ++dy)
            {
               const int i = qi(qy,dy,Q1D);
               const int j = dj(qy,dy,D1D);
               const int k = qk(qy,dy,Q1D);
               const int l = dl(qy,dy,D1D);
               const double s = sign(qy,dy);
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; dz++)
               {
                  u[dz] += DDQ1[dz][dy][qx] * B[i][j];
                  v[dz] += DDQ0[dz][dy][qx] * G[k][l] * s;
                  w[dz] += DDQ0[dz][dy][qx] * B[i][j];
               }
            }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; dz++)
            {
               DQQ0[dz][qy][qx] = u[dz];
               DQQ1[dz][qy][qx] = v[dz];
               DQQ2[dz][qy][qx] = w[dz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[Q1D], v[Q1D], w[Q1D];
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; qz++) { u[qz] = v[qz] = w[qz] = 0.0; }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; qz++)
               {
                  const int i = qi(qz,dz,Q1D);
                  const int j = dj(qz,dz,D1D);
                  const int k = qk(qz,dz,Q1D);
                  const int l = dl(qz,dz,D1D);
                  const double s = sign(qz,dz);
                  u[qz] += DQQ0[dz][qy][qx] * B[i][j];
                  v[qz] += DQQ1[dz][qy][qx] * B[i][j];
                  w[qz] += DQQ2[dz][qy][qx] * G[k][l] * s;
               }
            }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; qz++)
            {
               const double O11 = d(qx,qy,qz,0,e);
               const double O12 = d(qx,qy,qz,1,e);
               const double O13 = d(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : d(qx,qy,qz,3,e);
               const double O22 = symmetric ? d(qx,qy,qz,3,e) : d(qx,qy,qz,4,e);
               const double O23 = symmetric ? d(qx,qy,qz,4,e) : d(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : d(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : d(qx,qy,qz,7,e);
               const double O33 = symmetric ? d(qx,qy,qz,5,e) : d(qx,qy,qz,8,e);
               const double gX = u[qz];
               const double gY = v[qz];
               const double gZ = w[qz];
               QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
               QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ);
               QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(d,y,D1D)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            const int i = qi(q,d,Q1D);
            const int j = dj(q,d,D1D);
            const int k = qk(q,d,Q1D);
            const int l = dl(q,d,D1D);
            Bt[j][i] = b(q,d);
            Gt[l][k] = g(q,d) * sign(q,d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[Q1D], v[Q1D], w[Q1D];
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
            MFEM_UNROLL(MQ1)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int i = qi(qx,dx,Q1D);
               const int j = dj(qx,dx,D1D);
               const int k = qk(qx,dx,Q1D);
               const int l = dl(qx,dx,D1D);
               const double s = sign(qx,dx);
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u[qz] += QQQ0[qz][qy][qx] * Gt[l][k] * s;
                  v[qz] += QQQ1[qz][qy][qx] * Bt[j][i];
                  w[qz] += QQQ2[qz][qy][qx] * Bt[j][i];
               }
            }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               QQD0[qz][qy][dx] = u[qz];
               QQD1[qz][qy][dx] = v[qz];
               QQD2[qz][qy][dx] = w[qz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[Q1D], v[Q1D], w[Q1D];
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
            MFEM_UNROLL(MQ1)
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int i = qi(qy,dy,Q1D);
               const int j = dj(qy,dy,D1D);
               const int k = qk(qy,dy,Q1D);
               const int l = dl(qy,dy,D1D);
               const double s = sign(qy,dy);
               MFEM_UNROLL(MQ1)
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u[qz] += QQD0[qz][qy][dx] * Bt[j][i];
                  v[qz] += QQD1[qz][qy][dx] * Gt[l][k] * s;
                  w[qz] += QQD2[qz][qy][dx] * Bt[j][i];
               }
            }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               QDD0[qz][dy][dx] = u[qz];
               QDD1[qz][dy][dx] = v[qz];
               QDD2[qz][dy][dx] = w[qz];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[D1D], v[D1D], w[D1D];
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz) { u[dz] = v[dz] = w[dz] = 0.0; }
            MFEM_UNROLL(MQ1)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               MFEM_UNROLL(MD1)
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const int i = qi(qz,dz,Q1D);
                  const int j = dj(qz,dz,D1D);
                  const int k = qk(qz,dz,Q1D);
                  const int l = dl(qz,dz,D1D);
                  const double s = sign(qz,dz);
                  u[dz] += QDD0[qz][dy][dx] * Bt[j][i];
                  v[dz] += QDD1[qz][dy][dx] * Bt[j][i];
                  w[dz] += QDD2[qz][dy][dx] * Gt[l][k] * s;
               }
            }
            MFEM_UNROLL(MD1)
            for (int dz = 0; dz < D1D; ++dz)
            {
               y(dx,dy,dz,e) += (u[dz] + v[dz] + w[dz]);
            }
         }
      }
   });
}

static void PADiffusionApply(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const bool symm,
                             const Array<double> &B,
                             const Array<double> &G,
                             const Array<double> &Bt,
                             const Array<double> &Gt,
                             const Vector &D,
                             const Vector &X,
                             Vector &Y)
{
#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      if (dim == 2)
      {
         OccaPADiffusionApply2D(D1D,Q1D,NE,B,G,Bt,Gt,D,X,Y);
         return;
      }
      if (dim == 3)
      {
         OccaPADiffusionApply3D(D1D,Q1D,NE,B,G,Bt,Gt,D,X,Y);
         return;
      }
      MFEM_ABORT("OCCA PADiffusionApply unknown kernel!");
   }
#endif // MFEM_USE_OCCA
   const int ID = (D1D << 4) | Q1D;

   if (dim == 2)
   {
      switch (ID)
      {
         case 0x22: return SmemPADiffusionApply2D<2,2,16>(NE,symm,B,G,D,X,Y);
         case 0x33: return SmemPADiffusionApply2D<3,3,16>(NE,symm,B,G,D,X,Y);
         case 0x44: return SmemPADiffusionApply2D<4,4,8>(NE,symm,B,G,D,X,Y);
         case 0x55: return SmemPADiffusionApply2D<5,5,8>(NE,symm,B,G,D,X,Y);
         case 0x66: return SmemPADiffusionApply2D<6,6,4>(NE,symm,B,G,D,X,Y);
         case 0x77: return SmemPADiffusionApply2D<7,7,4>(NE,symm,B,G,D,X,Y);
         case 0x88: return SmemPADiffusionApply2D<8,8,2>(NE,symm,B,G,D,X,Y);
         case 0x99: return SmemPADiffusionApply2D<9,9,2>(NE,symm,B,G,D,X,Y);
         default:   return PADiffusionApply2D(NE,symm,B,G,Bt,Gt,D,X,Y,D1D,Q1D);
      }
   }

   if (dim == 3)
   {
      switch (ID)
      {
         case 0x23: return SmemPADiffusionApply3D<2,3>(NE,symm,B,G,D,X,Y);
         case 0x34: return SmemPADiffusionApply3D<3,4>(NE,symm,B,G,D,X,Y);
         case 0x45: return SmemPADiffusionApply3D<4,5>(NE,symm,B,G,D,X,Y);
         case 0x46: return SmemPADiffusionApply3D<4,6>(NE,symm,B,G,D,X,Y);
         case 0x56: return SmemPADiffusionApply3D<5,6>(NE,symm,B,G,D,X,Y);
         case 0x58: return SmemPADiffusionApply3D<5,8>(NE,symm,B,G,D,X,Y);
         case 0x67: return SmemPADiffusionApply3D<6,7>(NE,symm,B,G,D,X,Y);
         case 0x78: return SmemPADiffusionApply3D<7,8>(NE,symm,B,G,D,X,Y);
         case 0x89: return SmemPADiffusionApply3D<8,9>(NE,symm,B,G,D,X,Y);
         default:   return PADiffusionApply3D(NE,symm,B,G,Bt,Gt,D,X,Y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA Diffusion Apply kernel
void DiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      CeedAddMult(ceedDataPtr, x, y);
   }
   else
   {
      PADiffusionApply(dim, dofs1D, quad1D, ne, symmetric,
                       maps->B, maps->G, maps->Bt, maps->Gt,
                       pa_data, x, y);
   }
}

} // namespace mfem
