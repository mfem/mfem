// Copyright (c) 2019, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/forall.hpp"
#include "bilinteg.hpp"
#include "gridfunc.hpp"

using namespace std;

namespace mfem
{

// PA Vector Divergence Integrator

#ifdef MFEM_USE_OCCA
// OCCA 2D Assemble kernel
static void OccaPAVectorDivergenceSetup2D(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const Array<double> &W,
                                          const Vector &J,
                                          const double COEFF,
                                          Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaVecDivSetup2D_ker;
   if (OccaVecDivSetup2D_ker.find(id) == OccaVecDivSetup2D_ker.end())
   {
      const occa::kernel VectorDivergenceSetup2D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "VectorDivergenceSetup2D", props);
      OccaVecDivSetup2D_ker.emplace(id, VectorDivergenceSetup2D);
   }
   OccaVecDivSetup2D_ker.at(id)(NE, o_W, o_J, COEFF, o_op);
}

// OCCA 3D Assemble kernel
static void OccaPAVectorDivergenceSetup3D(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const Array<double> &W,
                                          const Vector &J,
                                          const double COEFF,
                                          Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaVecDivSetup3D_ker;
   if (OccaVecDivSetup3D_ker.find(id) == OccaVecDivSetup3D_ker.end())
   {
      const occa::kernel VectorDivergenceSetup3D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "VectorDivergenceSetup3D", props);
      OccaVecDivSetup3D_ker.emplace(id, VectorDivergenceSetup3D);
   }
   OccaVecDivSetup3D_ker.at(id)(NE, o_W, o_J, COEFF, o_op);
}
#endif // MFEM_USE_OCCA

// PA Vector Divergence Assemble 2D kernel
static void PAVectorDivergenceSetup2D(const int Q1D,
                                      const int NE,
                                      const Array<double> &w,
                                      const Vector &j,
                                      const double COEFF,
                                      Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto y = Reshape(op.Write(), NQ, 3, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double c_detJ = W[q] * COEFF / ((J11*J22)-(J21*J12));
         y(q,0,e) =  c_detJ * (J12*J12 + J22*J22); // 1,1
         y(q,1,e) = -c_detJ * (J12*J11 + J22*J21); // 1,2
         y(q,2,e) =  c_detJ * (J11*J11 + J21*J21); // 2,2
      }
   });
}

// PA Vector Divergence Assemble 3D kernel
static void PAVectorDivergenceSetup3D(const int Q1D,
                                      const int NE,
                                      const Array<double> &w,
                                      const Vector &j,
                                      const double COEFF,
                                      Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto y = Reshape(op.Write(), NQ, 6, NE);
   MFEM_FORALL(e, NE,
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
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         const double c_detJ = W[q] * COEFF / detJ;
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
         // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
         y(q,0,e) = c_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
         y(q,1,e) = c_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
         y(q,2,e) = c_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
         y(q,3,e) = c_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
         y(q,4,e) = c_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
         y(q,5,e) = c_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
      }
   });
}

static void PAVectorDivergenceSetup(const int dim,
                                    const int D1D,
                                    const int Q1D,
                                    const int NE,
                                    const Array<double> &W,
                                    const Vector &J,
                                    const double COEFF,
                                    Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PAVectorDivergenceSetup"); }
   if (dim == 2)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPAVectorDivergenceSetup2D(D1D, Q1D, NE, W, J, COEFF, op);
         return;
      }
#endif // MFEM_USE_OCCA
      PAVectorDivergenceSetup2D(Q1D, NE, W, J, COEFF, op);
   }
   if (dim == 3)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPAVectorDivergenceSetup3D(D1D, Q1D, NE, W, J, COEFF, op);
         return;
      }
#endif // MFEM_USE_OCCA
      PAVectorDivergenceSetup3D(Q1D, NE, W, J, COEFF, op);
   }
}

void VectorDivergenceIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());
   ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(Q);
   MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
   const double coeff = cQ->constant;
   PAVectorDivergenceSetup(dim, dofs1D, quad1D, ne, ir->GetWeights(), geom->J,
                           coeff, pa_data);
}

} // namespace mfem
