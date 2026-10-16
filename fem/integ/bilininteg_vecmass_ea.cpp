// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

/// Assemble the element matrices of the vector mass operator for tensor
/// product elements, coupling all the vector components. The element dofs are
/// ordered as in the E-vector, i.e. the local index of scalar dof i of
/// component c is i + c*nd. The quadrature data @a padata is ordered as in
/// VectorMassIntegrator::AssemblePA: for a matrix coefficient Q, the entry
/// a*VDIM + b holds Q(a,b), which couples the row component a to the column
/// component b.
template <int DIM>
static void EAVectorMassAssemble(const int NE, const int coeff_vdim,
                                 const Array<real_t> &basis,
                                 const Vector &padata, Vector &eadata,
                                 const bool add, const int D1D, const int Q1D)
{
   static constexpr int VDIM = DIM;
   static constexpr int MAX_COEFF_VDIM = VDIM*VDIM;
   const int ND = (DIM == 2) ? D1D*D1D : D1D*D1D*D1D;
   const int NQ = (DIM == 2) ? Q1D*Q1D : Q1D*Q1D*Q1D;
   const bool const_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == VDIM;
   MFEM_VERIFY(const_coeff || vector_coeff || coeff_vdim == MAX_COEFF_VDIM,
               "Invalid coefficient dimension.");
   MFEM_VERIFY(eadata.Size() == VDIM*ND*VDIM*ND*NE, "Invalid E-matrix size.");

   const auto B = Reshape(basis.Read(), Q1D, D1D);
   const auto D = Reshape(padata.Read(), NQ, coeff_vdim, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(),
                    VDIM*ND, VDIM*ND, NE);
   // Limit the thread block size; the thread loops below are strided.
   const int nThreads = ND < 16 ? ND : 16;
   mfem::forall_2D(NE, nThreads, nThreads, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(J, y, ND)
      {
         MFEM_FOREACH_THREAD(I, x, ND)
         {
            // Lexicographic indices of the dofs
            const int i1 = I % D1D, i2 = (I / D1D) % D1D, i3 = I / (D1D*D1D);
            const int j1 = J % D1D, j2 = (J / D1D) % D1D, j3 = J / (D1D*D1D);
            real_t val[MAX_COEFF_VDIM];
            for (int c = 0; c < coeff_vdim; c++) { val[c] = 0.0; }
            for (int q = 0; q < NQ; q++)
            {
               const int k1 = q % Q1D, k2 = (q / Q1D) % Q1D, k3 = q / (Q1D*Q1D);
               real_t phi = B(k1, i1)*B(k1, j1)*B(k2, i2)*B(k2, j2);
               if (DIM == 3) { phi *= B(k3, i3)*B(k3, j3); }
               for (int c = 0; c < coeff_vdim; c++) { val[c] += phi*D(q, c, e); }
            }
            // Entry in row (I,a) and column (J,b), stored at M(J + b*ND,
            // I + a*ND, e), following the convention of the EA operator.
            for (int a = 0; a < VDIM; a++)
            {
               for (int b = 0; b < VDIM; b++)
               {
                  real_t v;
                  if (const_coeff) { v = (a == b) ? val[0] : 0.0; }
                  else if (vector_coeff) { v = (a == b) ? val[a] : 0.0; }
                  else { v = val[a*VDIM + b]; }
                  if (add) { M(J + b*ND, I + a*ND, e) += v; }
                  else { M(J + b*ND, I + a*ND, e) = v; }
               }
            }
         }
      }
   });
}

void VectorMassIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                      Vector &emat,
                                      const bool add)
{
   MFEM_VERIFY(!DeviceCanUseCeed(),
               "VectorMassIntegrator::AssembleEA is not supported with libCEED.");
   AssemblePA(fes);
   if (ne == 0) { return; }
   MFEM_VERIFY(maps->mode == DofToQuad::TENSOR,
               "VectorMassIntegrator::AssembleEA requires tensor product "
               "elements.");
   if (dim == 2)
   {
      EAVectorMassAssemble<2>(ne, coeff_vdim, maps->B, pa_data, emat, add,
                              dofs1D, quad1D);
   }
   else if (dim == 3)
   {
      EAVectorMassAssemble<3>(ne, coeff_vdim, maps->B, pa_data, emat, add,
                              dofs1D, quad1D);
   }
   else { MFEM_ABORT("Dimension not supported."); }
}

} // namespace mfem
