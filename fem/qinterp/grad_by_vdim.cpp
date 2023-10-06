// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "dispatch.hpp"
#include "grad.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Tensor-product evaluation of quadrature point derivatives: dispatch function.
// Instantiation for the case QVectorLayout::byVDIM.
template<>
void TensorDerivatives<QVectorLayout::byVDIM>(const int NE,
                                              const int vdim,
                                              const DofToQuad &maps,
                                              const Vector &e_vec,
                                              Vector &q_der)
{
   if (NE == 0) { return; }
   const int dim = maps.FE->GetDim();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const double *J = nullptr; // not used in DERIVATIVES (non-GRAD_PHYS) mode
   const double *X = e_vec.Read();
   double *Y = q_der.Write();

   constexpr QVectorLayout L = QVectorLayout::byVDIM;
   constexpr bool P = false; // GRAD_PHYS

   const int id = (vdim<<8) | (D1D<<4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x134: return Derivatives2D<L,P,1,3,4,8>(NE,B,G,J,X,Y);
         case 0x146: return Derivatives2D<L,P,1,4,6,4>(NE,B,G,J,X,Y);
         case 0x158: return Derivatives2D<L,P,1,5,8,2>(NE,B,G,J,X,Y);

         case 0x234: return Derivatives2D<L,P,2,3,4,8>(NE,B,G,J,X,Y);
         case 0x246: return Derivatives2D<L,P,2,4,6,4>(NE,B,G,J,X,Y);
         case 0x258: return Derivatives2D<L,P,2,5,8,2>(NE,B,G,J,X,Y);
         default:
         {
            constexpr int MD = MAX_D1D;
            constexpr int MQ = MAX_Q1D;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                        << MQ << " 1D points are not supported!");
            Derivatives2D<L,P,0,0,0,0,MD,MQ>(NE,B,G,J,X,Y,vdim,D1D,Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x134: return Derivatives3D<L,P,1,3,4>(NE,B,G,J,X,Y);
         case 0x146: return Derivatives3D<L,P,1,4,6>(NE,B,G,J,X,Y);
         case 0x158: return Derivatives3D<L,P,1,5,8>(NE,B,G,J,X,Y);

         case 0x334: return Derivatives3D<L,P,3,3,4>(NE,B,G,J,X,Y);
         case 0x346: return Derivatives3D<L,P,3,4,6>(NE,B,G,J,X,Y);
         case 0x358: return Derivatives3D<L,P,3,5,8>(NE,B,G,J,X,Y);
         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                        << MQ << " 1D points are not supported!");
            Derivatives3D<L,P,0,0,0,MD,MQ>(NE,B,G,J,X,Y,vdim,D1D,Q1D);
            return;
         }
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Kernel not supported yet");
}

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem
