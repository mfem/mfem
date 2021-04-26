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

#include "quadinterpolator_dispatch.hpp"
#include "quadinterpolator_grad.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Tensor-product evaluation of quadrature point derivatives: dispatch function.
// Instantiation for the case QVectorLayout::byNODES.
template<>
void TensorDerivatives<QVectorLayout::byNODES>(const int NE,
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

   constexpr QVectorLayout L = QVectorLayout::byNODES;
   constexpr bool P = false; // GRAD_PHYS

   const int id = (vdim<<8) | (D1D<<4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x133: return Derivatives2D<L,P,1,3,3,16>(NE,B,G,J,X,Y);
         case 0x134: return Derivatives2D<L,P,1,3,4,16>(NE,B,G,J,X,Y);
         case 0x143: return Derivatives2D<L,P,1,4,3,16>(NE,B,G,J,X,Y);
         case 0x144: return Derivatives2D<L,P,1,4,4,16>(NE,B,G,J,X,Y);

         case 0x222: return Derivatives2D<L,P,2,2,2,16>(NE,B,G,J,X,Y);
         case 0x223: return Derivatives2D<L,P,2,2,3,8>(NE,B,G,J,X,Y);
         case 0x224: return Derivatives2D<L,P,2,2,4,4>(NE,B,G,J,X,Y);
         case 0x225: return Derivatives2D<L,P,2,2,5,4>(NE,B,G,J,X,Y);
         case 0x226: return Derivatives2D<L,P,2,2,6,2>(NE,B,G,J,X,Y);

         case 0x233: return Derivatives2D<L,P,2,3,3,2>(NE,B,G,J,X,Y);
         case 0x234: return Derivatives2D<L,P,2,3,4,4>(NE,B,G,J,X,Y);
         case 0x243: return Derivatives2D<L,P,2,4,3,4>(NE,B,G,J,X,Y);
         case 0x236: return Derivatives2D<L,P,2,3,6,2>(NE,B,G,J,X,Y);

         case 0x244: return Derivatives2D<L,P,2,4,4,2>(NE,B,G,J,X,Y);
         case 0x245: return Derivatives2D<L,P,2,4,5,2>(NE,B,G,J,X,Y);
         case 0x246: return Derivatives2D<L,P,2,4,6,2>(NE,B,G,J,X,Y);
         case 0x247: return Derivatives2D<L,P,2,4,7,2>(NE,B,G,J,X,Y);

         case 0x256: return Derivatives2D<L,P,2,5,6,2>(NE,B,G,J,X,Y);
         default:
         {
            constexpr int MD = MAX_D1D;
            constexpr int MQ = MAX_Q1D;
            if (D1D > MD || Q1D > MQ)
            {
               MFEM_ABORT("");
            }
            Derivatives2D<L,P,0,0,0,0,MD,MQ>(NE,B,G,J,X,Y,vdim,D1D,Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x124: return Derivatives3D<L,P,1,2,4>(NE,B,G,J,X,Y);
         case 0x133: return Derivatives3D<L,P,1,3,3>(NE,B,G,J,X,Y);
         case 0x134: return Derivatives3D<L,P,1,3,4>(NE,B,G,J,X,Y);
         case 0x136: return Derivatives3D<L,P,1,3,6>(NE,B,G,J,X,Y);
         case 0x144: return Derivatives3D<L,P,1,4,4>(NE,B,G,J,X,Y);
         case 0x148: return Derivatives3D<L,P,1,4,8>(NE,B,G,J,X,Y);

         case 0x323: return Derivatives3D<L,P,3,2,3>(NE,B,G,J,X,Y);
         case 0x324: return Derivatives3D<L,P,3,2,4>(NE,B,G,J,X,Y);
         case 0x325: return Derivatives3D<L,P,3,2,5>(NE,B,G,J,X,Y);
         case 0x326: return Derivatives3D<L,P,3,2,6>(NE,B,G,J,X,Y);

         case 0x333: return Derivatives3D<L,P,3,3,3>(NE,B,G,J,X,Y);
         case 0x334: return Derivatives3D<L,P,3,3,4>(NE,B,G,J,X,Y);
         case 0x335: return Derivatives3D<L,P,3,3,5>(NE,B,G,J,X,Y);
         case 0x336: return Derivatives3D<L,P,3,3,6>(NE,B,G,J,X,Y);
         case 0x344: return Derivatives3D<L,P,3,4,4>(NE,B,G,J,X,Y);
         case 0x346: return Derivatives3D<L,P,3,4,6>(NE,B,G,J,X,Y);
         case 0x347: return Derivatives3D<L,P,3,4,7>(NE,B,G,J,X,Y);
         case 0x348: return Derivatives3D<L,P,3,4,8>(NE,B,G,J,X,Y);
         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            if (D1D > MD || Q1D > MQ)
            {
               MFEM_ABORT("");
            }
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
