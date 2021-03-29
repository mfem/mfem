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

#include "quadinterpolator.hpp"
#include "quadinterpolator_grad.hpp"

namespace mfem
{

template<>
void QuadratureInterpolator::Derivatives<QVectorLayout::byVDIM>(
   const Vector &e_vec, Vector &q_der) const
{
   const int NE = fespace->GetNE();
   if (NE == 0) { return; }

   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const double *X = e_vec.Read();
   double *Y = q_der.Write();

   constexpr QVectorLayout L = QVectorLayout::byVDIM;

   const int id = (dim<<12) | (vdim<<8) | (D1D<<4) | Q1D;

   switch (id)
   {
      case 0x2134: return Grad2D<L,1,3,4,8>(NE,B,G,X,Y);
      case 0x2146: return Grad2D<L,1,4,6,4>(NE,B,G,X,Y);
      case 0x2158: return Grad2D<L,1,5,8,2>(NE,B,G,X,Y);

      case 0x2234: return Grad2D<L,2,3,4,8>(NE,B,G,X,Y);
      case 0x2246: return Grad2D<L,2,4,6,4>(NE,B,G,X,Y);
      case 0x2258: return Grad2D<L,2,5,8,2>(NE,B,G,X,Y);

      case 0x3134: return Grad3D<L,1,3,4>(NE,B,G,X,Y);
      case 0x3146: return Grad3D<L,1,4,6>(NE,B,G,X,Y);
      case 0x3158: return Grad3D<L,1,5,8>(NE,B,G,X,Y);

      case 0x3334: return Grad3D<L,3,3,4>(NE,B,G,X,Y);
      case 0x3346: return Grad3D<L,3,4,6>(NE,B,G,X,Y);
      case 0x3358: return Grad3D<L,3,5,8>(NE,B,G,X,Y);
      default:
      {
         constexpr int MD1 = 8;
         constexpr int MQ1 = 8;
         MFEM_VERIFY(D1D <= MD1, "Orders higher than " << MD1-1
                     << " are not supported!");
         MFEM_VERIFY(Q1D <= MQ1, "Quadrature rules with more than "
                     << MQ1 << " 1D points are not supported!");
         if (dim == 2) { Grad2D<L,0,0,0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D); }
         if (dim == 3) { Grad3D<L,0,0,0,MD1,MQ1>(NE,B,G,X,Y,vdim,D1D,Q1D); }
         return;
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Kernel not supported yet");
}

} // namespace mfem
