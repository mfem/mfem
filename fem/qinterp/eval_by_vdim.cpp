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

#include "../quadinterpolator.hpp"
#include "dispatch.hpp"
#include "eval.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Tensor-product evaluation of quadrature point values: dispatch function.
// Instantiation for the case QVectorLayout::byVDIM.
template<>
void TensorValues<QVectorLayout::byVDIM>(const int NE,
                                         const int vdim,
                                         const DofToQuad &maps,
                                         const Vector &e_vec,
                                         Vector &q_val)
{
   if (NE == 0) { return; }
   const int dim = maps.FE->GetDim();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *B = maps.B.Read();
   const double *X = e_vec.Read();
   double *Y = q_val.Write();

   constexpr QVectorLayout L = QVectorLayout::byVDIM;

   const int id = (vdim<<8) | (D1D<<4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x124: return Values2D<L,1,2,4,8>(NE,B,X,Y);
         case 0x136: return Values2D<L,1,3,6,4>(NE,B,X,Y);
         case 0x148: return Values2D<L,1,4,8,2>(NE,B,X,Y);

         case 0x224: return Values2D<L,2,2,4,8>(NE,B,X,Y);
         case 0x234: return Values2D<L,2,3,4,8>(NE,B,X,Y);
         case 0x236: return Values2D<L,2,3,6,4>(NE,B,X,Y);
         case 0x248: return Values2D<L,2,4,8,2>(NE,B,X,Y);

         default:
         {
            constexpr int MD = MAX_D1D;
            constexpr int MQ = MAX_Q1D;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                        << MQ << " 1D points are not supported!");
            Values2D<L,0,0,0,0,MD,MQ>(NE,B,X,Y,vdim,D1D,Q1D);
            return;
         }
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         case 0x124: return Values3D<L,1,2,4>(NE,B,X,Y);
         case 0x136: return Values3D<L,1,3,6>(NE,B,X,Y);
         case 0x148: return Values3D<L,1,4,8>(NE,B,X,Y);

         case 0x324: return Values3D<L,3,2,4>(NE,B,X,Y);
         case 0x336: return Values3D<L,3,3,6>(NE,B,X,Y);
         case 0x348: return Values3D<L,3,4,8>(NE,B,X,Y);

         default:
         {
            constexpr int MD = 8;
            constexpr int MQ = 8;
            MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                        << " are not supported!");
            MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                        << MQ << " 1D points are not supported!");
            Values3D<L,0,0,0,MD,MQ>(NE,B,X,Y,vdim,D1D,Q1D);
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
