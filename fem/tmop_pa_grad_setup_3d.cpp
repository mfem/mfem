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

#include "tmop.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#define MFEM_DBG_COLOR 212
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SetupGradPA_3D(const Vector &xe_,
                           const int NE,
                           const Array<double> &w_,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseMatrix &j_,
                           Vector &p_,
                           const int d1d = 0,
                           const int q1d = 0)
{

}

// *****************************************************************************
void TMOP_Integrator::AssembleGradPA_3D(const DenseMatrix &Jtr,
                                        const Vector &Xe) const
{
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B1d = maps->B;
   const Array<double> &G1d = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   switch (id)
   {
      case 0x21: { SetupGradPA_3D<2,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x22: { SetupGradPA_3D<2,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x23: { SetupGradPA_3D<2,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x24: { SetupGradPA_3D<2,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x25: { SetupGradPA_3D<2,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x26: { SetupGradPA_3D<2,6,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }

      case 0x31: { SetupGradPA_3D<3,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x32: { SetupGradPA_3D<3,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x33: { SetupGradPA_3D<3,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x34: { SetupGradPA_3D<3,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x35: { SetupGradPA_3D<3,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x36: { SetupGradPA_3D<3,6,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }

      case 0x41: { SetupGradPA_3D<4,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x42: { SetupGradPA_3D<4,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x43: { SetupGradPA_3D<4,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x44: { SetupGradPA_3D<4,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x45: { SetupGradPA_3D<4,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x46: { SetupGradPA_3D<4,6,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }

      case 0x51: { SetupGradPA_3D<5,1,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x52: { SetupGradPA_3D<5,2,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x53: { SetupGradPA_3D<5,3,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x54: { SetupGradPA_3D<5,4,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x55: { SetupGradPA_3D<5,5,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      case 0x56: { SetupGradPA_3D<5,6,1>(Xe,ne,W,B1d,G1d,Jtr,dPpa); break; }
      default:
      {
         dbg("kernel id: %x", id);
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

} // namespace mfem
