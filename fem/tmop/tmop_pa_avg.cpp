// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <cassert>

#include "../tmop.hpp"
#include "../fem/gridfunc.hpp"

#undef NVTX_COLOR
#define NVTX_COLOR ::gpu::nvtx::kOrchid
#include "general/nvtx.hpp"

namespace mfem
{

///////////////////////////////////////////////////////////////////
/*real_t GetLocalVolume_2D(const GridFunction &nodes)
{

   const int N = PA.ne;
   const int M = metric->Id();
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const real_t mn = metric_normal;
   const Vector &MC = PA.MC;
   const DenseTensor &J = PA.Jtr;
   const Array<real_t> &W = PA.ir->GetWeights();
   const Array<real_t> &B = PA.maps->B;
   const Array<real_t> &G = PA.maps->G;
   const Vector &O = PA.O;
   Vector &E = PA.E;

   Array<real_t> mp;
   if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(metric))
   {
      m->GetWeights(mp);
   }
}*/

void ComputeAvgMetricsPA(Array<TMOP_QualityMetric *> tmop_q_arr,
                         const GridFunction &nodes,
                         const TargetConstructor &tc,
                         const IntegrationRule *ir,
                         const Vector &averages_fa,
                         const real_t &volume_fa)
{

   const int m_cnt = tmop_q_arr.Size();

   dbg("m_cnt:{}", m_cnt);
   real_t volume_pa = 0.0;
   Vector averages_pa(m_cnt);
   averages_pa = 0.0;


   const MemoryType mt = Device::GetDeviceMemoryType();
   const FiniteElementSpace &nodal_fes = *nodes.FESpace();
   // const FiniteElement &fe0 = *fes.GetFE(0);
   Mesh &mesh = *nodal_fes.GetMesh();
   const int dim = mesh.Dimension(),
             ne = mesh.GetNE(),
             nq = ir->GetNPoints();

   dbg("dim:{} ne:{} nq:{}", dim, ne, nq);

   const auto lex = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *R = nodal_fes.GetElementRestriction(lex);

   Vector xe(R->Height(), mt);
   xe.UseDevice(true);
   dbg("xe.Size():{} R Height:{}", xe.Size(), R->Height());
   MFEM_VERIFY(xe.Size() == R->Height(), "Incorrect size of xe");
   R->Mult(nodes, xe);

   DenseTensor Jtr;
   Jtr.SetSize(dim, dim, ne * nq, mt);

   tc.ComputeAllElementTargets(nodal_fes, *ir, xe, Jtr); // not on device in 2D

   /*
      const int m_cnt = tmop_q_arr.Size(),
                NE    = nodes.FESpace()->GetNE(),
                dim   = nodes.FESpace()->GetMesh()->Dimension();

      dbg("m_cnt:{}", m_cnt);
      Vector averages_pa(m_cnt);

      // Integrals of all metrics.
      Array<int> pos_dofs;
      averages_pa = 0.0;
      real_t volume_pa = 0.0;

      for (int e = 0; e < NE; e++)
      {
         const FiniteElement &fe_pos = *nodes.FESpace()->GetFE(e);
         const IntegrationRule &ir = (IntRule) ? *IntRule
               : IntRules.Get(fe_pos.GetGeomType(),  2*fe_pos.GetOrder());
         const int nsp = ir.GetNPoints(), dof = fe_pos.GetDof();

         DenseMatrix dshape(dof, dim);
         DenseMatrix pos(dof, dim);
         pos.SetSize(dof, dim);
         Vector posV(pos.Data(), dof * dim);

         nodes.FESpace()->GetElementVDofs(e, pos_dofs);
         nodes.GetSubVector(pos_dofs, posV);

         DenseTensor W(dim, dim, nsp);
         DenseMatrix Winv(dim), T(dim), A(dim);
         tc.ComputeElementTargets(e, fe_pos, ir, posV, W);

         for (int q = 0; q < nsp; q++)
         {
            const DenseMatrix &Wj = W(q);
            CalcInverse(Wj, Winv);

            const IntegrationPoint &ip = ir.IntPoint(q);
            fe_pos.CalcDShape(ip, dshape);
            MultAtB(pos, dshape, A);
            Mult(A, Winv, T);

            const real_t w_detA = ip.weight * A.Det();
            for (int m = 0; m < m_cnt; m++)
            {
               tmop_q_arr[m]->SetTargetJacobian(Wj);
               averages_pa(m) += tmop_q_arr[m]->EvalW(T) * w_detA;
            }
            volume_pa += w_detA;
         }
      }*/
   MFEM_VERIFY(fabs(volume_pa - volume_fa) < 1e-12,
               "Volume of the PA metric is not equal to the volume of the "
               "FA metric. Volume PA: " << volume_pa << ", Volume FA: "
               << volume_fa);
   for (int m = 0; m < m_cnt; m++)
   {
      MFEM_VERIFY(fabs(averages_pa(m) - averages_fa(m)) < 1e-12,
                  "Average of the PA metric is not equal to the average of the "
                  "FA metric. Average PA: " << averages_pa(m) << ", Average "
                  "FA: " << averages_fa(m));
   }
}

} // namespace mfem