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

namespace mfem
{

real_t ComputeAvgMetricsPA(Array<TMOP_QualityMetric *> tmop_q_arr,
                           const GridFunction &nodes, const TargetConstructor &tc,
                           Vector &averages, const IntegrationRule *IntRule)
{
   assert(false);
   const int m_cnt = tmop_q_arr.Size(),
             NE    = nodes.FESpace()->GetNE(),
             dim   = nodes.FESpace()->GetMesh()->Dimension();

   averages.SetSize(m_cnt);

   // Integrals of all metrics.
   Array<int> pos_dofs;
   averages = 0.0;
   real_t volume = 0.0;

   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe_pos = *nodes.FESpace()->GetFE(e);
      const IntegrationRule &ir = (IntRule) ? *IntRule
                                  /* */     : IntRules.Get(fe_pos.GetGeomType(),
                                                           2*fe_pos.GetOrder());
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
            averages(m) += tmop_q_arr[m]->EvalW(T) * w_detA;
         }
         volume += w_detA;
      }
   }
   return volume;
}

} // namespace mfem