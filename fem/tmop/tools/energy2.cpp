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

#include "../pa.hpp"
#include "../../gridfunc.hpp"
#include "energy2.hpp"

namespace mfem
{

void TMOP_Integrator::GetLocalStateEnergyPA_2D(const Vector &X,
                                               real_t &energy) const
{
   const bool use_detA = false;
   const int mid = metric->Id();

   Vector L(PA.E.Size(), Device::GetMemoryType()); L.UseDevice(true);

   TMOPEnergyPA2D ker(this, X, L, use_detA);

   if (mid == 1) { tmop::Kernel<1>(ker); }
   else if (mid == 2) { tmop::Kernel<2>(ker); }
   else if (mid == 7) { tmop::Kernel<7>(ker); }
   else if (mid == 56) { tmop::Kernel<56>(ker); }
   else if (mid == 77) { tmop::Kernel<77>(ker); }
   else if (mid == 80) { tmop::Kernel<80>(ker); }
   else if (mid == 94) { tmop::Kernel<94>(ker); }
   else { MFEM_ABORT("Unsupported TMOP metric " << mid); }

   real_t lim_energy;
   ker.GetEnergy(energy, lim_energy);
}

void TMOP_Integrator::GetLocalNormalizationEnergiesPA_2D(const Vector &X,
                                                         real_t &met_energy,
                                                         real_t &lim_energy) const
{
   const bool use_detA = false;
   const int mid = metric->Id();

   Vector L(PA.E.Size(), Device::GetMemoryType()); L.UseDevice(true);

   const real_t mn = 1.0;
   Vector mc(1); mc = 1.0;

   TMOPEnergyPA2D ker(this, X, L, mn, mc, use_detA);

   if (mid == 1) { tmop::Kernel<1>(ker); }
   else if (mid == 2) { tmop::Kernel<2>(ker); }
   else if (mid == 7) { tmop::Kernel<7>(ker); }
   else if (mid == 56) { tmop::Kernel<56>(ker); }
   else if (mid == 77) { tmop::Kernel<77>(ker); }
   else if (mid == 80) { tmop::Kernel<80>(ker); }
   else if (mid == 94) { tmop::Kernel<94>(ker); }
   else { MFEM_ABORT("Unsupported TMOP metric " << mid); }

   ker.GetEnergy(met_energy, lim_energy);
}

void TMOP_Combo_QualityMetric::GetLocalEnergyPA_2D(const GridFunction &nodes,
                                                   const TargetConstructor &tc,
                                                   int m_index,
                                                   real_t &energy, real_t &vol,
                                                   const IntegrationRule &ir) const
{
   auto fes = nodes.FESpace();

   const int N = fes->GetNE();
   const auto metric = tmop_q_arr[m_index];
   const int mid = metric->Id();

   auto fe = fes->GetTypicalFE();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   auto maps = fe->GetDofToQuad(ir, mode);
   const int d = maps.ndof;
   const int q = maps.nqpt;

   const real_t metric_normal = 1.0;
   Vector metric_coeff(1); metric_coeff = 1.0;

   const Array<real_t> &B = maps.B, &G = maps.G;

   Vector E(N * ir.GetNPoints(), Device::GetDeviceMemoryType());
   Vector O(N * ir.GetNPoints(), Device::GetDeviceMemoryType()); O = 1.0;
   Vector L(N * ir.GetNPoints(), Device::GetDeviceMemoryType());

   auto R = fes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   Vector X(R->Height());
   R->Mult(nodes, X);

   DenseTensor Jtr(2, 2, N * ir.GetNPoints(), Device::GetDeviceMemoryType());
   tc.ComputeAllElementTargets(*fes, ir, X, Jtr);

   TMOPEnergyPA2D ker(X, E, L, O, true, d, q,
                      metric_normal, N, metric, B, G, Jtr, ir, metric_coeff);

   if (mid == 1) { tmop::Kernel<1>(ker); }
   else if (mid == 2) { tmop::Kernel<2>(ker); }
   else if (mid == 7) { tmop::Kernel<7>(ker); }
   else if (mid == 56) { tmop::Kernel<56>(ker); }
   else if (mid == 77) { tmop::Kernel<77>(ker); }
   else if (mid == 80) { tmop::Kernel<80>(ker); }
   else if (mid == 94) { tmop::Kernel<94>(ker); }
   else { MFEM_ABORT("Unsupported TMOP metric " << mid); }

   ker.GetEnergy(energy, vol);
}

} // namespace mfem
