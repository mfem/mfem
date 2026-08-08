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
#pragma once

#include "../mode/batch.hpp"
#include "../../../../general/globals.hpp"
#include <cstring>
#include "fields.hpp"

/// \cond DO_NOT_DOCUMENT

namespace mfem::internal::mma::form
{

/** Shared smem / launch plan (specialized (DIM,D1D,QND) or runtime Fallback).
    Fields are filled by porting existing device NB helpers. */
struct SmemPlan
{
   int nb = 0;
   int x_ld = 0;
   int u_ld = 0;
   int n_u_planes = 1;   // U planes in smem (e.g. dim for Grad)
   bool load_x = true;
   bool use_q_tile = false;
   int tq = 0;           // Q-tile length when use_q_tile; else full nq
   int smem_bytes = 0;
   int nthreads = 0;
};

/** Family A — Eval / 1-plane forms (Value×Value or None×Value): no Q-tile.
    Uses MassLikeNB device helpers (shared NB budget for 1-plane smem). */
template <int DIM, int D1D, int QND>
inline SmemPlan MakeEvalPlan(bool load_x = true)
{
   using mma::MassLikeNB;
   using mma::PadLdBank;
   using mma::MmaMapFor;
   using mma::SimplexNdof;
   using mma::SimplexMaxNq;
   using mma::LaunchNthreads;

   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int NB = MassLikeNB<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr int U_LD = PadLdBank<MAP>(MQ);

   SmemPlan p{};
   p.nb = NB;
   p.x_ld = X_LD;
   p.u_ld = U_LD;
   p.n_u_planes = 1;
   p.load_x = load_x;
   p.use_q_tile = false;
   p.tq = MQ;
   // Specialized mass allocates X+U even when budgeting; LF specialized may
   // only touch U in launch but NB still comes from MassLikeNB.
   p.smem_bytes = int(sizeof(real_t)) *
                  (load_x ? (X_LD + U_LD) : U_LD) * NB;
   p.nthreads = LaunchNthreads<QND>(MQ, BASIS);
   return p;
}

inline SmemPlan MakeEvalPlanRuntime(int ndof, int nq, bool load_x = true)
{
   using mma::MassLikeNBRuntime;
   using mma::PadLdBankRuntime;
   using mma::LaunchNthreads;

   SmemPlan p{};
   p.nb = MassLikeNBRuntime(ndof, nq);
   p.x_ld = PadLdBankRuntime(ndof);
   p.u_ld = PadLdBankRuntime(nq);
   p.n_u_planes = 1;
   p.load_x = load_x;
   p.use_q_tile = false;
   p.tq = nq;
   p.smem_bytes = int(sizeof(real_t)) *
                  (load_x ? (p.x_ld + p.u_ld) : p.u_ld) * p.nb;
   p.nthreads = LaunchNthreads(nq, ndof);
   return p;
}

/** Family B — Grad×Grad: X + n_u_planes·U, optional Q-tile (DIM==3).
    Uses BatchNB* helpers in mma/mode/batch.hpp (multi-plane smem). */
template <int DIM, int D1D, int QND>
inline SmemPlan MakeGradPlan()
{
   using mma::BatchNB;
   using mma::BatchUseQTile;
   using mma::BatchQTileFor;
   using mma::PadLdBank;
   using mma::MmaMapFor;
   using mma::SimplexNdof;
   using mma::SimplexMaxNq;
   using mma::LaunchNthreads;

   constexpr int MAP = MmaMapFor<DIM, D1D, QND>();
   constexpr int BASIS = SimplexNdof<DIM, D1D>();
   constexpr int MQ = SimplexMaxNq<DIM, QND>();
   constexpr int NB = BatchNB<DIM, D1D, QND>();
   constexpr int X_LD = PadLdBank<MAP>(BASIS);
   constexpr bool qtile = BatchUseQTile<DIM, D1D, QND>();

   SmemPlan p{};
   p.nb = NB;
   p.x_ld = X_LD;
   p.n_u_planes = DIM;
   p.load_x = true;
   p.use_q_tile = qtile;
   if constexpr (qtile)
   {
      constexpr int TQ = BatchQTileFor<DIM, D1D, QND>();
      constexpr int U_LD = PadLdBank<MAP>(TQ);
      p.u_ld = U_LD;
      p.tq = TQ;
      p.smem_bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      p.nthreads = LaunchNthreads<TQ>(TQ, BASIS);
   }
   else
   {
      constexpr int U_LD = PadLdBank<MAP>(MQ);
      p.u_ld = U_LD;
      p.tq = MQ;
      p.smem_bytes = int(sizeof(real_t)) * (X_LD + DIM * U_LD) * NB;
      p.nthreads = LaunchNthreads<QND>(QND, BASIS);
   }
   return p;
}

/** Runtime Grad plan (full-NQ or Q-tile via Batch*Runtime helpers). */
inline SmemPlan MakeGradPlanRuntime(int dim, int ndof, int nq)
{
   using mma::BatchNBRuntime;
   using mma::BatchUseQTileRuntime;
   using mma::BatchQTileForRuntime;
   using mma::PadLdBankRuntime;
   using mma::LaunchNthreads;

   SmemPlan p{};
   p.nb = BatchNBRuntime(dim, ndof, nq, dim);
   p.x_ld = PadLdBankRuntime(ndof);
   p.n_u_planes = dim;
   p.load_x = true;
   p.use_q_tile = BatchUseQTileRuntime(dim, ndof, nq, dim);
   p.tq = p.use_q_tile ? BatchQTileForRuntime(ndof, nq) : nq;
   p.u_ld = PadLdBankRuntime(p.use_q_tile ? p.tq : nq);
   p.smem_bytes = int(sizeof(real_t)) * (p.x_ld + dim * p.u_ld) * p.nb;
   p.nthreads = LaunchNthreads(p.use_q_tile ? p.tq : nq, ndof);
   return p;
}

/** Plan from QFn traits: Eval family (1-plane) or Grad family. */
template <typename QFn, int DIM, int D1D, int QND>
SmemPlan MakeDevicePlan()
{
   using Tr = qfn_traits<QFn>;
   if constexpr (Tr::trial_is_grad || Tr::test_is_grad)
   {
      static_assert(Tr::load_x, "Grad forms require trial DOFs");
      return MakeGradPlan<DIM, D1D, QND>();
   }
   else
   {
      return MakeEvalPlan<DIM, D1D, QND>(Tr::load_x);
   }
}


// ---------------------------------------------------------------------------
// Opt-in form dump (MFEM_MMA_FORM_DUMP)
// ---------------------------------------------------------------------------

/** True when MFEM_MMA_FORM_DUMP is set and not "0". Cached after first query. */
inline bool FormDumpEnabled()
{
   static int env = -1; // -1 unset, 0 off, 1 on
   if (env < 0)
   {
      const char *e = GetEnv("MFEM_MMA_FORM_DUMP");
      env = (e && std::strcmp(e, "0") != 0) ? 1 : 0;
   }
   return env == 1;
}

inline const char *FieldKindName(field_kind k)
{
   switch (k)
   {
      case field_kind::Eval: return "Eval";
      case field_kind::Grad: return "Grad";
      case field_kind::None: return "None";
   }
   return "?";
}

inline void DumpPlanFields(const SmemPlan &p)
{
   mfem::out << "  plan: NB=" << p.nb
             << " X_LD=" << p.x_ld
             << " U_LD=" << p.u_ld
             << " n_u_planes=" << p.n_u_planes
             << " load_x=" << (p.load_x ? 1 : 0)
             << " qtile=" << (p.use_q_tile ? 1 : 0)
             << " TQ=" << p.tq
             << " smem=" << p.smem_bytes
             << " nthreads=" << p.nthreads
             << '\n';
}

/** Dump specialized Apply traits + plan. No-op if dump disabled. */
template <typename QFn, int DIM, int D1D, int QND>
inline void DumpFormApply(const char *entry, int NE, int nq, int ndof)
{
   if (!FormDumpEnabled()) { return; }

   using Tr = qfn_traits<QFn>;
   const bool device = Device::Allows(Backend::DEVICE_MASK);

   mfem::out << "[MMA form] " << entry
             << " QFn"
             << " NE=" << NE
             << " DIM=" << DIM << " D1D=" << D1D << " QND=" << QND
             << " nq=" << nq << " ndof=" << ndof
             << '\n';
   mfem::out << "  trial=" << FieldKindName(
                field_traits<typename Tr::trial_kind>::kind)
             << " test=" << FieldKindName(field_traits<typename Tr::test_kind>::kind)
             << " load_x=" << (Tr::load_x ? 1 : 0)
             << " planes=" << Tr::u_planes(DIM)
             << '\n';
   mfem::out << "  path=" << (device ? "device" : "host") << '\n';

   if constexpr (Tr::trial_is_grad || Tr::test_is_grad)
   {
      DumpPlanFields(MakeGradPlan<DIM, D1D, QND>());
   }
   else if constexpr (Tr::load_x)
   {
      DumpPlanFields(MakeEvalPlan<DIM, D1D, QND>(true));
   }
   else
   {
      DumpPlanFields(MakeEvalPlan<DIM, D1D, QND>(false));
   }
}

/** Dump runtime Fallback Apply. */
template <typename QFn, int DIM>
inline void DumpFormApplyRuntime(const char *entry, int NE, int nq, int ndof)
{
   if (!FormDumpEnabled()) { return; }

   using Tr = qfn_traits<QFn>;
   const bool device = Device::Allows(Backend::DEVICE_MASK);

   mfem::out << "[MMA form] " << entry
             << " QFn (runtime)"
             << " NE=" << NE
             << " DIM=" << DIM
             << " nq=" << nq << " ndof=" << ndof
             << '\n';
   mfem::out << "  trial=" << FieldKindName(
                field_traits<typename Tr::trial_kind>::kind)
             << " test=" << FieldKindName(field_traits<typename Tr::test_kind>::kind)
             << " load_x=" << (Tr::load_x ? 1 : 0)
             << " planes=" << Tr::u_planes(DIM)
             << '\n';
   mfem::out << "  path=" << (device ? "device" : "host") << '\n';

   if constexpr (Tr::load_x && !Tr::trial_is_grad)
   {
      DumpPlanFields(MakeEvalPlanRuntime(ndof, nq, true));
   }
   else if constexpr (!Tr::load_x)
   {
      DumpPlanFields(MakeEvalPlanRuntime(ndof, nq, false));
   }
   else
   {
      DumpPlanFields(MakeGradPlanRuntime(DIM, ndof, nq));
   }
}

} // namespace mfem::internal::mma::form

/// \endcond
