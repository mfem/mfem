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

#ifndef MFEM_LOR_VAR_ORDER
#define MFEM_LOR_VAR_ORDER

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../pbilinearform.hpp"
#include "../../linalg/hypre.hpp"

#include <memory>

namespace mfem
{

/// Optional timing and sparsity breakdown for direct-sparse variable-order LOR.
struct VarOrderLORTimings
{
   double aux_space_s = 0.0;
   double lor_A_s = 0.0;
   double lor_aux_ops_s = 0.0;
   double hypre_init_s = 0.0;
   HYPRE_BigInt lor_nnz = 0;
   HYPRE_BigInt g_nnz = 0;
   HYPRE_BigInt c_nnz = 0;
   HYPRE_BigInt pi_nnz = 0;
};

/// @brief Direct-sparse O(p^3) parallel low-order-refined (LOR) assembly for
/// variable-order and/or nonconforming 3D hexahedral de Rham spaces
/// (H1 GaussLobatto, ND GaussLobatto/IntegratedGLL, RT GaussLobatto/IntegratedGLL).
class VarOrderLORAssembly
{
public:
   /// Return true if @a a_ho can be assembled via direct-sparse variable-order LOR.
   static bool FormIsSupported(BilinearForm &a_ho);

   /// Assemble the parallel true-DOF LOR operator A_lor = P^T A_loc P with
   /// essential boundary elimination (DIAG_ONE) directly in O(sum_K p_K^3) work.
   static void Assemble(ParBilinearForm &a_ho,
                        const Array<int> &ess_tdof_list,
                        OperatorHandle &A,
                        VarOrderLORTimings *timings = nullptr);

   /// Assemble exact 2-nnz/row discrete gradient (dir = -1) or piecewise-linear
   /// Nedelec interpolation Pi_d^ND (dir in {0, 1, 2}): R_ND * Op_loc * P_H1.
   static std::unique_ptr<HypreParMatrix> AssembleGradientOrPi(
      ParFiniteElementSpace &h1_fes,
      ParFiniteElementSpace &nd_fes,
      int dir);

   /// Assemble exact 4-nnz/row discrete curl: R_RT * C_loc * P_ND.
   static std::unique_ptr<HypreParMatrix> AssembleCurl(
      ParFiniteElementSpace &nd_fes,
      ParFiniteElementSpace &rt_fes);

   /// Assemble <=4-nnz/row Raviart-Thomas interpolation Pi_d^RT (dir in {0,1,2})
   /// using 2x2 Gauss-Lobatto corner surface quadrature: R_RT * Pi_loc * P_H1.
   static std::unique_ptr<HypreParMatrix> AssembleRTPi(
      ParFiniteElementSpace &h1_fes,
      ParFiniteElementSpace &rt_fes,
      int dir);

   /// Build a HypreAMS preconditioner using direct-sparse variable-order LOR
   /// assembly (storing the assembled LOR system matrix in @a A).
   static std::unique_ptr<HypreAMS> BuildAMS(
      ParBilinearForm &a_ho,
      const Array<int> &ess_tdof_list,
      OperatorHandle &A,
      VarOrderLORTimings *timings = nullptr);

   /// Build a HypreADS preconditioner using direct-sparse variable-order LOR
   /// assembly (storing the assembled LOR system matrix in @a A).
   static std::unique_ptr<HypreADS> BuildADS(
      ParBilinearForm &a_ho,
      const Array<int> &ess_tdof_list,
      OperatorHandle &A,
      VarOrderLORTimings *timings = nullptr);
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LOR_VAR_ORDER
