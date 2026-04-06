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

#include "mfem.hpp"

namespace mfem
{

// A BlockDiagonalPreconditioner which assumes that all the blocks are symmetric
// Convenient to use with Multigrid Class in which a MultTranspose is needed
class SymmetricBlockDiagonalPreconditioner : public BlockDiagonalPreconditioner
{
private:
   real_t c;
public:
   /// @brief Constructs a symmetric block-diagonal preconditioner with the given block offsets
   ///and scaling factor.
   /// @param offsets The offsets of the blocks in the block-diagonal preconditioner.
   /// @param c_ The scaling factor to be applied to the result.
   SymmetricBlockDiagonalPreconditioner(const Array<int> & offsets,
                                        real_t c_ = 1.0)
      : BlockDiagonalPreconditioner(offsets), c(c_) { }

   void Mult(const Vector & x, Vector & y) const override
   {
      BlockDiagonalPreconditioner::Mult(x,y);
      y*=c;
   }

   void MultTranspose (const Vector & x, Vector & y) const override
   {
      this->Mult(x,y);
   }
};

#ifdef MFEM_USE_MPI

/// @brief  Creates a default solver for a given parallel finite element space.
/// The default solvers are the folowing:
/// - For H1 and L2 spaces: HypreBoomerAMG
/// - For 3D RT spaces: HypreADS
/// - For 2D RT and ND spaces: HypreAMS
/// @param pfespace The parallel finite element space for which the solver is to be created.
/// @param print_level The printing level for the solver.
/// @return a pointer to the created solver.
Solver * MakeFESpaceDefaultSolver(
   const ParFiniteElementSpace * pfespace, int print_level);


/// Shared helper class (real/complex) p-refinement multigrid:
/// - builds FE hierarchy
/// - builds transfer operators
///
class PRefinementHierarchy
{
public:
   Array<int> orders;
   const Array<ParFiniteElementSpace*> &pfes;
   std::vector<Array<int>> ess_bdr_marker;
   std::vector<Array<int>> ess_tdof_list;
   ParMesh *pmesh = nullptr;
   int nblocks;
   int maxlevels = 1;

   // Owned levels: 0..maxlevels-2
   std::vector<std::vector<std::unique_ptr<FiniteElementCollection>>> fec_owned;
   std::vector<std::vector<std::unique_ptr<ParFiniteElementSpace>>>   fes_owned;

   // Transfer operators per level and block (owned here)
   std::vector<std::vector<std::unique_ptr<PRefinementTransferOperator>>> T_level;

   PRefinementHierarchy(const Array<ParFiniteElementSpace*> &pfes_,
                        const std::vector<Array<int>> & ess_bdr_marker_);

   const ParFiniteElementSpace* GetParFESpace(int lev, int b) const;

   int GetFESpaceMinimumOrder(const ParFiniteElementSpace *pfespace) const;

   /** @brief Computes orders/maxlevels and constructs fec/fes hierarchy 
       and T_level storage. */
   void BuildSpaceHierarchy(int mgmaxlevels = -1);

   /** @brief Builds block-diagonal prolongation for level lev (coarse=lev, fine=lev+1).
       Its diagonal blocks are HypreParMatrix*
       returned by the transfer operators stored in T_level[lev][b]. */
   BlockOperator *BuildProlongation(int lev);
};

/// @brief  Creates a p-refinement multigrid preconditioner for a given set of parallel
/// finite element space and block operator.
class PRefinementMultigrid : public Multigrid
{
private:
   PRefinementHierarchy hierarchy;
   const BlockOperator &Op;

   std::unique_ptr<Solver> coarse_prec;

public:
   PRefinementMultigrid(const Array<ParFiniteElementSpace*> &pfes_,
                        const std::vector<Array<int>> & ess_bdr_marker_,
                        const BlockOperator &Op_, int mgmaxlevels = -1,
                        real_t smoother_relax_factor = 2.0/3,
                        bool mumps_coarse_solver = false);

   ~PRefinementMultigrid() override = default;
};

/// @brief  Creates a p-refinement multigrid preconditioner for a given set of parallel
/// finite element space and complex operator.
class ComplexPRefinementMultigrid : public Multigrid
{
private:
   // NOTE: nblocks for the hierarchy is derived from Op.real() at construction time,
   // so we store hierarchy behind a pointer to avoid a "dummy nblocks" constructor.
   std::unique_ptr<PRefinementHierarchy> hierarchy;

   const ComplexOperator &Op;
   std::unique_ptr<Solver> coarse_prec;

public:
   ComplexPRefinementMultigrid(const Array<ParFiniteElementSpace*> &pfes_,
                               const std::vector<Array<int>> & ess_bdr_marker,
                               const ComplexOperator &Op_, int mgmaxlevels = -1,
                               real_t smoother_relax_factor = 2.0/3,
                               bool mumps_coarse_solver = false);

   ~ComplexPRefinementMultigrid() override = default;
};

#endif

// Applies a given real preconditioner to the real and imaginary parts of a complex vector
class ComplexPreconditioner : public Solver
{
private:
   const Operator *op = nullptr;
   const Solver * prec = nullptr;
   bool own_prec = false;

public:
   ComplexPreconditioner(const Solver * real_prec, bool own = false)
      : Solver(2*real_prec->Height()), prec(real_prec), own_prec(own) { }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      int n = x.Size()/2;
      MFEM_VERIFY(x.Size() == 2*n, "Invalid x vector size");
      MFEM_VERIFY(y.Size() == 2*n, "Invalid y vector size");

      Vector x_r(const_cast<Vector&>(x), 0, n);
      Vector x_i(const_cast<Vector&>(x), n, n);
      Vector y_r(y, 0, n);
      Vector y_i(y, n, n);

      // Apply the preconditioner to the real and imaginary parts separately
      prec->Mult(x_r, y_r);
      prec->Mult(x_i, y_i);
   }

   virtual void MultTranspose(const Vector &x, Vector &y) const override
   {
      int n = x.Size()/2;
      MFEM_VERIFY(x.Size() == 2*n, "Invalid x vector size");
      MFEM_VERIFY(y.Size() == 2*n, "Invalid y vector size");

      Vector x_r(const_cast<Vector&>(x), 0, n);
      Vector x_i(const_cast<Vector&>(x), n, n);
      Vector y_r(y, 0, n);
      Vector y_i(y, n, n);

      // Apply the preconditioner to the real and imaginary parts separately
      prec->MultTranspose(x_r, y_r);
      prec->MultTranspose(x_i, y_i);
   }

   void SetOperator(const Operator &op_) override
   {
      MFEM_VERIFY(dynamic_cast<const ComplexOperator*>(&op_),
                  "ComplexPreconditioner::SetOperator only accepts ComplexOperator");
      this->op = &op_;
   }

   ~ComplexPreconditioner()
   {
      if (own_prec) { delete prec; }
   }
};

} // namespace mfem
