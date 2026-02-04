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
// Convinient to use with Multigrid Class in which a MultTranspose is needed
class SymmetricBlockDiagonalPreconditioner : public BlockDiagonalPreconditioner
{
public:
   SymmetricBlockDiagonalPreconditioner(const Array<int> & offsets)
      : BlockDiagonalPreconditioner(offsets) { }
   void MultTranspose (const Vector & x, Vector & y) const override
   {
      this->Mult(x,y);
   }
};

#ifdef MFEM_USE_MPI

Solver * MakeFESpaceDefaultSolver(
   const ParFiniteElementSpace * pfespace, int print_level);


class PRefinementMultigrid : public Multigrid
{
private:
   // P_level : (level -> level+1), block-diagonal (one transfer per pfes block)
   Array<int> orders;
   const Array<ParFiniteElementSpace *> & pfes;
   ParMesh * pmesh=nullptr;
   int npfes;
   const BlockOperator & Op;
   int nblocks;
   int maxlevels = 1;

   // hierarchy of spaces
   // Owned levels: 0..maxlevels-2
   std::vector< std::vector<std::unique_ptr<FiniteElementCollection>> > fec_owned;
   std::vector< std::vector<std::unique_ptr<ParFiniteElementSpace>> >   fes_owned;
   std::vector< std::vector<PRefinementTransferOperator*> > T_level;
   // prolongation matrices per level and block (owned here)
   std::vector< std::vector<HypreParMatrix*> >  Pmat_level;

   const ParFiniteElementSpace* GetParFESpace(int lev, int b) const
   {
      if (lev == maxlevels-1) { return pfes[b]; }
      return fes_owned[lev][b].get();
   }

   const FiniteElementCollection* GetFEColl(int lev, int b) const
   {
      if (lev == maxlevels-1) { return pfes[b]->FEColl(); }
      return fec_owned[lev][b].get();
   }

   int GetMinimumOrder(const ParFiniteElementSpace * pfespace) const
   {
      return (dynamic_cast<const L2_FECollection*>(pfespace->FEColl()) ||
              dynamic_cast<const RT_FECollection*>(pfespace->FEColl())) ? 0 : 1;
   }

public:
   PRefinementMultigrid(const Array<ParFiniteElementSpace *> & pfes_,
                        const BlockOperator & Op_);

   ~PRefinementMultigrid();
};

class ComplexPRefinementMultigrid : public Multigrid
{
private:
   // P_level : (level -> level+1), block-diagonal (one transfer per pfes block)
   Array<int> orders;
   const Array<ParFiniteElementSpace *> & pfes;
   ParMesh * pmesh=nullptr;
   int npfes;
   const ComplexOperator & Op;
   int nblocks;
   int maxlevels = 1;

   // hierarchy of spaces
   // Owned levels: 0..maxlevels-2
   std::vector< std::vector<std::unique_ptr<FiniteElementCollection>> > fec_owned;
   std::vector< std::vector<std::unique_ptr<ParFiniteElementSpace>> >   fes_owned;
   std::vector< std::vector<PRefinementTransferOperator*> > T_level;
   // prolongation matrices per level and block (owned here)
   std::vector< std::vector<HypreParMatrix*> >  Pmat_level;

   const ParFiniteElementSpace* GetParFESpace(int lev, int b) const
   {
      if (lev == maxlevels-1) { return pfes[b]; }
      return fes_owned[lev][b].get();
   }

   const FiniteElementCollection* GetFEColl(int lev, int b) const
   {
      if (lev == maxlevels-1) { return pfes[b]->FEColl(); }
      return fec_owned[lev][b].get();
   }

   int GetMinimumOrder(const ParFiniteElementSpace * pfespace) const
   {
      return (dynamic_cast<const L2_FECollection*>(pfespace->FEColl()) ||
              dynamic_cast<const RT_FECollection*>(pfespace->FEColl())) ? 0 : 1;
   }

public:
   ComplexPRefinementMultigrid(const Array<ParFiniteElementSpace *> & pfes_,
                               const ComplexOperator & Op_);

   ~ComplexPRefinementMultigrid();
};


#endif

// Applies a given real preconditioner to the real and imaginary parts of a complex vector
class ComplexPreconditioner : public Solver
{
private:
   const Operator *op;
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
      this->op = &op;
   }

   ~ComplexPreconditioner()
   {
      if (own_prec) { delete prec; }
   }
};

} // namespace mfem
