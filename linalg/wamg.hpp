// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file

#ifndef MFEM_WAMG_HPP
#define MFEM_WAMG_HPP

#include "../fem/fem.hpp"
#include "wavelets.hpp"

namespace mfem
{

////////////////////////////////////////////////////////////////////////////////
struct wargs_t
{
   const OperatorHandle &op_h;
   Vector &diag;
   Array<int> &ess_tdof_list;
   int smoother_order, max_iter, print_level;

   wargs_t(const OperatorHandle &op,
           Vector &diag,
           Array<int> &ess_tdof_list,
           int smoother_order,
           int max_iter,
           int print_level):
      op_h(op),
      diag(diag),
      ess_tdof_list(ess_tdof_list),
      smoother_order(smoother_order),
      max_iter(max_iter),
      print_level(print_level) {}
};

////////////////////////////////////////////////////////////////////////////////
struct WaveletLevel
{
   virtual OperatorHandle OpHandle() = 0;
   virtual Operator *Prolongator() = 0;
};

////////////////////////////////////////////////////////////////////////////////
/// Wavelet Recursive level associated with an PA operator
struct WaveletRecursiveLevel: public WaveletLevel
{
   Operator *W, *Wt, *WAWt;
   WaveletRecursiveLevel(Wavelet::Type &wavelet, const bool &lowpass,
                         const Operator &A);
   ~WaveletRecursiveLevel();
   OperatorHandle OpHandle() override { return OperatorHandle(WAWt, false); }
   Operator *Prolongator() override { return Wt; }
};

////////////////////////////////////////////////////////////////////////////////
/// Wavelet Recursive level associated with a FULL operator
struct WaveletRecursiveLevelFA: public WaveletLevel
{
   Wavelet *W;
   Operator *Wt;
   SparseMatrix *M, *tM;
   HypreParMatrix *MAMt;
   WaveletRecursiveLevelFA(ParFiniteElementSpace &pfes,
                           Wavelet::Type &wavelet,
                           HypreParMatrix *Op_h);
   ~WaveletRecursiveLevelFA();
   OperatorHandle OpHandle() override;
   Operator *Prolongator() override;
};

////////////////////////////////////////////////////////////////////////////////
class WAMGRSolver : public Multigrid
{
public:
   WAMGRSolver(ParFiniteElementSpace &pfes,
               Wavelet::Type wavelet,
               const bool lowpass,
               int max_depth,
               int max_ndofs,
               wargs_t args,
               const bool to_bottom,
               const bool to_full);
};

////////////////////////////////////////////////////////////////////////////////
/// @brief WAMG solver
/// wavelet == HAAR | DAUBECHIES & lowpass
class WAMG : public Solver
{
   const int max_depth = 32;
   const int max_ndofs = 1024*1024;
   const bool lowpass = true;
   const bool to_bottom = false;
   const bool to_full = false;
   WAMGRSolver wavelet_solver;
public:
   WAMG(ParFiniteElementSpace &pfes, Wavelet::Type wavelet, wargs_t args);
   void Mult(const Vector&, Vector&) const override;
   void SetOperator(const Operator&) override { assert(false); }
   void AssembleDiagonal(Vector&) const override { assert(false); }
};

////////////////////////////////////////////////////////////////////////////////
/// @brief faWAMG solver
/// wavelet == HAAR | DAUBECHIES & lowpass
class faWAMG : public Solver
{
   const int max_depth = 32;
   const int max_ndofs = 16*1024;
   const bool lowpass = true;
   const bool to_bottom = false;
   const bool to_full = true;
   WAMGRSolver wavelet_solver;
public:
   faWAMG(ParFiniteElementSpace &pfes, Wavelet::Type wavelet, wargs_t args);
   void Mult(const Vector&, Vector&) const override;
   void SetOperator(const Operator&) override { assert(false); }
   void AssembleDiagonal(Vector&) const override { assert(false); }
};


} // mfem namespace

#endif // MFEM_WAMG_HPP
