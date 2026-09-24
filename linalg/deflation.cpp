// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. See file CONTRIBUTING.md for details.

#include "deflation.hpp"
#include "../general/device.hpp"
#include "../general/globals.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef MFEM_USE_MPI
#include "../general/communication.hpp"
#endif

namespace mfem
{

namespace
{

bool ValidTolerance(real_t value, bool allow_auto = false)
{
   return std::isfinite(value) && (value >= 0 || (allow_auto && value == -1));
}

void Like(Vector &target, const Vector &source)
{
   target.UseDevice(target.UseDevice() || source.UseDevice());
   target.SetSize(source.Size());
}

bool RangesOverlap(const real_t *first, int first_size,
                   const real_t *second, int second_size)
{
   const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
   const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
   const std::uintptr_t first_end = first_begin + first_size * sizeof(real_t);
   const std::uintptr_t second_end =
      second_begin + second_size * sizeof(real_t);
   return first_begin < second_end && second_begin < first_end;
}

/* Whether an input may share storage with an output that the caller is about
   to overwrite completely. The output is moved, without copying its data, to
   a memory space where the input is already valid, so the comparison never
   stages a fine vector. Only an input valid nowhere is conservatively treated
   as aliased. */
bool MayAlias(const Vector &input, Vector &output)
{
   if (&input == &output) { return true; }
   if (!input.Size() || !output.Size()) { return false; }
   const auto &in = input.GetMemory(), &out = output.GetMemory();
   bool on_device = false;
   if (in.HostIsValid() && (out.HostIsValid() || !in.DeviceIsValid()))
   {
      on_device = false;
   }
   else if (in.DeviceIsValid()) { on_device = true; }
   else { return true; }
   return RangesOverlap(input.Read(on_device), input.Size(),
                        output.Write(on_device), output.Size());
}

/// Whether two outputs, both about to be overwritten, share storage.
bool OutputsOverlap(Vector &first, Vector &second)
{
   if (&first == &second) { return true; }
   if (!first.Size() || !second.Size()) { return false; }
   const bool on_device = first.UseDevice() || second.UseDevice();
   return RangesOverlap(first.Write(on_device), first.Size(),
                        second.Write(on_device), second.Size());
}

real_t AutomaticTolerance(real_t requested, real_t floor, int columns)
{
   if (requested >= 0) { return requested; }
   return std::max(floor, real_t(32) * std::max(1, columns) *
                   std::numeric_limits<real_t>::epsilon());
}

} // namespace

void VectorDeflationBasis::Add(const Vector &v)
{
   MFEM_VERIFY(v.Size() == height, "Deflation vector has wrong local size");
   vectors.emplace_back(v);
   width = static_cast<int>(vectors.size());
}

void VectorDeflationBasis::Add(Vector &&v)
{
   MFEM_VERIFY(v.Size() == height, "Deflation vector has wrong local size");
   vectors.emplace_back(std::move(v));
   width = static_cast<int>(vectors.size());
}

void VectorDeflationBasis::Clear()
{
   vectors.clear();
   width = 0;
}

const Vector &VectorDeflationBasis::GetVector(int j) const
{
   MFEM_VERIFY(j >= 0 && j < width, "Deflation vector index out of range");
   return vectors[j];
}

void VectorDeflationBasis::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(x.Size() == width, "Invalid deflation basis coordinate");
   y.UseDevice(true);
   y.SetSize(height);
   y = 0.0;
   const real_t *coordinate = x.HostRead();
   for (int j = 0; j < width; ++j)
   {
      y.Add(coordinate[j], vectors[j]);
   }
}

class DeflationSpaces::Impl
{
public:
   const Operator *A = nullptr;
   const Operator *NR = nullptr, *NL = nullptr;
   const Operator *ZR = nullptr, *ZL = nullptr;
   bool null_shared = false, coarse_shared = false;
   bool ready = false, spd = false;
   int n = 0;
   DeflationSetupOptions options;
   struct Basis
   {
      // Fine-grid columns are never materialized through DenseMatrix's
      // host element access. E and its factor remain small host matrices.
      std::vector<Vector> columns;
      int Width() const { return static_cast<int>(columns.size()); }
      void Clear() { columns.clear(); }
   };
   // AU caches A U; ATV caches A^T V for nonsymmetric solution projection.
   Basis right_null, left_null, U, V, AU, ATV;
   DenseMatrix E, factor;
   Array<int> pivots;
#ifdef MFEM_USE_LAPACK
   DenseMatrix svd_U, svd_Vt;
   Vector svd_values;
   mutable Vector svd_work;
#endif
   // Separate slots are needed because projector operations call one another.
   mutable Vector project_source, project_coeff;
   mutable Vector coarse_source, coarse_coeff;
   mutable Vector residual_source, residual_coeff;
   mutable Vector solution_source, solution_Ax, solution_coarse;
   mutable Vector solution_coeff;
   mutable Vector compatibility_work, validation_image;

   // Operator coarse path: E = Z_L^T A Z_R, solved by the borrowed solver.
   Solver *coarse_solver = nullptr;
   bool coarse_exact = false;
   std::unique_ptr<Operator> coarse_operator;
   // A Z_R as a sparse product, when A and Z_R are HypreParMatrix objects.
   std::unique_ptr<Operator> AZ;
   struct OperatorWork
   {
      Vector projected, restricted, solved, expanded, image;
   };
   // One slot per public operation, since they call one another.
   mutable OperatorWork coarse_work, residual_work, solution_work;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
   explicit Impl(MPI_Comm c) : comm(c) { }
#endif
   Impl() = default;

   void Invalidate()
   {
      ready = false;
      n = 0;
      right_null.Clear();
      left_null.Clear();
      U.Clear();
      V.Clear();
      AU.Clear();
      ATV.Clear();
      coarse_operator.reset();
      AZ.reset();
      E.SetSize(0, 0);
      factor.SetSize(0, 0);
      pivots.SetSize(0);
#ifdef MFEM_USE_LAPACK
      svd_U.SetSize(0, 0);
      svd_Vt.SetSize(0, 0);
      svd_values.SetSize(0);
#endif
   }

   void Check(bool good, const char *message) const
   {
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         int bad = good ? 0 : 1, any_bad = 0;
         MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
         good = any_bad == 0;
      }
#endif
      MFEM_VERIFY(good, message);
   }

   int ConsistentColumns(const Operator *basis) const
   {
      int count = basis ? basis->Width() : 0;
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         int lo = 0, hi = 0;
         MPI_Allreduce(&count, &lo, 1, MPI_INT, MPI_MIN, comm);
         MPI_Allreduce(&count, &hi, 1, MPI_INT, MPI_MAX, comm);
         Check(lo == hi, "Basis column counts differ across MPI ranks");
      }
#endif
      return count;
   }

   /// Whether @a local holds on every rank; collective.
   bool AllRanks(bool local) const
   {
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         int value = local ? 1 : 0, all = 0;
         MPI_Allreduce(&value, &all, 1, MPI_INT, MPI_MIN, comm);
         return all != 0;
      }
#endif
      return local;
   }

   bool Agrees(bool local) const
   {
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         int value = local ? 1 : 0, lo = 0, hi = 0;
         MPI_Allreduce(&value, &lo, 1, MPI_INT, MPI_MIN, comm);
         MPI_Allreduce(&value, &hi, 1, MPI_INT, MPI_MAX, comm);
         return lo == hi;
      }
#endif
      return true;
   }

   real_t Sum(real_t local) const
   {
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL)
      {
         real_t result = 0;
         MPI_Allreduce(&local, &result, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, comm);
         return result;
      }
#endif
      return local;
   }

   real_t Dot(const Vector &a, const Vector &b) const
   {
      return Sum(a * b);
   }

   real_t Norm(const Vector &x) const
   {
      // A sum of squares is never negative, so no clamp is needed; keeping
      // NaN and infinity lets every caller's finiteness test reject them.
      return std::sqrt(Dot(x, x));
   }

   void Restrict(const Basis &basis, const Vector &x,
                 Vector &coords) const
   {
      const int k = basis.Width();
      coords.UseDevice(false);
      coords.SetSize(k);
      for (int j = 0; j < k; ++j)
      {
         coords(j) = basis.columns[j] * x;
      }
#ifdef MFEM_USE_MPI
      if (comm != MPI_COMM_NULL && k)
      {
         MPI_Allreduce(MPI_IN_PLACE, coords.HostReadWrite(), k,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      }
#endif
   }

   void Expand(const Basis &basis, const Vector &coords,
               Vector &x) const
   {
      if (basis.Width())
      {
         x.UseDevice(x.UseDevice() || basis.columns[0].UseDevice());
      }
      x.SetSize(n);
      x = 0.0;
      for (int j = 0; j < basis.Width(); ++j)
      {
         x.Add(coords(j), basis.columns[j]);
      }
   }

   void Project(const Basis &basis, const Vector &x, Vector &y) const
   {
      const Vector *source = &x;
      Vector &coeff = project_coeff;
      if (MayAlias(x, y))
      {
         Like(project_source, x);
         project_source = x;
         source = &project_source;
      }
      Check(source->Size() == n, "Projector vector has wrong local size");
      Restrict(basis, *source, coeff);
      y.UseDevice(y.UseDevice() || source->UseDevice() ||
                  (basis.Width() && basis.columns[0].UseDevice()));
      y = *source;
      for (int j = 0; j < basis.Width(); ++j)
      {
         y.Add(-coeff(j), basis.columns[j]);
      }
   }

   bool OperatorPath() const { return coarse_solver != nullptr; }

   /// w.restricted = Z_L^T Pi_L r.
   void RestrictLeft(const Vector &r, OperatorWork &w) const
   {
      Project(left_null, r, w.projected);
      w.restricted.UseDevice(true);
      w.restricted.SetSize(ZL->Width());
      ZL->MultTranspose(w.projected, w.restricted);
   }

   /// w.solved = S w.restricted.
   void SolveOperatorCoarse(OperatorWork &w) const
   {
      w.solved.UseDevice(true);
      w.solved.SetSize(ZR->Width());
      w.solved = 0.0;
      coarse_solver->Mult(w.restricted, w.solved);
   }

   /// y = Pi_R Z_R w.solved.
   void ExpandRight(OperatorWork &w, Vector &y) const
   {
      w.expanded.UseDevice(true);
      w.expanded.SetSize(n);
      ZR->Mult(w.solved, w.expanded);
      Project(right_null, w.expanded, y);
   }

   /// y = A Q r = A Z_R w.solved, using A Pi_R = A.
   void ApplyAQ(OperatorWork &w, const Vector &qr, Vector &y) const
   {
      y.UseDevice(true);
      y.SetSize(n);
      if (AZ) { AZ->Mult(w.solved, y); }
      else { A->Mult(qr, y); }
   }

   void ResidualProject(const Vector &r, Vector &projected,
                        Vector *coarse) const
   {
      const Vector *source = &r;
      Vector &coeff = residual_coeff;
      if (MayAlias(r, projected) || (coarse && MayAlias(r, *coarse)))
      {
         Like(residual_source, r);
         residual_source = r;
         source = &residual_source;
      }
      Check(source->Size() == n, "Residual vector has wrong local size");
      // Checked after the input is secured, since the test may relocate the
      // output storage without copying it.
      Check(!coarse || !OutputsOverlap(projected, *coarse),
            "Balanced coarse outputs must not overlap");
      if (OperatorPath())
      {
         OperatorWork &w = residual_work;
         RestrictLeft(*source, w);
         SolveOperatorCoarse(w);
         // Q r is needed for the coarse output, and to apply A when A Z_R is
         // not cached. ExpandRight leaves w.expanded free for A Q r.
         Vector &qr = coarse ? *coarse : w.image;
         if (coarse || !AZ) { ExpandRight(w, qr); }
         ApplyAQ(w, qr, w.expanded);
         projected.UseDevice(projected.UseDevice() || source->UseDevice());
         projected = *source;
         projected -= w.expanded;
         return;
      }
      if (!U.Width())
      {
         projected.UseDevice(projected.UseDevice() || source->UseDevice());
         projected = *source;
         if (coarse)
         {
            coarse->UseDevice(coarse->UseDevice() || source->UseDevice());
            coarse->SetSize(n);
            *coarse = 0.0;
         }
         return;
      }
      Restrict(V, *source, coeff);
      SolveCoarse(coeff);
      if (coarse) { Expand(U, coeff, *coarse); }
      projected.UseDevice(projected.UseDevice() || source->UseDevice());
      projected = *source;
      for (int j = 0; j < AU.Width(); ++j)
      {
         projected.Add(-coeff(j), AU.columns[j]);
      }
   }

   void BuildBasis(const Operator *input, const Basis *exclude,
                   Basis &output) const
   {
      const int k = ConsistentColumns(input);
      Check(input->Height() == n && k > 0,
            "Basis shape must be local fine rows by replicated columns");
      output.Clear();
      output.columns.reserve(k);
      // An explicit vector set is copied instead of applied column by column.
      const auto *vectors = dynamic_cast<const VectorDeflationBasis*>(input);
      Vector e(k), column(n), projected;
      const bool use_device = Device::Allows(Backend::DEVICE_MASK);
      e.UseDevice(use_device);
      column.UseDevice(use_device);
      const real_t threshold = AutomaticTolerance(options.basis_rank_rtol,
                                                  real_t(1e-10), k);
      for (int j = 0; j < k; ++j)
      {
         column.UseDevice(use_device);
         column.SetSize(n);
         if (vectors) { column = vectors->GetVector(j); }
         else
         {
            e = 0.0;
            e.HostReadWrite()[j] = 1.0;
            input->Mult(e, column);
         }
         const real_t original_norm = Norm(column);
         Check(std::isfinite(original_norm) && original_norm > 0,
               "Zero or nonfinite basis column");
         column /= original_norm;
         if (exclude)
         {
            Project(*exclude, column, projected);
            column = projected;
         }
         // Reorthogonalization makes the test independent of input scaling.
         for (int pass = 0; pass < 2; ++pass)
         {
            for (int h = 0; h < j; ++h)
            {
               const Vector &previous = output.columns[h];
               column.Add(-Dot(previous, column), previous);
            }
            if (exclude)
            {
               Project(*exclude, column, projected);
               column = projected;
            }
         }
         const real_t remaining = Norm(column);
         Check(std::isfinite(remaining) && remaining > threshold,
               "Null/coarse basis is rank deficient after projection");
         column /= remaining;
         output.columns.push_back(std::move(column));
      }
   }

   bool FactorCoarse()
   {
      const int k = E.Height();
      factor = E;
      pivots.SetSize(k);
      if (spd)
      {
         for (int j = 0; j < k; ++j)
         {
            real_t diagonal = factor(j, j);
            for (int h = 0; h < j; ++h)
            {
               diagonal -= factor(j, h) * factor(j, h);
            }
            if (!(diagonal > 0) || !std::isfinite(diagonal)) { return false; }
            factor(j, j) = std::sqrt(diagonal);
            for (int i = j + 1; i < k; ++i)
            {
               real_t value = factor(i, j);
               for (int h = 0; h < j; ++h)
               {
                  value -= factor(i, h) * factor(j, h);
               }
               factor(i, j) = value / factor(j, j);
            }
         }
         return true;
      }
      for (int j = 0; j < k; ++j)
      {
         int pivot = j;
         for (int i = j + 1; i < k; ++i)
         {
            if (std::abs(factor(i, j)) > std::abs(factor(pivot, j)))
            {
               pivot = i;
            }
         }
         if (!(std::abs(factor(pivot, j)) > 0) ||
             !std::isfinite(factor(pivot, j))) { return false; }
         pivots[j] = pivot;
         if (pivot != j)
         {
            for (int h = 0; h < k; ++h)
            {
               std::swap(factor(j, h), factor(pivot, h));
            }
         }
         for (int i = j + 1; i < k; ++i)
         {
            factor(i, j) /= factor(j, j);
            for (int h = j + 1; h < k; ++h)
            {
               factor(i, h) -= factor(i, j) * factor(j, h);
            }
         }
      }
      return true;
   }

   void SolveCoarse(Vector &rhs) const
   {
#ifdef MFEM_USE_LAPACK
      if (options.coarse_solve_method == CoarseSolveMethod::SVD)
      {
         const int k = svd_values.Size();
         svd_work.SetSize(k);
         for (int i = 0; i < k; ++i)
         {
            real_t value = 0;
            for (int j = 0; j < k; ++j)
            {
               value += svd_U(j, i) * rhs(j);
            }
            svd_work(i) = value / svd_values(i);
         }
         for (int i = 0; i < k; ++i)
         {
            real_t value = 0;
            for (int j = 0; j < k; ++j)
            {
               value += svd_Vt(j, i) * svd_work(j);
            }
            rhs(i) = value;
         }
         return;
      }
#endif
      const int k = factor.Height();
      if (spd)
      {
         for (int i = 0; i < k; ++i)
         {
            for (int j = 0; j < i; ++j)
            {
               rhs(i) -= factor(i, j) * rhs(j);
            }
            rhs(i) /= factor(i, i);
         }
         for (int i = k - 1; i >= 0; --i)
         {
            for (int j = i + 1; j < k; ++j)
            {
               rhs(i) -= factor(j, i) * rhs(j);
            }
            rhs(i) /= factor(i, i);
         }
         return;
      }
      for (int i = 0; i < k; ++i) { std::swap(rhs(i), rhs(pivots[i])); }
      for (int i = 0; i < k; ++i)
      {
         for (int j = 0; j < i; ++j)
         {
            rhs(i) -= factor(i, j) * rhs(j);
         }
      }
      for (int i = k - 1; i >= 0; --i)
      {
         for (int j = i + 1; j < k; ++j)
         {
            rhs(i) -= factor(i, j) * rhs(j);
         }
         rhs(i) /= factor(i, i);
      }
   }

   real_t ReciprocalCondition() const
   {
      const int k = E.Height();
      real_t norm_E = 0, norm_inverse = 0;
      for (int j = 0; j < k; ++j)
      {
         real_t sum = 0;
         for (int i = 0; i < k; ++i) { sum += std::abs(E(i, j)); }
         norm_E = std::max(norm_E, sum);
         Vector e(k);
         e = 0.0;
         e(j) = 1.0;
         SolveCoarse(e);
         sum = 0;
         for (int i = 0; i < k; ++i) { sum += std::abs(e(i)); }
         norm_inverse = std::max(norm_inverse, sum);
      }
      return real_t(1) / (norm_E * norm_inverse);
   }

   /** Form E = Z_L^T A Z_R and bind the coarse solver. Exact null spaces
       give Pi_L A Pi_R = A, so the candidates need no projection. */
   void SetupOperatorCoarse()
   {
      Check(ZR->Height() == n && ZL->Height() == n,
            "Coarse operator rows must match the local fine size");
      Check(ZR->Width() == ZL->Width(),
            "Trial and test coarse operators need equal local widths");
#ifdef MFEM_USE_MPI
      const auto *Ah = dynamic_cast<const HypreParMatrix*>(A);
      const auto *ZRh = dynamic_cast<const HypreParMatrix*>(ZR);
      const auto *ZLh = dynamic_cast<const HypreParMatrix*>(ZL);
      if (Agrees(Ah && ZRh && ZLh) && Ah && ZRh && ZLh)
      {
         HypreParMatrix *product = ParMult(Ah, ZRh, true);
         AZ.reset(product);
         // Pointer identity is local; the branch changes the communication,
         // so every rank must take the same one.
         if (coarse_shared || AllRanks(ZLh == ZRh))
         {
            coarse_operator.reset(RAP(Ah, ZRh));
         }
         else
         {
            std::unique_ptr<HypreParMatrix> ZLt(ZLh->Transpose());
            coarse_operator.reset(ParMult(ZLt.get(), product, true));
         }
      }
      else
#endif
      {
         coarse_operator.reset(new RAPOperator(*ZL, *A, *ZR));
      }
      coarse_solver->iterative_mode = false;
      coarse_solver->SetOperator(*coarse_operator);
      Check(coarse_solver->Height() == ZR->Width() &&
            coarse_solver->Width() == ZR->Width(),
            "Coarse solver dimensions differ from Z_L^T A Z_R");
   }

   void Setup(bool symmetric)
   {
      Invalidate();
      spd = symmetric;
      Check(A != nullptr, "Deflation needs an original operator");
      n = A->Height();
      Check(n >= 0 && A->Width() == n,
            "Deflation requires a square local fine operator");
      Check(ValidTolerance(options.basis_rank_rtol, true) &&
            ValidTolerance(options.coarse_rcond_min, true),
            "Invalid deflation setup tolerance");
      Check(options.coarse_solve_method == CoarseSolveMethod::DIRECT ||
            options.coarse_solve_method == CoarseSolveMethod::SVD,
            "Invalid coarse solve method");
      Check(Agrees(options.coarse_solve_method == CoarseSolveMethod::SVD),
            "Coarse solve method differs across MPI ranks");
#ifndef MFEM_USE_LAPACK
      Check(options.coarse_solve_method != CoarseSolveMethod::SVD,
            "SVD coarse solve requires MFEM_USE_LAPACK");
#endif
      Check(Agrees(options.cache_transpose_images),
            "Transpose image caching differs across MPI ranks");
      Check((NR == nullptr) == (NL == nullptr) &&
            (ZR == nullptr) == (ZL == nullptr),
            "Left/right space registration is incomplete");
      Check(Agrees(NR != nullptr) && Agrees(ZR != nullptr),
            "Space registration differs across MPI ranks");
      Check(Agrees(null_shared) && Agrees(coarse_shared),
            "Shared/paired basis registration differs across MPI ranks");
      Check(!symmetric || ((!NR || null_shared) && (!ZR || coarse_shared)),
            "CG requires shared null and coarse bases");
      Check(Agrees(coarse_solver != nullptr) && Agrees(coarse_exact),
            "Coarse-space representation differs across MPI ranks");
      if (NR)
      {
         const int right_count = ConsistentColumns(NR);
         const int left_count = ConsistentColumns(NL);
         Check(right_count == left_count,
               "Left/right null basis counts differ");
         BuildBasis(NR, nullptr, right_null);
         if (null_shared) { left_null = right_null; }
         else { BuildBasis(NL, nullptr, left_null); }
      }
      if (ZR && OperatorPath())
      {
         SetupOperatorCoarse();
      }
      else if (ZR)
      {
         const int right_count = ConsistentColumns(ZR);
         const int left_count = ConsistentColumns(ZL);
         Check(right_count == left_count,
               "Left/right coarse basis counts differ");
         BuildBasis(ZR, NR ? &right_null : nullptr, U);
         if (symmetric) { V = U; }
         else { BuildBasis(ZL, NL ? &left_null : nullptr, V); }
         const int k = U.Width();
         E.SetSize(k);
         AU.columns.reserve(k);
         Vector Au, coords;
         for (int j = 0; j < k; ++j)
         {
            const Vector &u = U.columns[j];
            Like(Au, u);
            A->Mult(u, Au);
            AU.columns.push_back(std::move(Au));
            Restrict(V, AU.columns.back(), coords);
            for (int i = 0; i < k; ++i) { E(i, j) = coords(i); }
         }
         // Symmetric setups reuse A U, since then V^T A = (A U)^T.
         if (!symmetric && options.cache_transpose_images)
         {
            ATV.columns.reserve(k);
            Vector image;
            for (int j = 0; j < k; ++j)
            {
               const Vector &v = V.columns[j];
               Like(image, v);
               A->MultTranspose(v, image);
               ATV.columns.push_back(std::move(image));
            }
         }
         const real_t threshold = AutomaticTolerance(options.coarse_rcond_min,
                                                      real_t(1e-12), k);
         if (options.coarse_solve_method == CoarseSolveMethod::SVD)
         {
#ifdef MFEM_USE_LAPACK
            if (symmetric)
            {
               Check(FactorCoarse(), "Coarse matrix factorization failed");
            }
            DenseMatrixSVD svd(E, 'S', 'S');
            svd.Eval(E);
            svd_U = svd.LeftSingularvectors();
            svd_Vt = svd.RightSingularvectors();
            svd_values = svd.Singularvalues();
            const real_t largest = svd_values(0);
            const real_t smallest = svd_values(k - 1);
            bool finite = true;
            for (int i = 0; i < k; ++i)
            {
               finite = finite && std::isfinite(svd_values(i));
            }
            Check(finite && largest > 0 && smallest > 0 &&
                  smallest / largest >= threshold,
                  "Coarse matrix is singular or too ill-conditioned");
#endif
         }
         else
         {
            Check(FactorCoarse(), "Coarse matrix factorization failed");
            const real_t rcond = ReciprocalCondition();
            Check(std::isfinite(rcond) && rcond >= threshold,
                  "Coarse matrix is singular or too ill-conditioned");
         }
      }
      ready = true;
   }
};

DeflationSpaces::DeflationSpaces() : impl(new Impl()) { }
#ifdef MFEM_USE_MPI
DeflationSpaces::DeflationSpaces(MPI_Comm comm) : impl(new Impl(comm))
{
   MFEM_VERIFY(comm != MPI_COMM_NULL, "Deflation communicator is null");
}
#endif
DeflationSpaces::~DeflationSpaces() = default;

void DeflationSpaces::SetOperator(const Operator &A)
{
   impl->A = &A;
   impl->Invalidate();
}
const Operator *DeflationSpaces::GetOriginalOperator() const { return impl->A; }
void DeflationSpaces::SetNullSpace(const Operator &N)
{
   impl->NR = impl->NL = &N;
   impl->null_shared = true;
   impl->Invalidate();
}
void DeflationSpaces::SetNullSpaces(const Operator &right,
                                   const Operator &left)
{
   impl->NR = &right;
   impl->NL = &left;
   impl->null_shared = false;
   impl->Invalidate();
}
void DeflationSpaces::ClearNullSpace()
{
   impl->NR = impl->NL = nullptr;
   impl->null_shared = false;
   impl->Invalidate();
}
void DeflationSpaces::SetCoarseSpace(const Operator &Z)
{
   impl->ZR = impl->ZL = &Z;
   impl->coarse_shared = true;
   impl->coarse_solver = nullptr;
   impl->coarse_exact = false;
   impl->Invalidate();
}
void DeflationSpaces::SetCoarseSpaces(const Operator &right,
                                     const Operator &left)
{
   impl->ZR = &right;
   impl->ZL = &left;
   impl->coarse_shared = false;
   impl->coarse_solver = nullptr;
   impl->coarse_exact = false;
   impl->Invalidate();
}
void DeflationSpaces::SetCoarseSpace(const Operator &Z, Solver &S,
                                     bool exact_coarse_solve)
{
   impl->ZR = impl->ZL = &Z;
   impl->coarse_shared = true;
   impl->coarse_solver = &S;
   impl->coarse_exact = exact_coarse_solve;
   impl->Invalidate();
}
void DeflationSpaces::SetCoarseSpaces(const Operator &right,
                                      const Operator &left, Solver &S,
                                      bool exact_coarse_solve)
{
   impl->ZR = &right;
   impl->ZL = &left;
   impl->coarse_shared = false;
   impl->coarse_solver = &S;
   impl->coarse_exact = exact_coarse_solve;
   impl->Invalidate();
}
void DeflationSpaces::ClearCoarseSpace()
{
   impl->ZR = impl->ZL = nullptr;
   impl->coarse_shared = false;
   impl->coarse_solver = nullptr;
   impl->coarse_exact = false;
   impl->Invalidate();
}
void DeflationSpaces::SetSetupOptions(const DeflationSetupOptions &options)
{
   impl->options = options;
   impl->Invalidate();
}
const DeflationSetupOptions &DeflationSpaces::GetSetupOptions() const
{ return impl->options; }
bool DeflationSpaces::HasNullSpace() const { return impl->NR != nullptr; }
bool DeflationSpaces::HasCoarseSpace() const { return impl->ZR != nullptr; }
bool DeflationSpaces::UsesSharedNullBasis() const { return impl->null_shared; }
bool DeflationSpaces::UsesSharedCoarseBasis() const
{ return impl->coarse_shared; }
bool DeflationSpaces::UsesCoarseSolver() const
{ return impl->OperatorPath(); }
bool DeflationSpaces::HasExactCoarseSolve() const
{ return !impl->OperatorPath() || impl->coarse_exact; }
bool DeflationSpaces::IsSetup() const { return impl->ready; }
void DeflationSpaces::Setup(bool symmetric_positive_definite)
{ impl->Setup(symmetric_positive_definite); }
void DeflationSpaces::Update(bool symmetric_positive_definite)
{ impl->Setup(symmetric_positive_definite); }
int DeflationSpaces::GetNullSpaceDimension() const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   return impl->right_null.Width();
}
int DeflationSpaces::GetCoarseSpaceDimension() const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   if (!impl->ZR) { return 0; }
   if (!impl->OperatorPath()) { return impl->U.Width(); }
   // Operator coarse coordinates are distributed; add the local widths.
   return static_cast<int>(impl->Sum(real_t(impl->ZR->Width())));
}
const DenseMatrix &DeflationSpaces::GetCoarseMatrix() const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   MFEM_VERIFY(!impl->OperatorPath(),
               "Use GetCoarseOperator() for an operator coarse space");
   return impl->E;
}
const Operator *DeflationSpaces::GetCoarseOperator() const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   if (!impl->ZR) { return nullptr; }
   if (impl->OperatorPath()) { return impl->coarse_operator.get(); }
   return &impl->E;
}
void DeflationSpaces::ProjectSolution(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   impl->Project(impl->right_null, x, y);
}
void DeflationSpaces::ProjectRHS(const Vector &b, Vector &y) const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   impl->Project(impl->left_null, b, y);
}
void DeflationSpaces::ApplyCoarseCorrection(const Vector &r, Vector &y) const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   const Vector *source = &r;
   Vector &coeff = impl->coarse_coeff;
   if (MayAlias(r, y))
   {
      Like(impl->coarse_source, r);
      impl->coarse_source = r;
      source = &impl->coarse_source;
   }
   impl->Check(source->Size() == impl->n,
               "Coarse vector has wrong local size");
   if (impl->OperatorPath())
   {
      Impl::OperatorWork &w = impl->coarse_work;
      impl->RestrictLeft(*source, w);
      impl->SolveOperatorCoarse(w);
      impl->ExpandRight(w, y);
      return;
   }
   if (!impl->U.Width())
   {
      y.UseDevice(y.UseDevice() || source->UseDevice());
      y.SetSize(impl->n);
      y = 0.0;
      return;
   }
   impl->Restrict(impl->V, *source, coeff);
   impl->SolveCoarse(coeff);
   impl->Expand(impl->U, coeff, y);
}
void DeflationSpaces::ApplyResidualProjector(const Vector &r, Vector &y) const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   impl->ResidualProject(r, y, nullptr);
}
void DeflationSpaces::ApplyResidualProjectorAndCoarseCorrection(
   const Vector &r, Vector &projected, Vector &coarse) const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   impl->ResidualProject(r, projected, &coarse);
}
void DeflationSpaces::ApplySolutionProjector(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   const Vector *source = &x;
   Vector &Ax = impl->solution_Ax;
   Vector &coarse = impl->solution_coarse;
   if (MayAlias(x, y))
   {
      Like(impl->solution_source, x);
      impl->solution_source = x;
      source = &impl->solution_source;
   }
   impl->Check(source->Size() == impl->n,
               "Solution vector has wrong local size");
   if (impl->OperatorPath())
   {
      Impl::OperatorWork &w = impl->solution_work;
      if (impl->AZ && impl->spd)
      {
         // Z^T Pi_L A x = Z^T A x = (A Z)^T x for symmetric A, shared Z.
         w.restricted.UseDevice(true);
         w.restricted.SetSize(impl->ZR->Width());
         impl->AZ->MultTranspose(*source, w.restricted);
      }
      else
      {
         Like(Ax, *source);
         impl->A->Mult(*source, Ax);
         impl->RestrictLeft(Ax, w);
      }
      impl->SolveOperatorCoarse(w);
      impl->ExpandRight(w, coarse);
      y.UseDevice(y.UseDevice() || source->UseDevice() || coarse.UseDevice());
      y = *source;
      y -= coarse;
      return;
   }
   if (!impl->U.Width())
   {
      y.UseDevice(y.UseDevice() || source->UseDevice());
      y = *source;
      return;
   }
   // V^T A x is (A^T V)^T x: A U when symmetric, else the optional A^T V.
   const Impl::Basis *transpose_images =
      impl->spd ? &impl->AU : (impl->ATV.Width() ? &impl->ATV : nullptr);
   if (transpose_images)
   {
      Vector &coeff = impl->solution_coeff;
      impl->Restrict(*transpose_images, *source, coeff);
      impl->SolveCoarse(coeff);
      impl->Expand(impl->U, coeff, coarse);
   }
   else
   {
      Like(Ax, *source);
      impl->A->Mult(*source, Ax);
      ApplyCoarseCorrection(Ax, coarse);
   }
   y.UseDevice(y.UseDevice() || source->UseDevice() || coarse.UseDevice());
   y = *source;
   y -= coarse;
}
real_t DeflationSpaces::GetIncompatibleRHSNorm(const Vector &b) const
{
   Vector &work = impl->compatibility_work;
   ProjectRHS(b, work);
   work.Neg();
   work += b;
   return impl->Norm(work);
}
bool DeflationSpaces::IsConsistent(const Vector &b, real_t atol,
                                  real_t rtol) const
{
   MFEM_VERIFY(ValidTolerance(atol) && ValidTolerance(rtol),
               "Invalid RHS tolerance");
   return GetIncompatibleRHSNorm(b) <= atol + rtol * impl->Norm(b);
}
bool DeflationSpaces::ValidateNullSpaces(real_t absolute_tolerance) const
{
   MFEM_VERIFY(impl->ready, "Deflation setup required");
   MFEM_VERIFY(ValidTolerance(absolute_tolerance),
               "Invalid null validation tolerance");
   if (!impl->NR) { return true; }
   real_t right_square = 0, left_square = 0;
   Vector &image = impl->validation_image;
   for (int j = 0; j < impl->right_null.Width(); ++j)
   {
      const Vector &column = impl->right_null.columns[j];
      Like(image, column);
      impl->A->Mult(column, image);
      const real_t norm = impl->Norm(image);
      right_square += norm * norm;
      const Vector &left_column = impl->left_null.columns[j];
      Like(image, left_column);
      impl->A->MultTranspose(left_column, image);
      const real_t transpose_norm = impl->Norm(image);
      left_square += transpose_norm * transpose_norm;
   }
   return std::sqrt(right_square) <= absolute_tolerance &&
          std::sqrt(left_square) <= absolute_tolerance;
}

template <typename KrylovSolver>
class DeflatedSolverBase<KrylovSolver>::Impl
{
public:
   DeflationSpaces spaces;
   const Operator *original = nullptr;
   Solver *user_prec = nullptr;
   CoarseCorrectionType mode = CoarseCorrectionType::BALANCED;
   DeflationRHSOptions rhs_options;
   DeflationSolveInfo info;
   bool bound = false;
   bool preconditioner_bound = false;
   struct Workspace
   {
      Vector b0_left, b0_fine;
      Vector bal_left, bal_projected, bal_fine, bal_right, bal_coarse, bal_sum;
      Vector adapter_image, recover_coarse, recover_projected, recover_gauged;
      Vector residual_Ax;
      Vector b_work, x0, original_initial, working_initial;
      Vector rhs, internal, physical, residual;
      Vector operator_image, cg_r, cg_z, direction, Ad;
      Vector krylov_rhs, krylov_residual, cycle_start, image, w;
      Vector coefficients, original_residual, projected_solution;
      Vector null_component;
      std::vector<Vector> v, z;
      DenseMatrix H;
      std::vector<real_t> cs, sn, g;
   };
   mutable Workspace work;

   struct OperatorAdapter : public Operator
   {
      Impl &owner;
      explicit OperatorAdapter(Impl &p) : Operator(0), owner(p) { }
      void Resize(int n) { height = width = n; }
      void Mult(const Vector &x, Vector &y) const override
      { owner.ApplyOperator(x, y); }
   } operator_adapter;

   struct PreconditionerAdapter : public Solver
   {
      Impl &owner;
      explicit PreconditionerAdapter(Impl &p) : Solver(0), owner(p) { }
      void Resize(int n) { height = width = n; }
      void SetOperator(const Operator &) override { }
      void Mult(const Vector &x, Vector &y) const override
      { owner.ApplyPreconditioner(x, y); }
   } preconditioner_adapter;

   Impl() : operator_adapter(*this), preconditioner_adapter(*this) { }
#ifdef MFEM_USE_MPI
   explicit Impl(MPI_Comm comm)
      : spaces(comm), operator_adapter(*this), preconditioner_adapter(*this) { }
#endif

   void Invalidate()
   {
      bound = false;
      info = DeflationSolveInfo();
   }

   void ApplyFine(const Vector &input, Vector &output) const
   {
      Like(output, input);
      if (!user_prec) { output = input; return; }
      output = 0.0;
      user_prec->Mult(input, output);
   }

   void ApplyB0(const Vector &input, Vector &output) const
   {
      Vector &left = work.b0_left, &fine = work.b0_fine;
      spaces.ProjectRHS(input, left);
      ApplyFine(left, fine);
      spaces.ProjectSolution(fine, output);
   }

   void ApplyBalanced(const Vector &input, Vector &output) const
   {
      Vector &left = work.bal_left, &projected = work.bal_projected;
      Vector &fine = work.bal_fine, &right = work.bal_right;
      Vector &coarse = work.bal_coarse, &sum = work.bal_sum;
      spaces.ProjectRHS(input, left);
      spaces.ApplyResidualProjectorAndCoarseCorrection(left, projected,
                                                       coarse);
      ApplyFine(projected, fine);
      spaces.ApplySolutionProjector(fine, right);
      sum.UseDevice(sum.UseDevice() || right.UseDevice());
      sum = right;
      sum += coarse;
      spaces.ProjectSolution(sum, output);
   }

   void ApplyPreconditioner(const Vector &input, Vector &output) const
   {
      if (mode == CoarseCorrectionType::PROJECTED)
      {
         ApplyB0(input, output);
      }
      else { ApplyBalanced(input, output); }
   }

   void ApplyOperator(const Vector &input, Vector &output) const
   {
      Vector &image = work.adapter_image;
      image.UseDevice(input.UseDevice() || output.UseDevice());
      image.SetSize(input.Size());
      original->Mult(input, image);
      if (mode == CoarseCorrectionType::PROJECTED)
      {
         spaces.ApplyResidualProjector(image, output);
      }
      else
      {
         output.UseDevice(output.UseDevice() || image.UseDevice());
         output = image;
      }
   }

   void WorkingRHS(const Vector &b_work, Vector &rhs) const
   {
      if (mode == CoarseCorrectionType::PROJECTED)
      {
         spaces.ApplyResidualProjector(b_work, rhs);
      }
      else
      {
         rhs.UseDevice(rhs.UseDevice() || b_work.UseDevice());
         rhs = b_work;
      }
   }

   void Recover(const Vector &internal, const Vector &b_work,
                Vector &physical) const
   {
      if (mode == CoarseCorrectionType::PROJECTED)
      {
         Vector &coarse = work.recover_coarse;
         Vector &projected = work.recover_projected;
         Vector &gauged = work.recover_gauged;
         spaces.ApplyCoarseCorrection(b_work, coarse);
         spaces.ApplySolutionProjector(internal, projected);
         spaces.ProjectSolution(projected, gauged);
         physical.UseDevice(physical.UseDevice() || coarse.UseDevice() ||
                            gauged.UseDevice());
         physical = coarse;
         physical += gauged;
      }
      else { spaces.ProjectSolution(internal, physical); }
   }

   void Residual(const Vector &rhs, const Vector &physical,
                 Vector &residual) const
   {
      Vector &Ax = work.residual_Ax;
      Ax.UseDevice(rhs.UseDevice() || physical.UseDevice());
      Ax.SetSize(rhs.Size());
      original->Mult(physical, Ax);
      residual.UseDevice(residual.UseDevice() || Ax.UseDevice());
      residual = rhs;
      residual -= Ax;
   }
};

template <typename KrylovSolver>
DeflatedSolverBase<KrylovSolver>::DeflatedSolverBase()
   : KrylovSolver(), impl(new Impl())
{
   this->oper = &impl->operator_adapter;
   this->prec = &impl->preconditioner_adapter;
}

#ifdef MFEM_USE_MPI
template <typename KrylovSolver>
DeflatedSolverBase<KrylovSolver>::DeflatedSolverBase(MPI_Comm comm)
   : KrylovSolver(comm), impl(new Impl(comm))
{
   this->oper = &impl->operator_adapter;
   this->prec = &impl->preconditioner_adapter;
}
#endif

template <typename KrylovSolver>
DeflatedSolverBase<KrylovSolver>::~DeflatedSolverBase()
{
   this->oper = nullptr;
   this->prec = nullptr;
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::Invalidate()
{
   impl->Invalidate();
   ResetSolveInfo();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::ResetSolveInfo()
{
   impl->info = DeflationSolveInfo();
   this->final_iter = -1;
   this->converged = false;
   this->initial_norm = this->final_norm = -1;
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::CheckCollective(
   bool good, const char *message) const
{
#ifdef MFEM_USE_MPI
   const MPI_Comm comm = this->GetComm();
   if (comm != MPI_COMM_NULL)
   {
      int bad = good ? 0 : 1, any_bad = 0;
      MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
      good = any_bad == 0;
   }
#endif
   MFEM_VERIFY(good, message);
}

template <typename KrylovSolver>
bool DeflatedSolverBase<KrylovSolver>::AnyRank(bool local) const
{
#ifdef MFEM_USE_MPI
   const MPI_Comm comm = this->GetComm();
   if (comm != MPI_COMM_NULL)
   {
      int value = local ? 1 : 0, any = 0;
      MPI_Allreduce(&value, &any, 1, MPI_INT, MPI_MAX, comm);
      return any != 0;
   }
#endif
   return local;
}

template <typename KrylovSolver>
bool DeflatedSolverBase<KrylovSolver>::CollectiveStop(bool local_stop) const
{
#ifdef MFEM_USE_MPI
   const MPI_Comm comm = this->GetComm();
   if (comm != MPI_COMM_NULL)
   {
      int local = local_stop ? 1 : 0, global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm);
      return global != 0;
   }
#endif
   return local_stop;
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetOperator(const Operator &A)
{
   impl->original = &A;
   impl->preconditioner_bound = false;
   impl->spaces.SetOperator(A);
   this->height = A.Height();
   this->width = A.Width();
   impl->operator_adapter.Resize(A.Height());
   impl->preconditioner_adapter.Resize(A.Height());
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetPreconditioner(Solver &M_inverse)
{
   impl->user_prec = &M_inverse;
   impl->preconditioner_bound = false;
   M_inverse.iterative_mode = false;
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::ClearPreconditioner()
{
   impl->user_prec = nullptr;
   impl->preconditioner_bound = false;
   Invalidate();
}

template <typename KrylovSolver>
const Operator *DeflatedSolverBase<KrylovSolver>::GetOriginalOperator() const
{ return impl->original; }

template <typename KrylovSolver>
Solver *DeflatedSolverBase<KrylovSolver>::GetUserPreconditioner() const
{ return impl->user_prec; }

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetNullSpace(const Operator &N)
{
   impl->spaces.SetNullSpace(N);
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetNullSpaces(const Operator &right,
                                                    const Operator &left)
{
   const bool cg = std::is_same<KrylovSolver, CGSolver>::value;
   MFEM_VERIFY(!cg,
               "CG accepts only shared null bases");
   impl->spaces.SetNullSpaces(right, left);
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::ClearNullSpace()
{
   impl->spaces.ClearNullSpace();
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetCoarseSpace(const Operator &Z)
{
   impl->spaces.SetCoarseSpace(Z);
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetCoarseSpaces(const Operator &right,
                                                      const Operator &left)
{
   const bool cg = std::is_same<KrylovSolver, CGSolver>::value;
   MFEM_VERIFY(!cg,
               "CG accepts only shared coarse bases");
   impl->spaces.SetCoarseSpaces(right, left);
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetCoarseSpace(const Operator &Z,
                                                     Solver &S,
                                                     bool exact_coarse_solve)
{
   impl->spaces.SetCoarseSpace(Z, S, exact_coarse_solve);
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetCoarseSpaces(
   const Operator &right, const Operator &left, Solver &S,
   bool exact_coarse_solve)
{
   const bool cg = std::is_same<KrylovSolver, CGSolver>::value;
   MFEM_VERIFY(!cg,
               "CG accepts only shared coarse bases");
   impl->spaces.SetCoarseSpaces(right, left, S, exact_coarse_solve);
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::ClearCoarseSpace()
{
   impl->spaces.ClearCoarseSpace();
   Invalidate();
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetCoarseCorrectionType(
   CoarseCorrectionType type)
{
   MFEM_VERIFY(type == CoarseCorrectionType::PROJECTED ||
               type == CoarseCorrectionType::BALANCED,
               "Invalid coarse correction type");
   impl->mode = type;
   ResetSolveInfo();
}

template <typename KrylovSolver>
CoarseCorrectionType
DeflatedSolverBase<KrylovSolver>::GetCoarseCorrectionType() const
{ return impl->mode; }

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetSetupOptions(
   const DeflationSetupOptions &options)
{
   impl->spaces.SetSetupOptions(options);
   Invalidate();
}

template <typename KrylovSolver>
const DeflationSetupOptions &
DeflatedSolverBase<KrylovSolver>::GetSetupOptions() const
{ return impl->spaces.GetSetupOptions(); }

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::SetRHSOptions(
   const DeflationRHSOptions &options)
{
   impl->rhs_options = options;
   ResetSolveInfo();
}

template <typename KrylovSolver>
const DeflationRHSOptions &
DeflatedSolverBase<KrylovSolver>::GetRHSOptions() const
{ return impl->rhs_options; }

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::Setup()
{
   Invalidate();
   // Rebuild on every rank if any rank's algebra is stale, so that all ranks
   // follow the same collective sequence.
   if (AnyRank(!impl->spaces.IsSetup()))
   {
      impl->spaces.Setup(std::is_same<KrylovSolver, CGSolver>::value);
   }
#ifdef MFEM_USE_MPI
   if (this->GetComm() != MPI_COMM_NULL)
   {
      const int has_prec = impl->user_prec ? 1 : 0;
      int lo = 0, hi = 0;
      MPI_Allreduce(&has_prec, &lo, 1, MPI_INT, MPI_MIN, this->GetComm());
      MPI_Allreduce(&has_prec, &hi, 1, MPI_INT, MPI_MAX, this->GetComm());
      MFEM_VERIFY(lo == hi,
                  "Fine preconditioner registration differs across ranks");
   }
#endif
   CheckCollective(!impl->spaces.HasNullSpace() ||
                   impl->spaces.UsesSharedNullBasis() || impl->user_prec,
                   "Paired null spaces need an explicit fine preconditioner");
   if (impl->user_prec)
   {
      impl->user_prec->iterative_mode = false;
      // Registration agrees across ranks, so every rank reaches this
      // decision; rebind everywhere if any rank's binding is stale.
      if (AnyRank(!impl->preconditioner_bound))
      {
         impl->user_prec->SetOperator(*impl->original);
      }
      CheckCollective(impl->user_prec->Height() == this->height &&
                      impl->user_prec->Width() == this->width,
                      "Fine preconditioner has wrong local dimensions");
      impl->preconditioner_bound = true;
   }
   impl->bound = true;
}

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::Update()
{
   Invalidate();
   impl->preconditioner_bound = false;
   impl->spaces.Update(std::is_same<KrylovSolver, CGSolver>::value);
   Setup();
}

template <typename KrylovSolver>
bool DeflatedSolverBase<KrylovSolver>::IsSetup() const
{ return impl->bound && impl->spaces.IsSetup(); }

template <typename KrylovSolver>
const DeflationSpaces &
DeflatedSolverBase<KrylovSolver>::GetDeflationSpaces() const
{ return impl->spaces; }

template <typename KrylovSolver>
const DeflationSolveInfo &
DeflatedSolverBase<KrylovSolver>::GetSolveInfo() const
{ return impl->info; }

template <typename KrylovSolver>
int DeflatedSolverBase<KrylovSolver>::RestartDimension() const { return 0; }

template <>
int DeflatedSolverBase<GMRESSolver>::RestartDimension() const
{ return this->m; }

template <>
int DeflatedSolverBase<FGMRESSolver>::RestartDimension() const
{ return this->m; }

namespace
{

bool BackSubstitute(const DenseMatrix &H, const std::vector<real_t> &g,
                    int columns, Vector &coeff)
{
   coeff.SetSize(columns);
   for (int i = columns - 1; i >= 0; --i)
   {
      real_t value = g[i];
      for (int j = i + 1; j < columns; ++j)
      {
         value -= H(i, j) * coeff(j);
      }
      if (!(std::abs(H(i, i)) > 0) || !std::isfinite(H(i, i)))
      {
         return false;
      }
      coeff(i) = value / H(i, i);
   }
   return true;
}

} // namespace

template <typename KrylovSolver>
void DeflatedSolverBase<KrylovSolver>::Mult(const Vector &b, Vector &x) const
{
   impl->info = DeflationSolveInfo();
   this->final_iter = -1;
   this->converged = false;
   this->initial_norm = this->final_norm = -1;
   const bool use_device = b.UseDevice() || x.UseDevice() ||
                           Device::Allows(Backend::DEVICE_MASK);
   // Reject any overlap of b and x, not only equal start addresses: x is
   // written before the original-system residual reads b. Both are read in
   // the same memory space, which keeps x's initial guess intact and asks for
   // no fine-grid device-to-host copy.
   const bool overlap = b.Size() && x.Size() &&
                        RangesOverlap(b.Read(use_device), b.Size(),
                                      x.Read(use_device), x.Size());
   CheckCollective(&b != &x && !overlap,
                   "Deflation Mult does not support b/x aliasing");
   x.UseDevice(use_device);
   CheckCollective(this->dot_oper == nullptr,
                   "Deflation requires Euclidean inner products");
   CheckCollective(this->max_iter >= 0, "Negative iteration budget");
   CheckCollective(ValidTolerance(this->rel_tol) &&
                   ValidTolerance(this->abs_tol),
                   "Invalid stopping tolerance");
   const bool cg = std::is_same<KrylovSolver, CGSolver>::value;
   const bool flexible = std::is_same<KrylovSolver, FGMRESSolver>::value;
   const int restart = RestartDimension();
   CheckCollective(cg || restart > 0,
                   "GMRES restart dimension must be positive");
   CheckCollective(ValidTolerance(impl->rhs_options.atol) &&
                   ValidTolerance(impl->rhs_options.rtol),
                   "Invalid RHS tolerance");
   CheckCollective(impl->rhs_options.action == IncompatibleRHSAction::REJECT ||
                   impl->rhs_options.action == IncompatibleRHSAction::PROJECT,
                   "Invalid RHS action");
#ifdef MFEM_USE_MPI
   if (this->GetComm() != MPI_COMM_NULL)
   {
      int local[6] = { static_cast<int>(impl->mode),
                       static_cast<int>(impl->rhs_options.action),
                       this->max_iter, restart, this->iterative_mode ? 1 : 0,
                       this->controller ? 1 : 0 };
      int lo[6], hi[6];
      MPI_Allreduce(local, lo, 6, MPI_INT, MPI_MIN, this->GetComm());
      MPI_Allreduce(local, hi, 6, MPI_INT, MPI_MAX, this->GetComm());
      bool same = true;
      for (int i = 0; i < 6; ++i) { same = same && lo[i] == hi[i]; }
      real_t local_tol[4] = { this->rel_tol, this->abs_tol,
                              impl->rhs_options.rtol, impl->rhs_options.atol };
      real_t lo_tol[4], hi_tol[4];
      MPI_Allreduce(local_tol, lo_tol, 4, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, this->GetComm());
      MPI_Allreduce(local_tol, hi_tol, 4, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, this->GetComm());
      for (int i = 0; i < 4; ++i)
      {
         same = same && lo_tol[i] == hi_tol[i];
      }
      MFEM_VERIFY(same, "Deflation solve options differ across MPI ranks");
   }
#endif

   // Decide collectively, so that all ranks enter Setup() and its reductions.
   if (AnyRank(!IsSetup()))
   {
      const_cast<DeflatedSolverBase *>(this)->Setup();
   }
   CheckCollective(b.Size() == this->height && x.Size() == this->width,
                   "Deflation vector has wrong local size");
   // Its projected operator and recovery assume P_L and P_R are projectors.
   CheckCollective(impl->mode != CoarseCorrectionType::PROJECTED ||
                   impl->spaces.HasExactCoarseSolve(),
                   "PROJECTED correction requires an exact coarse solve");
   bool monitor_physical = this->controller || this->print_options.iterations;
#ifdef MFEM_USE_MPI
   if (this->GetComm() != MPI_COMM_NULL)
   {
      int local = monitor_physical ? 1 : 0, global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, this->GetComm());
      monitor_physical = global != 0;
   }
#endif

   const real_t removed = impl->spaces.GetIncompatibleRHSNorm(b);
   const real_t original_rhs_norm = this->Norm(b);
   CheckCollective(std::isfinite(original_rhs_norm) && std::isfinite(removed),
                   "Deflation RHS is not finite");
   const bool compatible = removed <= impl->rhs_options.atol +
                           impl->rhs_options.rtol * original_rhs_norm;
   impl->info.rhs_compatible = compatible;
   impl->info.removed_rhs_norm = removed;
   CheckCollective(compatible ||
                   impl->rhs_options.action == IncompatibleRHSAction::PROJECT,
                   "RHS is incompatible with the registered left null space");

   auto &work = impl->work;
   Vector &b_work = work.b_work, &x0 = work.x0;
   Vector &original_initial = work.original_initial;
   Vector &working_initial = work.working_initial;
   x0.UseDevice(use_device);
   x0.SetSize(this->width);
   impl->spaces.ProjectRHS(b, b_work);
   if (this->iterative_mode) { impl->spaces.ProjectSolution(x, x0); }
   else { x0 = 0.0; }
   impl->Residual(b_work, x0, working_initial);
   impl->Residual(b, x0, original_initial);
   const real_t working_initial_norm = this->Norm(working_initial);
   const real_t original_initial_norm = this->Norm(original_initial);
   const real_t target = std::max(this->abs_tol,
                                  this->rel_tol * working_initial_norm);
   const real_t original_target =
      std::max(this->abs_tol, this->rel_tol * original_initial_norm);
   this->initial_norm = working_initial_norm;
   impl->info.initial_working_residual_norm = working_initial_norm;
   impl->info.initial_original_residual_norm = original_initial_norm;

   Vector &rhs = work.rhs, &internal = work.internal;
   Vector &physical = work.physical, &residual = work.residual;
   Like(internal, x0);
   internal = x0;
   impl->WorkingRHS(b_work, rhs);
   impl->Recover(internal, b_work, physical);
   impl->Residual(b_work, physical, residual);
   real_t physical_norm = this->Norm(residual);
   if (this->print_options.first_and_last)
   {
      mfem::out << "Deflated solver initial ||b_work-Ax|| = "
                << physical_norm << '\n';
   }
   this->final_iter = 0;
   bool stop = CollectiveStop(this->Monitor(0, physical_norm,
                                            residual, physical));
   bool success = physical_norm <= target;
   int iterations = 0;

   if (!stop && !success && this->max_iter > 0 && cg)
   {
      Vector &operator_image = work.operator_image, &r = work.cg_r;
      Vector &z = work.cg_z, &direction = work.direction, &Ad = work.Ad;
      // Return r^T r and r^T z from one global reduction.
      const auto fused_dots = [&](const Vector &res, const Vector &prec_res,
                                  real_t &res_res, real_t &res_prec)
      {
         real_t dots[2] = { res * res, res * prec_res };
#ifdef MFEM_USE_MPI
         if (this->GetComm() != MPI_COMM_NULL)
         {
            MPI_Allreduce(MPI_IN_PLACE, dots, 2, MPITypeMap<real_t>::mpi_type,
                          MPI_SUM, this->GetComm());
         }
#endif
         res_res = dots[0];
         res_prec = dots[1];
      };
      impl->ApplyOperator(internal, operator_image);
      r.UseDevice(rhs.UseDevice());
      r = rhs;
      r -= operator_image;
      impl->ApplyPreconditioner(r, z);
      real_t rho = this->Dot(r, z);
      if (std::isfinite(rho) && rho > 0)
      {
         direction.UseDevice(z.UseDevice());
         direction = z;
         for (int it = 1; it <= this->max_iter; ++it)
         {
            impl->ApplyOperator(direction, Ad);
            const real_t denominator = this->Dot(direction, Ad);
            if (!(denominator > 0) || !std::isfinite(denominator)) { break; }
            const real_t alpha = rho / denominator;
            internal.Add(alpha, direction);
            r.Add(-alpha, Ad);
            iterations = it;
            this->final_iter = iterations;
            // Precondition before the stopping test so ||r|| and r^T z share
            // one reduction; only a converging iteration wastes this work.
            const bool last = it == this->max_iter;
            real_t residual_square = 0, next_rho = 0;
            if (last) { residual_square = this->Dot(r, r); }
            else
            {
               impl->ApplyPreconditioner(r, z);
               fused_dots(r, z, residual_square, next_rho);
            }
            const bool check_physical =
               monitor_physical || last ||
               std::sqrt(std::max(real_t(0), residual_square)) <= target;
            if (check_physical)
            {
               impl->Recover(internal, b_work, physical);
               impl->Residual(b_work, physical, residual);
               physical_norm = this->Norm(residual);
               if (this->print_options.iterations)
               {
                  mfem::out << "Deflated CG iteration " << it
                            << ": ||b_work-Ax|| = " << physical_norm << '\n';
               }
               if (this->controller)
               {
                  stop = CollectiveStop(this->Monitor(it, physical_norm,
                                                       residual, physical));
               }
               success = physical_norm <= target;
            }
            if (stop || success || last) { break; }
            if (!(next_rho > 0) || !std::isfinite(next_rho)) { break; }
            direction *= next_rho / rho;
            direction += z;
            rho = next_rho;
         }
      }
   }

   if (!stop && !success && this->max_iter > 0 && !cg)
   {
      Vector &operator_image = work.operator_image;
      Vector &krylov_rhs = work.krylov_rhs;
      Vector &krylov_residual = work.krylov_residual;
      Vector &cycle_start = work.cycle_start;
      Vector &image = work.image, &w = work.w;
      Vector &coefficients = work.coefficients;
      std::vector<Vector> &v = work.v, &z = work.z;
      DenseMatrix &H = work.H;
      std::vector<real_t> &cs = work.cs, &sn = work.sn, &g = work.g;
      if (static_cast<int>(v.size()) < restart + 1)
      {
         v.resize(restart + 1);
      }
      if (flexible && static_cast<int>(z.size()) < restart)
      {
         z.resize(restart);
      }
      for (Vector &column : v) { column.UseDevice(use_device); }
      if (flexible)
      {
         for (Vector &column : z) { column.UseDevice(use_device); }
      }
      H.SetSize(restart + 1, restart);
      cs.resize(restart);
      sn.resize(restart);
      g.resize(restart + 1);
      while (iterations < this->max_iter && !stop && !success)
      {
         impl->ApplyOperator(internal, operator_image);
         krylov_residual.UseDevice(rhs.UseDevice());
         krylov_residual = rhs;
         krylov_residual -= operator_image;
         if (!flexible)
         {
            impl->ApplyPreconditioner(krylov_residual, krylov_rhs);
            krylov_residual = krylov_rhs;
         }
         const real_t beta = this->Norm(krylov_residual);
         if (!(beta > 0) || !std::isfinite(beta)) { break; }

         const int cycle = std::min(restart, this->max_iter - iterations);
         H = 0.0;
         std::fill(cs.begin(), cs.end(), real_t(0));
         std::fill(sn.begin(), sn.end(), real_t(0));
         std::fill(g.begin(), g.end(), real_t(0));
         g[0] = beta;
         v[0] = krylov_residual;
         v[0] /= beta;
         Like(cycle_start, internal);
         cycle_start = internal;
         const real_t cycle_physical_norm = physical_norm;
         const real_t estimate_target = flexible ? target :
            target * beta / std::max(cycle_physical_norm,
                                     std::numeric_limits<real_t>::min());
         bool breakdown = false;
         int produced = 0;
         int checked_columns = 0;

         const auto update_candidate = [&](int columns)
         {
            if (!BackSubstitute(H, g, columns, coefficients)) { return false; }
            internal = cycle_start;
            for (int i = 0; i < columns; ++i)
            {
               internal.Add(coefficients(i), flexible ? z[i] : v[i]);
            }
            return true;
         };

         for (int j = 0; j < cycle; ++j)
         {
            if (flexible)
            {
               impl->ApplyPreconditioner(v[j], z[j]);
               impl->ApplyOperator(z[j], w);
            }
            else
            {
               impl->ApplyOperator(v[j], image);
               impl->ApplyPreconditioner(image, w);
            }
            const real_t input_norm = this->Norm(w);
            for (int pass = 0; pass < 2; ++pass)
            {
               for (int i = 0; i <= j; ++i)
               {
                  const real_t projection = this->Dot(v[i], w);
                  H(i, j) += projection;
                  w.Add(-projection, v[i]);
               }
            }
            const real_t h_next = this->Norm(w);
            const real_t breakdown_scale = std::max(input_norm,
                                                     std::abs(H(j, j)));
            breakdown = !std::isfinite(h_next) ||
                        h_next <= real_t(32) *
                        std::numeric_limits<real_t>::epsilon() *
                        breakdown_scale;
            H(j + 1, j) = breakdown ? 0 : h_next;
            if (!breakdown)
            {
               v[j + 1] = w;
               v[j + 1] /= h_next;
            }
            for (int i = 0; i < j; ++i)
            {
               const real_t first = H(i, j), second = H(i + 1, j);
               H(i, j) = cs[i] * first + sn[i] * second;
               H(i + 1, j) = -sn[i] * first + cs[i] * second;
            }
            const real_t pivot = std::hypot(H(j, j), H(j + 1, j));
            if (!(pivot > 0) || !std::isfinite(pivot))
            {
               breakdown = true;
               break;
            }
            cs[j] = H(j, j) / pivot;
            sn[j] = H(j + 1, j) / pivot;
            H(j, j) = pivot;
            H(j + 1, j) = 0;
            g[j + 1] = -sn[j] * g[j];
            g[j] *= cs[j];

            produced = j + 1;
            ++iterations;
            this->final_iter = iterations;
            const bool cycle_end = produced == cycle ||
                                   iterations == this->max_iter;
            const bool estimate_ready =
               std::abs(g[j + 1]) <= estimate_target;
            const bool check_physical = monitor_physical || breakdown ||
                                        cycle_end || estimate_ready;
            if (check_physical)
            {
               if (!update_candidate(produced))
               {
                  --iterations;
                  --produced;
                  breakdown = true;
                  break;
               }
               checked_columns = produced;
               impl->Recover(internal, b_work, physical);
               impl->Residual(b_work, physical, residual);
               physical_norm = this->Norm(residual);
               if (this->print_options.iterations)
               {
                  mfem::out << "Deflated " << (flexible ? "FGMRES" : "GMRES")
                            << " iteration " << iterations
                            << ": ||b_work-Ax|| = " << physical_norm << '\n';
               }
               if (this->controller)
               {
                  stop = CollectiveStop(this->Monitor(iterations, physical_norm,
                                                       residual, physical));
               }
               success = physical_norm <= target;
            }
            if (stop || success || breakdown) { break; }
         }
         if (produced > 0 && checked_columns != produced)
         {
            if (!update_candidate(produced)) { breakdown = true; }
            else
            {
               impl->Recover(internal, b_work, physical);
               impl->Residual(b_work, physical, residual);
               physical_norm = this->Norm(residual);
               success = physical_norm <= target;
            }
         }
         if (breakdown || produced == 0) { break; }
      }
   }

   // Recompute after every exit, including zero budget and controller stops.
   impl->Recover(internal, b_work, x);
   impl->Residual(b_work, x, residual);
   const real_t final_working = this->Norm(residual);
   Vector &original_residual = work.original_residual;
   Vector &projected_solution = work.projected_solution;
   Vector &null_component = work.null_component;
   impl->Residual(b, x, original_residual);
   impl->spaces.ProjectSolution(x, projected_solution);
   null_component.UseDevice(x.UseDevice());
   null_component = x;
   null_component -= projected_solution;
   impl->info.valid = true;
   impl->info.iterations = iterations;
   impl->info.final_working_residual_norm = final_working;
   impl->info.final_original_residual_norm = this->Norm(original_residual);
   impl->info.solution_null_component_norm = this->Norm(null_component);
   impl->info.converged_working_system = final_working <= target;
   impl->info.converged_original_system =
      impl->info.final_original_residual_norm <= original_target;
   this->final_iter = iterations;
   this->final_norm = final_working;
   this->converged = impl->info.converged_working_system;
   if (this->print_options.summary || this->print_options.first_and_last ||
       (this->print_options.warnings && !this->converged))
   {
      mfem::out << "Deflated solver: " << iterations << " iterations, "
                << "||b_work-Ax|| = " << final_working
                << (this->converged ? " (converged)" : " (not converged)")
                << '\n';
   }
   this->Monitor(iterations, final_working, residual, x, true);
}

template class DeflatedSolverBase<CGSolver>;
template class DeflatedSolverBase<GMRESSolver>;
template class DeflatedSolverBase<FGMRESSolver>;

DeflatedCGSolver::DeflatedCGSolver() = default;
DeflatedGMRESSolver::DeflatedGMRESSolver() = default;
DeflatedFGMRESSolver::DeflatedFGMRESSolver() = default;
#ifdef MFEM_USE_MPI
DeflatedCGSolver::DeflatedCGSolver(MPI_Comm comm)
   : DeflatedSolverBase<CGSolver>(comm) { }
DeflatedGMRESSolver::DeflatedGMRESSolver(MPI_Comm comm)
   : DeflatedSolverBase<GMRESSolver>(comm) { }
DeflatedFGMRESSolver::DeflatedFGMRESSolver(MPI_Comm comm)
   : DeflatedSolverBase<FGMRESSolver>(comm) { }
#endif

} // namespace mfem
