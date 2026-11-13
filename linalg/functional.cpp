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

#include "functional.hpp"
#include <algorithm>
#include <cmath>

namespace mfem
{

namespace optimization
{

WeightedSumOperator::WeightedSumOperator(Array<const Operator *> operators,
                                         Vector weight)
   : ops(std::move(operators)), w(std::move(weight))
{
   bool isnull = true;
   MFEM_VERIFY(ops.Size() == w.Size(),
               "Size mismatch between operators and weights.");
   for (int i=0; i<ops.Size(); i++)
   {
      if (ops[i] == nullptr) { continue; } // zero block
      if (isnull)
      {
         isnull = false; height = ops[i]->Height(); width = ops[i]->Width();
      }
      else
      {
         MFEM_VERIFY(height == ops[i]->Height() && width == ops[i]->Width(),
                     "Size mismatch between operators.");
      }
   }
   MFEM_VERIFY(!isnull, "All operators are null. Cannot deduce size");
}

void WeightedSumOperator::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(height);
   y = 0.0;
   const real_t *weights = w.HostRead();
   for (int i=0; i<ops.Size(); i++)
   {
      if (ops[i] == nullptr) { continue; }
      ops[i]->AddMult(x, y, weights[i]);
   }
}

void BlockMVOperator::MultMV(const MultiVector &x,
                                      MultiVector &y) const
{
   MFEM_VERIFY(x.NumBlocks() == NumColBlocks() && y.NumBlocks() == NumRowBlocks(),
               "Expected " << NumColBlocks() << " input and " << NumRowBlocks()
               << " output blocks, but got " << x.NumBlocks() << " and "
               << y.NumBlocks());
   mv_tmp.UseDevice(true);
   for (int i=0; i<NumRowBlocks(); i++)
   {
      y[i] = 0.0;
      mv_tmp.SetSize(y[i].Size());
      for (int j=0; j<NumColBlocks(); j++)
      {
         if (IsZeroBlock(i, j) || GetBlockCoef(i, j) == 0.0) { continue; }
         GetBlock(i, j).Mult(x[j], mv_tmp);
         y[i].Add(GetBlockCoef(i, j), mv_tmp);
      }
   }
}

#ifdef MFEM_USE_MPI
GlobalFunctional::GlobalFunctional(MPI_Comm comm_, Array<int> field_sizes_,
                                   int output_size)
   : GlobalFunctional(std::move(field_sizes_), output_size)
{
   comm = comm_;
   int max_output_size = 0;
   MPI_Allreduce(&output_size, &max_output_size, 1, MPI_INT, MPI_MAX, comm);
   MFEM_VERIFY(max_output_size == output_size,
               "Output size should be the same for all ranks in MPI.");
}
#endif

void GlobalFunctional::ComponentDerivative(const MultiVector &x, int i,
                                           MultiVector &df_i) const
{
   MFEM_ABORT("ComponentDerivative is not implemented for this functional.");
}

void GlobalFunctional::Derivative(const MultiVector &x,
                                  Array<MultiVector*> &df) const
{
   MFEM_VERIFY(df.Size() == height,
               "Expected " << height << " components in df, but got " << df.Size());
   for (int i=0; i<height; i++)
   {
      MFEM_VERIFY(df[i] != nullptr, "df[" << i << "] is nullptr.");
      ComponentDerivative(x, i, *df[i]);
   }
}

Operator *GlobalFunctional::ComponentHessian(const MultiVector &x, int k,
                                             int i, int j) const
{
   MFEM_ABORT("ComponentHessian is not implemented for this functional.");
   return nullptr;
}

Operator *GlobalFunctional::ComponentHessian(const MultiVector &x, int k) const
{
   VerifyInputSize(x.GetSizes());
   const int nf = field_sizes.Size();
   Array<Operator *> blocks(nf*nf);
   for (int i=0; i<nf; i++)
   {
      for (int j=0; j<nf; j++) { blocks[i*nf + j] = ComponentHessian(x, k, i, j); }
   }
   if ((int)full_hess.size() != height) { full_hess.resize(height); }
   return CollectBlocks(blocks, full_hess[k]);
}

Operator *GlobalFunctional::WeightedSumHessian(const MultiVector &x,
                                               const Vector &lambda,
                                               int i, int j) const
{
   VerifyInputSize(x.GetSizes());
   MFEM_VERIFY(lambda.Size() == height,
               "Expected " << height << " components in lambda, but got " << lambda.Size());
   Array<const Operator *> ops(height);
   Operator *last_valid = nullptr;
   int valid_count = 0;
   int last_valid_idx = -1;
   for (int k=0; k<height; k++)
   {
      Operator *op = ComponentHessian(x, k, i, j);
      ops[k] = op;
      valid_count += (op != nullptr);
      if (op != nullptr) { last_valid_idx = k; last_valid = op; }
   }
   if (valid_count == 0) { return nullptr; } // zero block
   if (valid_count == 1 && lambda.HostRead()[last_valid_idx] == 1.0)
   {
      // Return the single non-null Hessian if the weight is 1.0
      return last_valid;
   }
   const int nf = field_sizes.Size();
   if ((int)block_sum_hess.size() != nf*nf) { block_sum_hess.resize(nf*nf); }
   block_sum_hess[i*nf + j] = std::make_unique<WeightedSumOperator>(ops, lambda);
   return block_sum_hess[i*nf + j].get();
}

Operator *GlobalFunctional::WeightedSumHessian(const MultiVector &x,
                                               const Vector &lambda) const
{
   VerifyInputSize(x.GetSizes());
   const int nf = field_sizes.Size();
   Array<Operator *> blocks(nf*nf);
   for (int i=0; i<nf; i++)
   {
      for (int j=0; j<nf; j++)
      {
         blocks[i*nf + j] = WeightedSumHessian(x, lambda, i, j);
      }
   }
   return CollectBlocks(blocks, full_sum_hess);
}

Operator *GlobalFunctional::CollectBlocks(const Array<Operator *> &blocks,
                                          std::unique_ptr<BlockMVOperator> &hess) const
{
   const int nf = field_sizes.Size();
   bool all_null = true;
   for (Operator *block : blocks) { all_null = all_null && block == nullptr; }
   if (all_null) { return nullptr; }

   Array<int> offsets(nf + 1);
   offsets[0] = 0;
   for (int b=0; b<nf; b++) { offsets[b+1] = offsets[b] + field_sizes[b]; }
   if (!(offsets == hess_offsets))
   {
      // The block operators refer to hess_offsets.
      for (auto &h : full_hess) { h.reset(); }
      full_sum_hess.reset();
      hess_offsets = offsets;
   }

   bool same = hess != nullptr;
   for (int i=0; same && i<nf; i++)
   {
      for (int j=0; same && j<nf; j++)
      {
         Operator *block = blocks[i*nf + j];
         same = hess->IsZeroBlock(i, j) ? block == nullptr
                : block == &hess->GetBlock(i, j);
      }
   }
   if (same) { return hess.get(); }

   hess = std::make_unique<BlockMVOperator>(hess_offsets);
   for (int i=0; i<nf; i++)
   {
      for (int j=0; j<nf; j++)
      {
         if (blocks[i*nf + j]) { hess->SetBlock(i, j, blocks[i*nf + j]); }
      }
   }
   return hess.get();
}

bool GlobalFunctional::IsParallel() const
{
#ifdef MFEM_USE_MPI
   return comm != MPI_COMM_NULL;
#else
   return false;
#endif
}

void GlobalFunctional::VerifyInputSize(const Array<int> &sizes) const
{
   MFEM_VERIFY(sizes.Size() == field_sizes.Size(),
               "Expected " << field_sizes.Size() << " fields in input, but got "
               << sizes.Size());
   for (int i=0; i<field_sizes.Size(); i++)
   {
      MFEM_VERIFY(sizes[i] == field_sizes[i],
                  "Expected field " << i << " to have size " << field_sizes[i]
                  << ", but got " << sizes[i]);
   }
}

void GlobalFunctional::Reduce(Vector &vec) const
{
#ifdef MFEM_USE_MPI
   if (IsParallel())
   {
      MPI_Allreduce(MPI_IN_PLACE, vec.HostReadWrite(), vec.Size(),
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   }
#endif
}

real_t GlobalFunctional::Reduce(real_t value) const
{
#ifdef MFEM_USE_MPI
   if (IsParallel())
   {
      MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
   }
#endif
   return value;
}

real_t GlobalFunctional::Dot(const Vector &x, const Vector &y) const
{
   return Reduce(x*y);
}

WeightedSumFunctional::~WeightedSumFunctional()
{
   for (auto *f : owned_functionals) { delete f; }
}

void WeightedSumFunctional::Add(int blk_row, Array<int> subfield_ids,
                                Vector weight,
                                const GlobalFunctional *f, bool own)
{
   MFEM_VERIFY(f != nullptr, "Functional pointer is null.");
   MFEM_VERIFY(!finalized, "WeightedSumFunctional is already finalized.");
   MFEM_VERIFY(blk_row >= 0, "Row index must be nonnegative.");
   Array<int> sizes;
   for (auto id : subfield_ids)
   {
      MFEM_VERIFY(id >= 0 && id < field_sizes.Size(),
                  "Field id " << id << " is out of bounds.");
      sizes.Append(field_sizes[id]);
   }
   f->VerifyInputSize(sizes);
   // A functional added more than once is owned once.
   if (own && owned_functionals.Find(f) < 0) { owned_functionals.Append(f); }
   // A weighted functional adds one row, sum_k w_k f_k.
   MFEM_VERIFY(weight.Size() == 0 || weight.Size() == f->Height(),
               "Expected " << f->Height() << " weights, but got " << weight.Size());
   // A functional without outputs contributes nothing.
   if (f->Height() == 0) { return; }
   const int row_size = weight.Size() == 0 ? f->Height() : 1;
   if (blk_row + 2 > out_offsets.Size())
   {
      const int old_size = out_offsets.Size();
      out_offsets.SetSize(blk_row + 2);
      for (int r = old_size; r < blk_row + 2; r++) { out_offsets[r] = 0; }
   }
   if (out_offsets[blk_row+1] == 0) { out_offsets[blk_row+1] = row_size; }
   MFEM_VERIFY(out_offsets[blk_row+1] == row_size,
               "Output size mismatch for row " << blk_row << ": expected "
               << out_offsets[blk_row+1] << ", but got " << row_size);

   block_row_idx.Append(blk_row);
   field_ids.push_back(subfield_ids);
   weights.push_back(weight);
   functionals.Append(f);
}

void WeightedSumFunctional::Finalize()
{
   MFEM_VERIFY(!finalized, "WeightedSumFunctional is already finalized.");
   out_offsets.PartialSum();
   height = out_offsets.Last();
   work.SetSizes(field_sizes);

   const int nf = field_sizes.Size();
   unit.SetSize(height);
   component_hess.resize(height*nf*nf);
   sum_hess.resize(nf*nf);
   finalized = true;
}

void WeightedSumFunctional::Eval(const MultiVector &x, Vector &fx) const
{
   MFEM_VERIFY(finalized,
               "WeightedSumFunctional is not finalized. Call Finalize() first.");
   fx.SetSize(height);
   real_t *out = fx.HostWrite();
   for (int k=0; k<height; k++) { out[k] = 0.0; }
   for (int i=0; i<block_row_idx.Size(); i++)
   {
      const int offset = out_offsets[block_row_idx[i]];
      const auto &subfield_ids = field_ids[i];
      const auto *f = functionals[i];

      const MultiVector subx = x.GetBlockViews(subfield_ids);
      Vector subfx(f->Height());
      f->Eval(subx, subfx);

      const real_t *sub = subfx.HostRead();
      if (weights[i].Size() == 0)
      {
         for (int k=0; k<subfx.Size(); k++) { out[offset + k] += sub[k]; }
      }
      else
      {
         const real_t *weight = weights[i].HostRead();
         for (int k=0; k<subfx.Size(); k++) { out[offset] += weight[k] * sub[k]; }
      }
   }
}

void WeightedSumFunctional::ComponentDerivative(const MultiVector &x, int i,
                                                MultiVector &df_i) const
{
   MFEM_VERIFY(finalized,
               "WeightedSumFunctional is not finalized. Call Finalize() first.");
   for (int b=0; b<df_i.NumBlocks(); b++) { df_i[b] = 0.0; }
   for (int j=0; j<block_row_idx.Size(); j++)
   {
      const auto &subfield_ids = field_ids[j];
      const auto &weight = weights[j];
      const auto *f = functionals[j];

      const int offset = out_offsets[block_row_idx[j]];
      const bool weighted = weight.Size() > 0;
      const int row_size = weighted ? 1 : f->Height();
      if (i < offset || i >= offset + row_size) { continue; } // irrelevant component

      const MultiVector subx = x.GetBlockViews(subfield_ids);
      MultiVector subdf = work.GetBlockViews(subfield_ids);
      // Weighted: d/dx sum_k w_k f_k; otherwise: d f_{i - offset}/dx.
      const int k_begin = weighted ? 0 : i - offset;
      const int k_end = weighted ? f->Height() : k_begin + 1;
      for (int k=k_begin; k<k_end; k++)
      {
         const real_t w = weighted ? weight.HostRead()[k] : 1.0;
         if (w == 0.0) { continue; }
         f->ComponentDerivative(subx, k, subdf);
         for (int b=0; b<subfield_ids.Size(); b++)
         {
            df_i[subfield_ids[b]].Add(w, subdf[b]);
         }
      }
   }
}

Operator *WeightedSumFunctional::ComponentHessian(const MultiVector &x, int k,
                                                  int i, int j) const
{
   MFEM_VERIFY(finalized,
               "WeightedSumFunctional is not finalized. Call Finalize() first.");
   real_t *e = unit.HostWrite();
   for (int r=0; r<height; r++) { e[r] = 0.0; }
   e[k] = 1.0;
   const int nf = field_sizes.Size();
   return HessianBlock(x, unit, i, j, component_hess[(k*nf + i)*nf + j]);
}

Operator *WeightedSumFunctional::WeightedSumHessian(const MultiVector &x,
                                                    const Vector &lambda,
                                                    int i, int j) const
{
   MFEM_VERIFY(finalized,
               "WeightedSumFunctional is not finalized. Call Finalize() first.");
   MFEM_VERIFY(lambda.Size() == height,
               "Expected " << height << " components in lambda, but got "
               << lambda.Size());
   return HessianBlock(x, lambda, i, j, sum_hess[i*field_sizes.Size() + j]);
}

Operator *WeightedSumFunctional::HessianBlock(const MultiVector &x,
                                              const Vector &lambda,
                                              int i, int j,
                                              std::unique_ptr<Operator> &hess) const
{
   const real_t *l = lambda.HostRead();
   Array<const Operator *> ops;
   Array<real_t> coefs;
   Operator *first = nullptr;
   for (int t=0; t<functionals.Size(); t++)
   {
      // Fields i and j as fields a and b of the functional.
      const int a = field_ids[t].Find(i), b = field_ids[t].Find(j);
      if (a < 0 || b < 0) { continue; }
      const GlobalFunctional *f = functionals[t];
      const int offset = out_offsets[block_row_idx[t]];
      const bool weighted = weights[t].Size() > 0;
      const real_t *weight = weighted ? weights[t].HostRead() : nullptr;
      const MultiVector subx = x.GetBlockViews(field_ids[t]);
      for (int c=0; c<f->Height(); c++)
      {
         const real_t coef = weighted ? l[offset] * weight[c] : l[offset + c];
         if (coef == 0.0) { continue; }
         Operator *h = f->ComponentHessian(subx, c, a, b);
         if (h == nullptr) { continue; }
         if (ops.Size() == 0) { first = h; }
         ops.Append(h);
         coefs.Append(coef);
      }
   }
   if (ops.Size() == 0) { return nullptr; }
   if (ops.Size() == 1 && coefs[0] == 1.0) { return first; }
   Vector w(coefs.Size());
   real_t *wp = w.HostWrite();
   for (int n=0; n<coefs.Size(); n++) { wp[n] = coefs[n]; }
   hess = std::make_unique<WeightedSumOperator>(ops, w);
   return hess.get();
}

ConstrainedFunctional::ConstrainedFunctional(std::vector<const Array<int>*>
                                             ess_dofs_,
                                             const GlobalFunctional *f_, bool own_)
   : GlobalFunctional(f_ ? f_->GetFieldSizes() : Array<int>(),
                      f_ ? f_->Height() : 0)
   , f(f_), own(own_), ess_dofs(std::move(ess_dofs_))
{
   MFEM_VERIFY(f_ != nullptr, "ConstrainedFunctional: f is nullptr.");
   MFEM_VERIFY((int)ess_dofs.size() == field_sizes.Size(),
               "ConstrainedFunctional: expected " << field_sizes.Size()
               << " essential dof lists, but got " << ess_dofs.size());
#ifdef MFEM_USE_MPI
   if (f->IsParallel()) { SetComm(f->GetComm()); }
#endif
   const int nf = field_sizes.Size();
   constrained_hessians.resize((height + 1)*nf*nf);
   wrapped_hessians.resize((height + 1)*nf*nf, nullptr);
}

void ConstrainedFunctional::ComponentDerivative(const MultiVector &x, int i,
                                                MultiVector &df_i) const
{
   f->ComponentDerivative(x, i, df_i);
   for (int b=0; b<df_i.NumBlocks(); b++)
   {
      if (ess_dofs[b] != nullptr && ess_dofs[b]->Size() > 0)
      { df_i[b].SetSubVector(*ess_dofs[b], 0.0); }
   }
}

Operator *ConstrainedFunctional::ComponentHessian(const MultiVector &x, int k,
                                                  int i, int j) const
{
   const int nf = field_sizes.Size();
   return Constrain(f->ComponentHessian(x, k, i, j), i, j, (k*nf + i)*nf + j);
}

Operator *ConstrainedFunctional::WeightedSumHessian(const MultiVector &x,
                                                    const Vector &lambda,
                                                    int i, int j) const
{
   const int nf = field_sizes.Size();
   return Constrain(f->WeightedSumHessian(x, lambda, i, j), i, j,
                    (height*nf + i)*nf + j);
}

Operator *ConstrainedFunctional::Constrain(Operator *hess, int i, int j,
                                           int slot) const
{
   const Array<int> &test_dofs = ess_dofs[i] ? *ess_dofs[i] : no_dofs;
   const Array<int> &trial_dofs = ess_dofs[j] ? *ess_dofs[j] : no_dofs;
   if (hess == nullptr || (test_dofs.Size() == 0 && trial_dofs.Size() == 0))
   {
      return hess;
   }
   if (hess != wrapped_hessians[slot])
   {
      if (i == j)
      {
         constrained_hessians[slot] =
            std::make_unique<ConstrainedOperator>(hess, test_dofs);
      }
      else
      {
         constrained_hessians[slot] =
            std::make_unique<RectangularConstrainedOperator>(hess, trial_dofs,
                                                             test_dofs);
      }
      wrapped_hessians[slot] = hess;
   }
   return constrained_hessians[slot].get();
}

LinearFunctional::LinearFunctional(const Vector *b_, bool own_)
   : GlobalFunctional(Array<int>({b_ ? b_->Size() : 0}), 1), b(b_), own(own_)
{
   MFEM_VERIFY(b_ != nullptr, "LinearFunctional: b is nullptr.");
}

#ifdef MFEM_USE_MPI
LinearFunctional::LinearFunctional(MPI_Comm comm, const Vector *b_, bool own_)
   : LinearFunctional(b_, own_)
{
   SetComm(comm);
}
#endif

void LinearFunctional::Eval(const MultiVector &x, Vector &fx) const
{
   MFEM_VERIFY(x.NumBlocks() == 1 && x[0].Size() == b->Size(),
               "LinearFunctional: input size mismatch.");
   fx.SetSize(1);
   fx.HostWrite()[0] = Dot(*b, x[0]);
}

void LinearFunctional::ComponentDerivative(const MultiVector &x, int i,
                                           MultiVector &df_i) const
{
   MFEM_VERIFY(i == 0, "LinearFunctional: only one component.");
   MFEM_VERIFY(x.NumBlocks() == 1 && x[0].Size() == b->Size(),
               "LinearFunctional: input size mismatch.");
   MFEM_VERIFY(df_i.NumBlocks() == 1 && df_i[0].Size() == b->Size(),
               "LinearFunctional: output size mismatch.");
   df_i[0] = *b;
}

Operator *LinearFunctional::ComponentHessian(const MultiVector &x, int k,
                                             int i, int j) const
{
   MFEM_VERIFY(k == 0, "LinearFunctional: only one component.");
   return nullptr; // linear functional has no Hessian
}

QuadraticFunctional::QuadraticFunctional(const Operator *A_, const Vector *b_,
                                         bool own_, bool own_b_)
   : GlobalFunctional(Array<int>({A_ ? A_->Width() : 0}), 1), A(A_), own(own_)
, b(b_), own_b(own_b_)
{
   MFEM_VERIFY(A_ != nullptr, "QuadraticFunctional: A is nullptr.");
   MFEM_VERIFY(A_->Height() == A_->Width(),
               "QuadraticFunctional: A must be square.");
}

#ifdef MFEM_USE_MPI
QuadraticFunctional::QuadraticFunctional(MPI_Comm comm, const Operator *A_,
                                         const Vector *b_,
                                         bool own_, bool own_b_)
   : QuadraticFunctional(A_, b_, own_, own_b_)
{
   SetComm(comm);
}
#endif

QuadraticFunctional::~QuadraticFunctional()
{
   if (own) { delete A; }
   if (own_b) { delete b; }
}

void QuadraticFunctional::Eval(const MultiVector &x, Vector &fx) const
{
   MFEM_VERIFY(x.NumBlocks() == 1 && x[0].Size() == A->Width(),
               "QuadraticFunctional: input size mismatch.");
   work.UseDevice(x[0].UseDevice());
   work.SetSize(A->Height());
   A->Mult(x[0], work);
   if (b) { work.Add(-2.0, *b); }
   fx.SetSize(1);
   fx.HostWrite()[0] = 0.5 * Dot(x[0], work);
}

void QuadraticFunctional::ComponentDerivative(const MultiVector &x, int i,
                                              MultiVector &df_i) const
{
   MFEM_VERIFY(i == 0, "QuadraticFunctional: only one component.");
   MFEM_VERIFY(x.NumBlocks() == 1 && x[0].Size() == A->Width(),
               "QuadraticFunctional: input size mismatch.");
   df_i[0].SetSize(A->Width());
   A->Mult(x[0], df_i[0]);
   if (b) { df_i[0] -= *b; }
}

Operator *QuadraticFunctional::ComponentHessian(const MultiVector &x, int k,
                                                int i, int j) const
{
   MFEM_VERIFY(k == 0, "QuadraticFunctional: only one component.");
   return const_cast<Operator*>(A); // Hessian is constant and equal to A
}

real_t DiffDerivativeAgainstCentralDiff(const GlobalFunctional &f,
                                        const MultiVector &x,
                                        const MultiVector &d, real_t h)
{
   const Array<int> sizes = x.GetSizes();
   f.VerifyInputSize(sizes);
   f.VerifyInputSize(d.GetSizes());
   const int num_blocks = sizes.Size();

   MultiVector xp(sizes), xm(sizes), df(sizes);
   for (int b=0; b<num_blocks; b++)
   {
      xp[b] = x[b]; xp[b].Add(h, d[b]);
      xm[b] = x[b]; xm[b].Add(-h, d[b]);
   }
   Vector fp, fm;
   f.Eval(xp, fp);
   f.Eval(xm, fm);

   real_t diff = 0.0;
   for (int i=0; i<f.Height(); i++)
   {
      f.ComponentDerivative(x, i, df);
      real_t df_d = 0.0;
      for (int b=0; b<num_blocks; b++) { df_d += df[b] * d[b]; }
      df_d = f.Reduce(df_d);
      diff = std::max(diff, std::abs(df_d - (fp[i] - fm[i]) / (2 * h)));
   }
   return diff;
}

} // namespace optimization

} // namespace mfem
