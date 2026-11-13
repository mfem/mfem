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

SubBlockSumOperator::SubBlockSumOperator(int n,
                                         Array<const Operator *> operators,
                                         std::vector<const Array<int> *> indices_)
   : Operator(n), ops(std::move(operators)), indices(std::move(indices_))
{
   MFEM_VERIFY(ops.Size() == (int)indices.size(),
               "Size mismatch between operators and indices.");
}

void SubBlockSumOperator::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(height);
   y = 0.0;
   for (int j=0; j<ops.Size(); j++)
   {
      if (indices[j] == nullptr) { ops[j]->AddMult(x, y); continue; }
      xs.UseDevice(x.UseDevice());
      ys.UseDevice(y.UseDevice());
      x.GetSubVector(*indices[j], xs);
      ys.SetSize(ops[j]->Height());
      ops[j]->Mult(xs, ys);
      y.AddElementVector(*indices[j], ys);
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

const Operator *GlobalFunctional::ComponentHessian(const MultiVector &x,
                                                   int i) const
{
   MFEM_ABORT("ComponentHessian is not implemented for this functional.");
   return nullptr;
}

const Operator *GlobalFunctional::WeightedSumHessian(const MultiVector &x,
                                                     const Vector &lambda) const
{
   VerifyInputSize(x.GetSizes());
   MFEM_VERIFY(lambda.Size() == height,
               "Expected " << height << " components in lambda, but got " << lambda.Size());
   Array<const Operator *> ops(height);
   int valid_count = 0;
   int last_valid_idx = -1;
   for (int i=0; i<height; i++)
   {
      ops[i] = ComponentHessian(x, i);
      valid_count += (ops[i] != nullptr);
      if (ops[i] != nullptr) { last_valid_idx = i; }
   }
   if (valid_count == 0) { return nullptr; } // linear functional
   if (valid_count == 1 && lambda.HostRead()[last_valid_idx] == 1.0)
   {
      // Return the single non-null Hessian if the weight is 1.0
      return ops[last_valid_idx];
   }
   contracted_hess = std::make_unique<WeightedSumOperator>(ops, lambda);
   return contracted_hess.get();
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

   Array<int> field_offsets(field_sizes.Size() + 1);
   field_offsets[0] = 0;
   for (int b=0; b<field_sizes.Size(); b++)
   {
      field_offsets[b+1] = field_offsets[b] + field_sizes[b];
   }
   term_group.SetSize(functionals.Size());
   for (int j=0; j<functionals.Size(); j++)
   {
      int g = 0;
      while (g < (int)groups.size() && !(groups[g].f == functionals[j] &&
                                         groups[g].subfield_ids == field_ids[j]))
      { g++; }
      if (g == (int)groups.size())
      {
         HessianGroup group{functionals[j], field_ids[j], Array<int>(),
                            field_ids[j].Size() == field_sizes.Size(),
                            Vector(functionals[j]->Height())};
         for (int b=0; b<field_ids[j].Size(); b++)
         {
            const int id = field_ids[j][b];
            group.all_fields = group.all_fields && id == b;
            for (int k=field_offsets[id]; k<field_offsets[id+1]; k++)
            {
               group.indices.Append(k);
            }
         }
         groups.push_back(std::move(group));
      }
      term_group[j] = g;
   }
   unit.SetSize(height);
   component_hess.resize(height);
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

const Operator* WeightedSumFunctional::ComponentHessian(const MultiVector &x,
                                                        int i) const
{
   MFEM_VERIFY(finalized,
               "WeightedSumFunctional is not finalized. Call Finalize() first.");
   real_t *e = unit.HostWrite();
   for (int k=0; k<height; k++) { e[k] = 0.0; }
   e[i] = 1.0;
   return Hessian(x, unit, component_hess[i]);
}

const Operator *WeightedSumFunctional::WeightedSumHessian(const MultiVector &x,
                                                          const Vector &lambda) const
{
   MFEM_VERIFY(finalized,
               "WeightedSumFunctional is not finalized. Call Finalize() first.");
   MFEM_VERIFY(lambda.Size() == height,
               "Expected " << height << " components in lambda, but got "
               << lambda.Size());
   return Hessian(x, lambda, sum_hess);
}

const Operator *WeightedSumFunctional::Hessian(const MultiVector &x,
                                               const Vector &lambda,
                                               std::unique_ptr<Operator> &hess) const
{
   const real_t *l = lambda.HostRead();
   for (auto &group : groups)
   {
      real_t *mu = group.mu.HostWrite();
      for (int k=0; k<group.mu.Size(); k++) { mu[k] = 0.0; }
   }
   for (int j=0; j<functionals.Size(); j++)
   {
      Vector &group_mu = groups[term_group[j]].mu;
      real_t *mu = group_mu.HostReadWrite();
      const int offset = out_offsets[block_row_idx[j]];
      if (weights[j].Size() > 0)
      {
         const real_t *weight = weights[j].HostRead();
         for (int k=0; k<group_mu.Size(); k++) { mu[k] += l[offset] * weight[k]; }
      }
      else
      {
         for (int k=0; k<group_mu.Size(); k++) { mu[k] += l[offset + k]; }
      }
   }

   Array<const Operator *> ops;
   std::vector<const Array<int> *> indices;
   for (const auto &group : groups)
   {
      const real_t *mu = group.mu.HostRead();
      int num_nonzero = 0, last_nonzero = -1;
      for (int k=0; k<group.mu.Size(); k++)
      {
         if (mu[k] != 0.0) { num_nonzero++; last_nonzero = k; }
      }
      if (num_nonzero == 0) { continue; }
      const MultiVector subx = x.GetBlockViews(group.subfield_ids);
      const Operator *h = (num_nonzero == 1 && mu[last_nonzero] == 1.0)
                          ? group.f->ComponentHessian(subx, last_nonzero)
                          : group.f->WeightedSumHessian(subx, group.mu);
      if (h == nullptr) { continue; }
      ops.Append(h);
      indices.push_back(group.all_fields ? nullptr : &group.indices);
   }
   if (ops.Size() == 0) { return nullptr; }
   if (ops.Size() == 1 && indices[0] == nullptr) { return ops[0]; }
   hess = std::make_unique<SubBlockSumOperator>(Width(), ops, indices);
   return hess.get();
}

ConstrainedFunctional::ConstrainedFunctional(std::vector<const Array<int>*>
                                             ess_dofs_,
                                             const GlobalFunctional *f_, bool own_)
   : GlobalFunctional(f_ ? f_->GetFieldSizes() : Array<int>(),
                      f_ ? f_->Height() : 0)
   , f(f_), own(own_), ess_dofs(std::move(ess_dofs_))
   , constrained_hessians(height), wrapped_hessians(height, nullptr)
{
   MFEM_VERIFY(f_ != nullptr, "ConstrainedFunctional: f is nullptr.");
   MFEM_VERIFY((int)ess_dofs.size() == field_sizes.Size(),
               "ConstrainedFunctional: expected " << field_sizes.Size()
               << " essential dof lists, but got " << ess_dofs.size());
#ifdef MFEM_USE_MPI
   if (f->IsParallel()) { SetComm(f->GetComm()); }
#endif
   for (int b=0, offset=0; b<field_sizes.Size(); offset += field_sizes[b++])
   {
      if (ess_dofs[b] == nullptr) { continue; }
      const int *dofs = ess_dofs[b]->HostRead();
      for (int k=0; k<ess_dofs[b]->Size(); k++)
      {
         ess_tdofs.Append(dofs[k] + offset);
      }
   }
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

const Operator* ConstrainedFunctional::ComponentHessian(const MultiVector &x,
                                                        int i) const
{
   return Constrain(f->ComponentHessian(x, i), wrapped_hessians[i],
                    constrained_hessians[i]);
}

const Operator *ConstrainedFunctional::WeightedSumHessian(const MultiVector &x,
                                                          const Vector &lambda) const
{
   return Constrain(f->WeightedSumHessian(x, lambda), wrapped_sum_hessian,
                    constrained_sum_hessian);
}

const Operator *ConstrainedFunctional::Constrain(const Operator *hess,
                                                 const Operator *&wrapped,
                                                 std::unique_ptr<ConstrainedOperator> &constrained) const
{
   if (hess == nullptr || ess_tdofs.Size() == 0) { return hess; }
   if (hess != wrapped)
   {
      // ConstrainedOperator does not call non-const methods.
      constrained = std::make_unique<ConstrainedOperator>(
                       const_cast<Operator*>(hess), ess_tdofs);
      wrapped = hess;
   }
   return constrained.get();
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

const Operator *LinearFunctional::ComponentHessian(const MultiVector &x,
                                                   int i) const
{
   MFEM_VERIFY(i == 0, "LinearFunctional: only one component.");
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

const Operator *QuadraticFunctional::ComponentHessian(const MultiVector &x,
                                                      int i) const
{
   MFEM_VERIFY(i == 0, "QuadraticFunctional: only one component.");
   return A; // Hessian is constant and equal to A
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
      diff = std::max(diff, std::abs(df_d - (fp[i] - fm[i]) / (2.0 * h)));
   }
   return diff;
}

} // namespace optimization

} // namespace mfem
