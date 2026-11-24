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

#include "optproblem.hpp"
#include <utility>

namespace mfem
{

namespace
{
/// Return a shared pointer to a deep copy of @a v (@a copy = true), or a
/// non-owning shared pointer to @a v itself (@a copy = false).
template <typename T>
std::shared_ptr<const T> ShareOrCopy(const T &v, bool copy)
{
   if (copy) { return std::make_shared<const T>(v); }
   return std::shared_ptr<const T>(&v, [](const T *) {});
}

/// Return a sub-multivector of X with blocks in ids.
/// If copy is true, the blocks are copied to Xsub. Otherwise, Xsub references the blocks in X.
void SubMultiVector(const MultiVector &X, const Array<int> &ids,
                    MultiVector &Xsub, bool copy)
{
   MFEM_VERIFY(ids.Size() == 0 || (ids.Min() >= 0 && ids.Max() < X.NumBlocks()),
               "Field index out of range.");
   Xsub.SetNumBlocks(ids.Size());
   for (int i=0; i<ids.Size(); i++)
   {
      int id = ids[i];
      if (copy) { Xsub[i] = X[id]; }
      else { Xsub.MakeRef(i, X[id]); }
   }
}

void SubMultiVector(MultiVector &X, const Array<int> &ids,
                    MultiVector &Xsub, bool copy)
{
   MFEM_VERIFY(ids.Size() == 0 || (ids.Min() >= 0 && ids.Max() < X.NumBlocks()),
               "Field index out of range.");
   Xsub.SetNumBlocks(ids.Size());
   for (int i=0; i<ids.Size(); i++)
   {
      int id = ids[i];
      if (copy) { Xsub[i] = X[id]; }
      else { Xsub.MakeRef(i, X[id]); }
   }
}

/// Scatter the sub-vector xsub without essential dofs to the vector x.
/// ess_dofs must be sorted in ascending order. Host-only
void ScatterVector(const Vector &xsub, const Array<int> &ess_dofs, Vector &x)
{
   if (ess_dofs.Size() == 0) { x = xsub; return; }
   const real_t *xsub_data = xsub.HostRead();
   // essential entries of x are kept, so read-write, not write-only
   real_t *x_data = x.HostReadWrite();
   const int* ess = ess_dofs.HostRead();
   int sub_i = 0;
   for (int i=0; i<ess_dofs.Size() + 1; i++)
   {
      int start = (i == 0) ? 0 : ess[i-1] + 1;
      int end = (i == ess_dofs.Size()) ? x.Size() : ess[i];
      MFEM_ASSERT(start <= end, "Essential dofs must be sorted in ascending order.");
      for (int j=start; j<end; j++) { x_data[j] = xsub_data[sub_i++]; }
   }
}

/// Gather the vector x to the sub-vector xsub without essential dofs.
/// ess_dofs must be sorted in ascending order. Host-only
void GatherVector(const Vector &x, const Array<int> &ess_dofs, Vector &xsub)
{
   if (ess_dofs.Size() == 0) { xsub = x; return; }
   const real_t *x_data = x.HostRead();
   real_t *xsub_data = xsub.HostWrite();
   const int* ess = ess_dofs.HostRead();
   int sub_i = 0;
   for (int i=0; i<ess_dofs.Size() + 1; i++)
   {
      int start = (i == 0) ? 0 : ess[i-1] + 1;
      int end = (i == ess_dofs.Size()) ? x.Size() : ess[i];
      MFEM_ASSERT(start <= end, "Essential dofs must be sorted in ascending order.");
      for (int j=start; j<end; j++) { xsub_data[sub_i++] = x_data[j]; }
   }
}

} // anonymous namespace

int OptProblem::RegisterField(int size)
{
   finalized = false;
   field_sizes.Append(size);
   field_ess_dofs.push_back(std::make_shared<const Array<int>>());
   field_lower_bounds.push_back(nullptr);
   field_upper_bounds.push_back(nullptr);
   return field_sizes.Size() - 1;
}

int OptProblem::RegisterField(int size, const Array<int> &ess_dofs,
                              bool copy_ess)
{
   finalized = false;
   int id = RegisterField(0);
   UpdateFieldSize(id, size, ess_dofs, copy_ess);
   return id;
}

void OptProblem::UpdateFieldSize(int field_id, int new_size)
{
   finalized = false;
   MFEM_VERIFY(field_id >= 0 && field_id < field_sizes.Size(),
               "Field index out of range.");
   MFEM_VERIFY(new_size >= 0, "Field size must be non-negative.");
   field_sizes[field_id] = new_size;
   field_ess_dofs[field_id] = std::make_shared<const Array<int>>();
   // bounds of the old size are no longer valid
   field_lower_bounds[field_id] = nullptr;
   field_upper_bounds[field_id] = nullptr;
}

void OptProblem::UpdateFieldSize(int field_id, int new_size,
                                 const Array<int> &new_ess_dofs, bool copy_ess)
{
   finalized = false;
   const int n_ess = new_ess_dofs.Size();
   MFEM_VERIFY(n_ess == 0
               || (new_ess_dofs.Min() >= 0 && new_ess_dofs.Max() < new_size),
               "Essential dofs must be in the range [0, " << new_size << ").");
   // scatter/gather of essential dofs rely on this
   const int *ess = new_ess_dofs.HostRead();
   for (int i = 1; i < n_ess; i++)
   {
      MFEM_VERIFY(ess[i-1] < ess[i],
                  "Essential dofs must be sorted in ascending order and unique.");
   }
   UpdateFieldSize(field_id, new_size);
   field_ess_dofs[field_id] = ShareOrCopy(new_ess_dofs, copy_ess);
}

void OptProblem::SetFieldBounds(int field_id, const Vector &lower,
                                const Vector &upper,
                                bool copy_bounds)
{
   finalized = false;
   MFEM_VERIFY(field_id >= 0 && field_id < field_sizes.Size(),
               "Field index out of range.");
   MFEM_VERIFY(lower.Size() == field_sizes[field_id]
               && lower.Size() == upper.Size(),
               "Lower and upper bounds must have the same size as the field.");
   field_lower_bounds[field_id] = ShareOrCopy(lower, copy_bounds);
   field_upper_bounds[field_id] = ShareOrCopy(upper, copy_bounds);
}

int OptProblem::RegisterFunctional(FunctionalRole role,
                                   const Array<int> &field_ids,
                                   Functional *f, bool take_ownership)
{
   finalized = false;
   int n_fields = field_sizes.Size();

   MFEM_VERIFY(f, "Functional must not be null.");
   MFEM_VERIFY(field_ids.Size() == 0
               || (field_ids.Min() >= 0 && field_ids.Max() < n_fields),
               "Field ids must be in the range [0, " << n_fields << ").");
   int width=0;
   for (auto id : field_ids) { width += field_sizes[id]; }
   MFEM_VERIFY(f->Width() == width,
               "Functional width " << f->Width()
               << " does not match the total size of the input fields "
               << width << ".");

   if (take_ownership) { funcs.emplace_back(f); }
   else { funcs.emplace_back(f, [](Functional *) {}); }
   func_input_ids.push_back(field_ids);
   funcs_roles.push_back(std::move(role));
   return (int)funcs.size() - 1;
}

void OptProblem::ChangeRole(int func_id, FunctionalRole role)
{
   finalized = false;
   MFEM_VERIFY(func_id >= 0 && func_id < (int)funcs_roles.size(),
               "Functional index out of range.");
   funcs_roles[func_id] = std::move(role);
}

void OptProblem::Finalize()
{
   if (finalized) { return; }
   int n_funcs = static_cast<int>(funcs.size());
   constraints.SetSize(n_funcs);
   constraint_ids.SetSize(n_funcs);
   lower.SetSize(n_funcs);
   upper.SetSize(n_funcs);
   num_eq = 0;
   int num_ineq = 0;
   obj_id = -1;
   for (int i=0; i<n_funcs; i++)
   {
      const FunctionalRole &role = funcs_roles[i];
      const Functional *f = funcs[i].get();
      MFEM_VERIFY(f, "Functional must not be null.");
      const int width = InputWidth(i);
      MFEM_VERIFY(f->Width() == width, "Functional width " << f->Width()
                  << " does not match the total size of the input fields "
                  << width << ".");
      if (role.IsObjective())
      {
         MFEM_VERIFY(obj_id == -1, "Multiple objectives are not supported.");
         obj_id = i;
      }
      else
      {
         // equalities from the beginning, inequalities from the end
         int id = role.IsEquality() ? num_eq++ : n_funcs - ++num_ineq;
         constraints[id] = funcs[i].get();
         constraint_ids[id] = i;
         lower[id] = role.GetLower();
         upper[id] = role.GetUpper();
      }
   }
   // reverse the inequality constraints order
   std::reverse(constraints.begin() + num_eq, constraints.end());
   std::reverse(constraint_ids.begin() + num_eq, constraint_ids.end());
   std::reverse(lower.begin() + num_eq, lower.end());
   std::reverse(upper.begin() + num_eq, upper.end());
   // drop the gap left by the objective, if any
   constraints.SetSize(num_eq + num_ineq);
   constraint_ids.SetSize(num_eq + num_ineq);
   lower.SetSize(num_eq + num_ineq);
   upper.SetSize(num_eq + num_ineq);

   // clear cached state-less derivatives
   func_derivatives.clear();
   func_derivatives.resize(funcs.size());

   finalized = true;
}

const Vector &OptProblem::GetConstraintsLowerBounds() const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before getting lower bounds.");
   return lower;
}

const Vector &OptProblem::GetConstraintsUpperBounds() const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before getting upper bounds.");
   return upper;
}

int OptProblem::InputWidth(int i) const
{
   int width = 0;
   for (auto id : func_input_ids[i]) { width += field_sizes[id]; }
   return width;
}

real_t OptProblem::Eval(int i, const MultiVector &X) const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before evaluating functionals.");
   MFEM_VERIFY(i >= 0 && i < (int)funcs.size(),
               "Functional index out of range.");
   MultiVector Xsub;
   SubMultiVector(X, func_input_ids[i], Xsub, false);
   return funcs[i]->Eval(Xsub);
}

void OptProblem::EvalDerivative(int i, const MultiVector &X,
                                MultiVector &dfdX) const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before evaluating functionals.");
   MFEM_VERIFY(i >= 0 && i < (int)funcs.size(),
               "Functional index out of range.");
   const Array<int> &ids = func_input_ids[i];
   dfdX.SetNumBlocks(field_sizes.Size());
   for (int j = 0; j < field_sizes.Size(); j++)
   {
      dfdX[j].SetSize(field_sizes[j]);
      dfdX[j].UseDevice(true);
      if (ids.Find(j) == -1) { dfdX[j] = 0.0; }
   }
   MultiVector dfdXin;
   SubMultiVector(dfdX, ids, dfdXin, false);
   EvalDerivativeCompact(i, X, dfdXin);
}

void OptProblem::EvalDerivativeCompact(int i, const MultiVector &X,
                                       MultiVector &dfdXin) const
{
   MultiVector Xin;
   SubMultiVector(X, func_input_ids[i], Xin, false);
   if (!func_derivatives[i])
   { func_derivatives[i] = funcs[i]->Derivative(); }
   func_derivatives[i]->MultMV(Xin, dfdXin);
}

real_t OptProblem::EvalObjective(const MultiVector &X) const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before evaluating the objective.");
   MFEM_VERIFY(obj_id >= 0 && obj_id < (int)funcs.size(),
               "Objective functional is not registered.");
   return Eval(obj_id, X);
}

void OptProblem::EvalObjectiveDerivative(const MultiVector &X,
                                         MultiVector &dfdX) const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before evaluating the objective.");
   MFEM_VERIFY(obj_id >= 0 && obj_id < (int)funcs.size(),
               "Objective functional is not registered.");
   EvalDerivative(obj_id, X, dfdX);
}

void OptProblem::EvalConstraintJacobianTranspose(const MultiVector &X,
                                                 DenseMatrix &JT, int start, int end) const
{
   if (end == -1) { end = constraints.Size(); }
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before evaluating constraints.");
   MFEM_VERIFY(start >= 0 && end <= constraints.Size() && start <= end,
               "Invalid range for constraints: [" << start << ", " << end << ").");
   const int total_size = field_sizes.Sum();
   const int num_constraints = end - start;
   JT.SetSize(total_size, num_constraints);
   Memory<real_t> &mem = JT.GetMemory();
   mem.UseDevice(true);
   // columns as device-safe aliases of jt's memory (GetColumnReference is
   // host only)
   Vector jt_flat, col;
   jt_flat.NewMemoryAndSize(mem, JT.TotalSize(), false);
   for (int i=start; i<end; i++)
   {
      const int cid = constraint_ids[i];
      col.MakeRef(jt_flat, (i - start)*total_size, total_size);
      MultiVector dcdX;
      dcdX.MakeRef(col, field_sizes);
      EvalDerivative(cid, X, dcdX);
      // propagate memory validity flags: blocks -> column -> JT
      for (int j = 0; j < dcdX.NumBlocks(); j++) { dcdX[j].SyncAliasMemory(col); }
      col.SyncAliasMemory(jt_flat);
   }
   mem.Sync(jt_flat.GetMemory());
}

const Array<const Functional*> &OptProblem::GetConstraints() const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before getting constraints.");
   return constraints;
}

void OptProblem::EvalConstraintDerivative(int i, const MultiVector &X,
                                          MultiVector &dc_dX) const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before evaluating constraints.");
   MFEM_VERIFY(i >= 0 && i < constraints.Size(),
               "Constraint index out of range.");
   EvalDerivative(constraint_ids[i], X, dc_dX);
}

SubOptProblem OptProblem::MakeSubProblem(const Array<int> &active_field_ids,
                                         const Array<int> &active_func_ids,
                                         const MultiVector *X, bool copy_X,
                                         bool remove_ess_dofs) const
{
   return SubOptProblem(*this, active_field_ids, active_func_ids, X, copy_X,
                        remove_ess_dofs);
}

SubOptProblem::SubOptProblem(const OptProblem &parent,
                             const Array<int> &active_field_ids,
                             const Array<int> &active_func_ids,
                             const MultiVector *X0_, bool copy_X,
                             bool remove_ess_dofs_)
   : parent_field_ids(active_field_ids),
     parent_field_sizes(parent.field_sizes),
     parent_ess_dofs(parent.field_ess_dofs),
     remove_ess_dofs(remove_ess_dofs_)
{
   const int np = parent.field_sizes.Size();
   const int nf = static_cast<int>(parent.funcs.size());

   // X0: deep copy (MultiVector's copy would copy block pointers) or share
   if (X0_)
   {
      MFEM_VERIFY(X0_->NumBlocks() == np,
                  "X0 must have one block per parent field.");
      if (copy_X)
      {
         auto Xc = std::make_shared<MultiVector>(np);
         for (int p = 0; p < np; p++) { (*Xc)[p] = (*X0_)[p]; }
         X0 = Xc;
      }
      else { X0 = ShareOrCopy(*X0_, false); }
      for (int p = 0; p < np; p++)
      {
         MFEM_VERIFY((*X0)[p].Size() == parent_field_sizes[p],
                     "X0 block " << p << " has size " << (*X0)[p].Size()
                     << ", expected " << parent_field_sizes[p] << ".");
      }
   }

   // active fields
   sub_field_ids.SetSize(np);
   sub_field_ids = -1;
   for (int s = 0; s < parent_field_ids.Size(); s++)
   {
      const int p = parent_field_ids[s];
      MFEM_VERIFY(p >= 0 && p < np, "Active field id " << p << " out of range.");
      MFEM_VERIFY(sub_field_ids[p] == -1, "Duplicate active field id " << p);
      sub_field_ids[p] = s;

      const Array<int> &ess = *parent_ess_dofs[p];
      if (IsEssRemoved(p))
      {
         MFEM_VERIFY(X0, "X0 is required to supply values at the removed "
                     "essential dofs of field " << p << ".");
         RegisterField(parent_field_sizes[p] - ess.Size());
         if (parent.field_lower_bounds[p])
         {
            Vector lb(field_sizes[s]), ub(field_sizes[s]);
            GatherVector(*parent.field_lower_bounds[p], ess, lb);
            GatherVector(*parent.field_upper_bounds[p], ess, ub);
            SetFieldBounds(s, lb, ub, true);
         }
      }
      else
      {
         RegisterField(parent_field_sizes[p]);
         field_ess_dofs[s] = parent_ess_dofs[p]; // shared
         field_lower_bounds[s] = parent.field_lower_bounds[p]; // shared
         field_upper_bounds[s] = parent.field_upper_bounds[p];
      }
   }
   xwork.resize(np);
   dwork.resize(np);

   // active functionals: share functionals, copy roles
   Array<bool> taken(nf);
   taken = false;
   for (int k = 0; k < active_func_ids.Size(); k++)
   {
      const int i = active_func_ids[k];
      MFEM_VERIFY(i >= 0 && i < nf, "Active functional id " << i
                  << " out of range.");
      MFEM_VERIFY(!taken[i], "Duplicate active functional id " << i);
      taken[i] = true;

      const Array<int> &pids = parent.func_input_ids[i];
      Array<int> sids;
      for (int p : pids)
      {
         if (sub_field_ids[p] >= 0) { sids.Append(sub_field_ids[p]); }
         else
         {
            MFEM_VERIFY(X0, "Functional " << i << " depends on inactive field "
                        << p << "; X0 is required.");
         }
      }
      funcs.push_back(parent.funcs[i]);
      funcs_roles.push_back(parent.funcs_roles[i]);
      func_input_ids.push_back(sids);
      parent_input_ids.push_back(pids);
   }
   Finalize();
}

int SubOptProblem::InputWidth(int i) const
{
   int width = 0;
   for (auto p : parent_input_ids[i]) { width += parent_field_sizes[p]; }
   return width;
}

void SubOptProblem::BuildInput(int i, const MultiVector &X,
                               MultiVector &Xin) const
{
   MFEM_VERIFY(X.NumBlocks() == field_sizes.Size(),
               "Input must have one block per subproblem field.");
   const Array<int> &pids = parent_input_ids[i];
   Xin.SetNumBlocks(pids.Size());
   for (int k = 0; k < pids.Size(); k++)
   {
      const int p = pids[k];
      const int s = sub_field_ids[p];
      if (s < 0) { Xin.MakeRef(k, (*X0)[p]); } // X0 checked in the constructor
      else if (IsEssRemoved(p))
      {
         xwork[p] = (*X0)[p]; // values at essential dofs
         ScatterVector(X[s], *parent_ess_dofs[p], xwork[p]);
         Xin.MakeRef(k, std::as_const(xwork[p]));
      }
      else { Xin.MakeRef(k, X[s]); }
   }
}

real_t SubOptProblem::Eval(int i, const MultiVector &X) const
{
   MFEM_VERIFY(finalized,
               "OptProblem must be finalized before evaluating functionals.");
   MFEM_VERIFY(i >= 0 && i < (int)funcs.size(),
               "Functional index out of range.");
   MultiVector Xin;
   BuildInput(i, X, Xin);
   return funcs[i]->Eval(Xin);
}

void SubOptProblem::EvalDerivativeCompact(int i, const MultiVector &X,
                                          MultiVector &dfdXin) const
{
   const Array<int> &pids = parent_input_ids[i];
   MultiVector Xin;
   BuildInput(i, X, Xin);

   // dfin points (not aliases) to the caller's block if the field is active
   // with all dofs kept, and to dwork[p] otherwise (inactive: discarded; ess
   // dofs removed: gathered below).
   const Array<int> &sids = func_input_ids[i];
   MultiVector dfin;
   dfin.SetNumBlocks(pids.Size());
   for (int k = 0; k < pids.Size(); k++)
   {
      const int p = pids[k];
      const int s = sub_field_ids[p];
      if (s >= 0 && !IsEssRemoved(p)) { dfin.MakeRef(k, dfdXin[sids.Find(s)]); }
      else
      {
         dwork[p].SetSize(parent_field_sizes[p]);
         dwork[p].UseDevice(true);
         dfin.MakeRef(k, dwork[p]);
      }
   }
   if (!func_derivatives[i])
   { func_derivatives[i] = funcs[i]->Derivative(); }
   func_derivatives[i]->MultMV(Xin, dfin);

   for (int c = 0; c < sids.Size(); c++)
   {
      const int p = parent_field_ids[sids[c]];
      if (IsEssRemoved(p))
      { GatherVector(dwork[p], *parent_ess_dofs[p], dfdXin[c]); }
   }
}

} // namespace mfem
