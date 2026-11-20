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

#include "optimization.hpp"

namespace mfem
{

namespace optimization
{

Inequality::Inequality(Vector lower_, Vector upper_)
   : lower(std::move(lower_)), upper(std::move(upper_))
{
   if (lower.Size() == 0) { lower.SetSize(upper.Size()); lower = -mfem::infinity(); }
   else if (upper.Size() == 0) { upper.SetSize(lower.Size()); upper = mfem::infinity(); }
   MFEM_VERIFY(lower.Size() == upper.Size(),
               "Lower and upper bounds must have the same size.");

   const real_t *l = lower.HostRead(), *u = upper.HostRead();
   for (int i=0; i<lower.Size(); i++)
   {
      MFEM_VERIFY(l[i] <= u[i],
                  "Lower bound must be less than or equal to upper bound.");
   }
}

Inequality::Inequality(real_t l, real_t u, int n)
   : lower(n), upper(n)
{
   MFEM_VERIFY(l <= u, "Lower bound must be less than or equal to upper bound.");
   MFEM_VERIFY(n >= 0, "Size of bounds must be nonnegative.");
   lower = l; upper = u;
}

SimplexConstraint::SimplexConstraint(int stride_)
   : stride(stride_)
{
   MFEM_VERIFY(stride > 0, "Stride must be positive.");
}

Problem::~Problem()
{
   for (auto &rf : functionals)
   {
      if (rf.owned) { delete rf.f; }
   }
   for (auto &lc : local_constraints)
   {
      if (lc.own_set) { delete lc.set; }
      if (lc.own_op) { delete lc.op; }
   }
   for (auto &fm : field_metrics) { DeleteOwned(fm, FieldMetric()); }
}

int Problem::RegisterField(int size)
{
   finalized = false;
   field_sizes.Append(size);
   field_fixed.Append(false);
   field_ess_dofs.push_back(Array<int>());
   field_metrics.push_back(FieldMetric());
   return field_sizes.Size()-1;
}

int Problem::RegisterFunctional(Array<int> input_idx, const GlobalFunctional *f,
                                bool owned)
{
   finalized = false;
   MFEM_VERIFY(f != nullptr, "Functional pointer is null.");
   MFEM_VERIFY(input_idx.Max() < field_sizes.Size() && input_idx.Min() >= 0,
               "Input index is out of bounds.");
   functionals.push_back({f, owned, input_idx, std::monostate{}, true});
   return (int)functionals.size() - 1;
}

void Problem::SetRole(int functional_id, Role role)
{
   MFEM_VERIFY(functional_id >= 0 && functional_id < (int)functionals.size(),
               "Functional id " << functional_id << " is out of bounds.");
   MFEM_VERIFY(role.index() != 0, "Role must be specified (not std::monostate).");
   finalized = false;
   functionals[functional_id].role = std::move(role);
}

Role Problem::GetRole(int functional_id) const
{
   MFEM_VERIFY(functional_id >= 0 && functional_id < (int)functionals.size(),
               "Functional id " << functional_id << " is out of bounds.");
   return functionals[functional_id].role;
}

void Problem::DeactivateFunctional(int functional_id, bool deactivate)
{
   MFEM_VERIFY(functional_id >= 0 && functional_id < (int)functionals.size(),
               "Functional id " << functional_id << " is out of bounds.");
   finalized = false;
   functionals[functional_id].active = !deactivate;
}

void Problem::DeactivateFunctionals(const Array<int> &functional_ids,
                                    bool deactivate)
{
   for (auto id : functional_ids) { DeactivateFunctional(id, deactivate); }
}

void Problem::SetFieldFixed(int field_id, bool fixed)
{
   MFEM_VERIFY(field_id >= 0 && field_id < (int)field_sizes.Size(),
               "Field id " << field_id << " is out of bounds.");
   finalized = false;
   field_fixed[field_id] = fixed;
}

Array<int> Problem::GetFixedFields() const
{
   Array<int> fixed;
   for (int i=0; i<field_fixed.Size(); i++)
   {
      if (field_fixed[i]) { fixed.Append(i); }
   }
   return fixed;
}

void Problem::SetFieldEssentialDofs(int field_id, const Array<int> &ess_dofs)
{
   MFEM_VERIFY(field_id >= 0 && field_id < (int)field_sizes.Size(),
               "Field id " << field_id << " is out of bounds.");
   finalized = false;
   ess_dofs.Copy(field_ess_dofs[field_id]);
}

void Problem::SetFieldMetric(int field_id, const Operator *metric,
                             const Operator *riesz,
                             bool own_metric, bool own_riesz)
{
   MFEM_VERIFY(field_id >= 0 && field_id < (int)field_sizes.Size(),
               "Field id " << field_id << " is out of bounds.");
   const FieldMetric fm = {metric, own_metric, riesz, own_riesz};
   DeleteOwned(field_metrics[field_id], fm);
   field_metrics[field_id] = fm;
   finalized = false;
}

real_t Problem::InnerProduct(int field_id, const Vector &x,
                             const Vector &y) const
{
   MFEM_VERIFY(field_id >= 0 && field_id < (int)field_sizes.Size(),
               "Field id " << field_id << " is out of bounds.");
   const Operator *metric = field_metrics[field_id].metric;
   const Vector *My = &y;
   if (metric)
   {
      metric_work.UseDevice(y.UseDevice());
      metric_work.SetSize(metric->Height());
      metric->Mult(y, metric_work);
      My = &metric_work;
   }
#ifdef MFEM_USE_MPI
   if (comm != MPI_COMM_NULL) { return mfem::InnerProduct(comm, x, *My); }
#endif
   return x * (*My);
}

real_t Problem::InnerProduct(const MultiVector &x, const MultiVector &y) const
{
   MFEM_VERIFY(x.NumBlocks() == field_sizes.Size()
               && y.NumBlocks() == field_sizes.Size(),
               "Expected " << field_sizes.Size() << " fields.");
   real_t sum = 0.0;
   for (int i=0; i<field_sizes.Size(); i++)
   {
      sum += InnerProduct(i, x[i], y[i]);
   }
   return sum;
}

void Problem::ApplyRiesz(int field_id, const Vector &dual, Vector &primal) const
{
   MFEM_VERIFY(field_id >= 0 && field_id < (int)field_sizes.Size(),
               "Field id " << field_id << " is out of bounds.");
   const Operator *riesz = field_metrics[field_id].riesz;
   if (riesz == nullptr) { primal = dual; return; }
   primal.SetSize(riesz->Height());
   riesz->Mult(dual, primal);
}

void Problem::ApplyRiesz(const MultiVector &duals, MultiVector &primals) const
{
   MFEM_VERIFY(duals.NumBlocks() == field_sizes.Size()
               && primals.NumBlocks() == field_sizes.Size(),
               "Expected " << field_sizes.Size() << " fields.");
   for (int i=0; i<field_sizes.Size(); i++)
   {
      ApplyRiesz(i, duals[i], primals[i]);
   }
}

int Problem::RegisterLocalConstraint(int field_id, const ConstraintSet *set,
                                     const Operator *op,
                                     bool own_set, bool own_op)
{
   MFEM_VERIFY(field_id >= 0 && field_id < (int)field_sizes.Size(),
               "Field id " << field_id << " is out of bounds.");
   MFEM_VERIFY(set != nullptr, "Constraint set pointer is null.");
   finalized = false;
   local_constraints.push_back({field_id, set, own_set, op, own_op});
   return (int)local_constraints.size() - 1;
}

std::tuple<int, const ConstraintSet&, const Operator*>
Problem::GetLocalConstraint(int constraint_id) const
{
   MFEM_VERIFY(constraint_id >= 0
               && constraint_id < (int)local_constraints.size(),
               "Local constraint id " << constraint_id << " is out of bounds.");
   const auto &lc = local_constraints[constraint_id];
   return {lc.field_id, *lc.set, lc.op};
}

void Problem::Finalize()
{
   held_dofs.resize(field_sizes.Size());
   fixed_field_dofs.resize(field_sizes.Size());
   for (int i=0; i<field_sizes.Size(); i++)
   {
      held_dofs[i] = &field_ess_dofs[i];
#ifdef MFEM_DEBUG
      const int *dofs = field_ess_dofs[i].HostRead();
      for (int k=0; k<field_ess_dofs[i].Size(); k++)
      {
         MFEM_ASSERT(dofs[k] >= 0 && dofs[k] < field_sizes[i],
                     "Field " << i << ": essential dof " << dofs[k]
                     << " is out of bounds for size " << field_sizes[i]);
      }
#endif
      fixed_field_dofs[i].SetSize(field_fixed[i] ? field_sizes[i] : 0);
      for (int k=0; k<fixed_field_dofs[i].Size(); k++)
      {
         fixed_field_dofs[i][k] = k;
      }
      if (field_fixed[i]) { held_dofs[i] = &fixed_field_dofs[i]; }
   }
   for (int c=0; c<(int)local_constraints.size(); c++)
   {
      const auto &lc = local_constraints[c];
      const int n = field_sizes[lc.field_id];
      MFEM_VERIFY(lc.op == nullptr || lc.op->Width() == n,
                  "Local constraint " << c << ": expected an operator of width "
                  << n << ", but got " << lc.op->Width());
      const int m = lc.op ? lc.op->Height() : n;
      if (auto *box = dynamic_cast<const BoxConstraint*>(lc.set))
      {
         MFEM_VERIFY(box->lower.Size() == m,
                     "Local constraint " << c << ": expected " << m
                     << " bounds, but got " << box->lower.Size());
      }
      if (auto *simplex = dynamic_cast<const SimplexConstraint*>(lc.set))
      {
         MFEM_VERIFY(m % simplex->stride == 0,
                     "Local constraint " << c << ": size " << m
                     << " is not a multiple of the stride " << simplex->stride);
      }
   }
   for (int i=0; i<field_sizes.Size(); i++)
   {
      const auto &fm = field_metrics[i];
      const int n = field_sizes[i];
      MFEM_VERIFY(fm.metric == nullptr
                  || (fm.metric->Height() == n && fm.metric->Width() == n),
                  "Field " << i << ": expected a metric of size " << n
                  << " x " << n);
      MFEM_VERIFY(fm.riesz == nullptr
                  || (fm.riesz->Height() == n && fm.riesz->Width() == n),
                  "Field " << i << ": expected a Riesz map of size " << n
                  << " x " << n);
   }

   auto objs = MakeSum();
   auto eqs = MakeSum();
   auto ineqs = MakeSum();
   std::vector<const Vector *> eq_targets, ineq_lower, ineq_upper;
   Array<int> eq_offsets(1), ineq_offsets(1);
   eq_offsets[0] = 0; ineq_offsets[0] = 0;
   equality_index.SetSize((int)functionals.size());
   inequality_index.SetSize((int)functionals.size());
   equality_index = -1;
   inequality_index = -1;
   int num_eq_rows = 0, num_ineq_rows = 0;

   for (int i=0; i<(int)functionals.size(); i++)
   {
      auto &rf = functionals[i];
      if (!rf.active) { continue; }

      rf.f->VerifyInputSize(InputSizes(rf.input_idx));
      const int m = rf.f->Height();

      if (std::holds_alternative<Objective>(rf.role))
      {
         const auto &obj = std::get<Objective>(rf.role);
         if (m == 0) { continue; }
         MFEM_VERIFY(obj.weight.Size() == m,
                     "Functional " << i << ": expected " << m
                     << " objective weights, but got " << obj.weight.Size());
         objs->Add(0, rf.input_idx, obj.weight, rf.f);
      }
      else if (std::holds_alternative<Equality>(rf.role))
      {
         const auto &eq = std::get<Equality>(rf.role);
         MFEM_VERIFY(eq.target.Size() == m,
                     "Functional " << i << ": expected " << m
                     << " equality targets, but got " << eq.target.Size());
         eqs->Add(eq_targets.size(), rf.input_idx, Vector(), rf.f);
         eq_targets.push_back(&eq.target);
         eq_offsets.Append(m);
         equality_index[i] = num_eq_rows;
         num_eq_rows += m;
      }
      else if (std::holds_alternative<Inequality>(rf.role))
      {
         const auto &ineq = std::get<Inequality>(rf.role);
         MFEM_VERIFY(ineq.lower.Size() == m && ineq.upper.Size() == m,
                     "Functional " << i << ": expected " << m
                     << " inequality bounds, but got " << ineq.lower.Size());
         ineqs->Add(ineq_lower.size(), rf.input_idx, Vector(), rf.f);
         ineq_lower.push_back(&ineq.lower);
         ineq_upper.push_back(&ineq.upper);
         ineq_offsets.Append(m);
         inequality_index[i] = num_ineq_rows;
         num_ineq_rows += m;
      }
      else { MFEM_ABORT("Active functional " << i << " has no role"); }
   }
   objs->Finalize();
   objective = std::make_unique<ConstrainedFunctional>(
                  held_dofs, objs.release(), true);
   eqs->Finalize();
   equality = std::make_unique<ConstrainedFunctional>(
                 held_dofs, eqs.release(), true);
   ineqs->Finalize();
   inequality = std::make_unique<ConstrainedFunctional>(
                   held_dofs, ineqs.release(), true);

   eq_offsets.PartialSum();
   ineq_offsets.PartialSum();
   equality_target.target.SetSize(eq_offsets.Last());
   inequality_bounds.lower.SetSize(ineq_offsets.Last());
   inequality_bounds.upper.SetSize(ineq_offsets.Last());
   for (int k=0; k<(int)eq_targets.size(); k++)
   {
      equality_target.target.SetVector(*eq_targets[k], eq_offsets[k]);
   }
   for (int k=0; k<(int)ineq_lower.size(); k++)
   {
      inequality_bounds.lower.SetVector(*ineq_lower[k], ineq_offsets[k]);
      inequality_bounds.upper.SetVector(*ineq_upper[k], ineq_offsets[k]);
   }
   finalized = true;
}

const GlobalFunctional &Problem::GetObjective() const
{
   MFEM_VERIFY(finalized, "Problem is not finalized. Call Finalize() first.");
   return *objective;
}

std::tuple<const GlobalFunctional&, const Equality&> Problem::GetEquality()
const
{
   MFEM_VERIFY(finalized, "Problem is not finalized. Call Finalize() first.");
   return {*equality, equality_target};
}

std::tuple<const GlobalFunctional&, const Inequality&> Problem::GetInequality()
const
{
   MFEM_VERIFY(finalized, "Problem is not finalized. Call Finalize() first.");
   return {*inequality, inequality_bounds};
}

int Problem::GetEqualityIndex(int functional_id) const
{
   MFEM_VERIFY(finalized, "Problem is not finalized. Call Finalize() first.");
   MFEM_VERIFY(functional_id >= 0 && functional_id < (int)functionals.size(),
               "Functional id " << functional_id << " is out of bounds.");
   return equality_index[functional_id];
}

int Problem::GetInequalityIndex(int functional_id) const
{
   MFEM_VERIFY(finalized, "Problem is not finalized. Call Finalize() first.");
   MFEM_VERIFY(functional_id >= 0 && functional_id < (int)functionals.size(),
               "Functional id " << functional_id << " is out of bounds.");
   return inequality_index[functional_id];
}

const Array<int> &Problem::GetFieldEssentialDofs(int field_id) const
{
   MFEM_VERIFY(field_id >= 0 && field_id < (int)field_sizes.Size(),
               "Field id " << field_id << " is out of bounds.");
   return field_ess_dofs[field_id];
}

void Problem::SetFieldSize(int field_idx, int size)
{
   MFEM_VERIFY(field_idx >= 0 && field_idx < (int)field_sizes.Size(),
               "Field index " << field_idx << " is out of bounds.");
   finalized = false;
   field_sizes[field_idx] = size;
   field_ess_dofs[field_idx].DeleteAll(); // invalidate essential dofs
}

void Problem::SetFieldSizes(const Array<int> &sizes)
{
   MFEM_VERIFY(sizes.Size() == field_sizes.Size(),
               "Expected " << field_sizes.Size() << " fields, but got " << sizes.Size());
   finalized = false;
   field_sizes = sizes;
   // invalidate essential dofs
   for (auto &ess : field_ess_dofs) { ess.DeleteAll(); }
}

const GlobalFunctional *Problem::GetFunctional(int functional_id) const
{
   MFEM_VERIFY(functional_id >= 0 && functional_id < (int)functionals.size(),
               "Functional id " << functional_id << " is out of bounds.");
   return functionals[functional_id].f;
}

void Problem::DeleteOwned(const FieldMetric &fm, const FieldMetric &next)
{
   const bool keep_metric = fm.metric == next.metric
                            || fm.metric == next.riesz;
   const bool keep_riesz = fm.riesz == next.metric || fm.riesz == next.riesz;
   if (fm.own_metric && !keep_metric) { delete fm.metric; }
   if (fm.own_riesz && !keep_riesz
       && !(fm.own_metric && fm.riesz == fm.metric)) { delete fm.riesz; }
}

std::unique_ptr<WeightedSumFunctional> Problem::MakeSum() const
{
#ifdef MFEM_USE_MPI
   if (comm != MPI_COMM_NULL)
   {
      return std::make_unique<WeightedSumFunctional>(comm, field_sizes);
   }
#endif
   return std::make_unique<WeightedSumFunctional>(field_sizes);
}

Array<int> Problem::InputSizes(const Array<int> &field_ids) const
{
   Array<int> sizes(field_ids.Size());
   for (int i=0; i<field_ids.Size(); i++) { sizes[i] = field_sizes[field_ids[i]]; }
   return sizes;
}

} // namespace optimization

} // namespace mfem
