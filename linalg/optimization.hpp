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

#ifndef MFEM_OPTIMIZATION_HPP
#define MFEM_OPTIMIZATION_HPP

#include "../config/config.hpp"
#include "functional.hpp"
#include <memory>
#include <tuple>
#include <variant>
#include <vector>

namespace mfem
{

namespace optimization
{

/// @brief Objective role, representing sum w_i f_i(x) in optimization problem.
struct Objective
{
   Vector weight;
   /// @brief Construct an objective with given weight vector, sum w_i f_i(x)
   Objective(Vector weight_) : weight(std::move(weight_)) {}
   /// @brief Construct an objective with given scalar weight, w sum f_i(x).
   Objective(real_t w=1.0, int n=1) : weight(n) { weight = w; }
};

struct Equality
{
   Vector target;
   /// @brief Construct an equality constraint with given target vector, f_i(x) = target_i
   Equality(Vector target_) : target(std::move(target_)) {}
   /// @brief Construct an equality constraint with given scalar target, f_i(x) = target.
   Equality(real_t t=0.0, int n=1) : target(n) { target = t; }
};

struct Inequality
{
   /// @brief Lower and upper bounds for inequality constraints, f_i(x) in [lower_i, upper_i]
   Vector lower, upper;
   /// @brief Construct an empty inequality constraint (no components).
   Inequality() = default;
   /// @brief Construct an inequality constraint with given lower and upper bounds,
   /// f_i(x) in [lower_i, upper_i]. When lower or upper is empty, it means no lower or upper bound.
   /// When both are empty, the constraint has no components.
   Inequality(Vector lower_, Vector upper_);

   Inequality(real_t l, real_t u, int n=1);
};

typedef std::variant<std::monostate, Objective, Equality, Inequality>
Role;

/// @brief A set C in a local constraint y in C; see
/// Problem::RegisterLocalConstraint(). Solvers identify the set by its type.
struct ConstraintSet
{
   virtual ~ConstraintSet() = default;
};

/// @brief lower_i <= y_i <= upper_i for each entry y_i of y.
struct BoxConstraint : public ConstraintSet, public Inequality
{
   using Inequality::Inequality;
};

/// @brief Each stride of @a stride consecutive entries of y sums to 1 and is
/// nonnegative.
struct SimplexConstraint : public ConstraintSet
{
   int stride;
   SimplexConstraint(int stride_);
};



class Problem
{
protected:
   bool finalized = false;
   struct RoledFunctional
   {
      const GlobalFunctional *f;
      bool owned;
      Array<int> input_idx;
      Role role;
      bool active;
   };
   std::vector<RoledFunctional> functionals;
   std::unique_ptr<ConstrainedFunctional> objective;
   std::unique_ptr<ConstrainedFunctional> equality;
   Equality equality_target;
   std::unique_ptr<ConstrainedFunctional> inequality;
   Inequality inequality_bounds;

   Array<int> field_sizes;
   Array<bool> field_fixed;
   std::vector<Array<int>> field_ess_dofs;

   struct LocalConstraint
   {
      int field_id;
      const ConstraintSet *set;
      bool own_set;
      const Operator *op;
      bool own_op;
   };
   std::vector<LocalConstraint> local_constraints;
   // Dofs held fixed in the functionals: all dofs of fixed fields, the
   // essential dofs otherwise.
   std::vector<Array<int>> fixed_field_dofs;
   std::vector<const Array<int>*> held_dofs;

   struct FieldMetric
   {
      const Operator *metric = nullptr;
      bool own_metric = false;
      const Operator *riesz = nullptr;
      bool own_riesz = false;
   };
   std::vector<FieldMetric> field_metrics;
   mutable Vector metric_work;
   // First row of each functional in the equality and inequality constraints,
   // or -1.
   Array<int> equality_index, inequality_index;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
#endif
public:
   Problem() = default;
#ifdef MFEM_USE_MPI
   Problem(MPI_Comm communicator): comm(communicator) {}
   MPI_Comm GetComm() const { return comm; }
#endif
   virtual ~Problem();
   Problem(const Problem &) = delete;
   Problem &operator=(const Problem &) = delete;
   int RegisterField(int size);

   /// @brief Register a functional of the fields @a input_idx; returns its id.
   int RegisterFunctional(Array<int> input_idx, const GlobalFunctional *f,
                          bool owned=false);

   /// @brief Set the role of a registered functional, e.g., objective, equality, or inequality.
   void SetRole(int functional_id, Role role);
   /// @brief The role of a registered functional; std::monostate if not set.
   Role GetRole(int functional_id) const;

   void DeactivateFunctional(int functional_id, bool deactivate=true);
   void DeactivateFunctionals(const Array<int> &functional_ids,
                              bool deactivate=true);

   /// @brief Set a field as fixed (not optimized) or free (optimized).
   void SetFieldFixed(int field_id, bool fixed=true);
   void SetFieldsFixed(const Array<int> &field_ids, bool fixed=true)
   { for (auto field_id : field_ids) { SetFieldFixed(field_id, fixed); } }

   Array<int> GetFixedFields() const;


   /** @brief Set the essential degrees of freedom for a field,
   * copied and removed when the field size changes.
   * The entries are dof indices in [0, field size), without sign encoding. */
   void SetFieldEssentialDofs(int field_id, const Array<int> &ess_dofs);

   /// @brief Set the metric M of a field, (x, y)_M = x^T M y, and its Riesz map
   /// R, which maps a dual vector (e.g., a derivative) to a primal vector,
   /// typically R = M^{-1} or an approximation of it.
   ///
   /// nullptr means the identity, which is the default for both.
   void SetFieldMetric(int field_id, const Operator *metric, const Operator *riesz,
                       bool own_metric=false, bool own_riesz=false);

   /// @brief The inner product (x, y)_M of the field @a field_id, reduced over
   /// the communicator.
   real_t InnerProduct(int field_id, const Vector &x, const Vector &y) const;
   /// @brief The sum of the inner products of the fields.
   real_t InnerProduct(const MultiVector &x, const MultiVector &y) const;

   /// @brief primal = R dual for the field @a field_id.
   void ApplyRiesz(int field_id, const Vector &dual, Vector &primal) const;
   /// @brief primals[i] = R_i duals[i] for each field i.
   void ApplyRiesz(const MultiVector &duals, MultiVector &primals) const;

   /// @brief Register a local constraint op(x) in set on the field @a field_id;
   /// returns its id. op == nullptr means the identity, x in set (dofwise).
   ///
   /// The constraint assumes that each stride of op(x) is in the set: one entry
   /// for BoxConstraint, and SimplexConstraint::stride consecutive entries for
   /// SimplexConstraint.
   int RegisterLocalConstraint(int field_id, const ConstraintSet *set,
                               const Operator *op=nullptr,
                               bool own_set=false, bool own_op=false);
   int NumLocalConstraints() const { return (int)local_constraints.size(); }
   /// @brief The field, set, and operator (nullptr for the identity) of the
   /// local constraint @a constraint_id.
   std::tuple<int, const ConstraintSet&, const Operator*>
   GetLocalConstraint(int constraint_id) const;

   /// @brief Finalize the problem layout (field sizes, functionals, and roles)
   ///
   /// Builds the objective (one row, sum of weighted objectives), and the
   /// equality and inequality constraints (one block of rows per functional).
   /// A role without functionals gives a functional with zero rows.
   ///
   /// The functionals hold the essential dofs and all dofs of fixed fields:
   /// their derivatives are zero and their Hessians are the identity there,
   /// see ConstrainedFunctional.
   /// Finalize() checks the sizes that the setters leave unchecked: functional
   /// inputs, essential dofs, local constraints, metrics, and Riesz maps.
   ///
   /// Any modification to the field sizes, functionals, or roles will invalidate the layout
   void Finalize();

   const GlobalFunctional &GetObjective() const;
   /// @brief The equality constraints c(x) = target; zero rows if none.
   std::tuple<const GlobalFunctional&, const Equality&> GetEquality() const;
   /// @brief The inequality constraints lower <= c(x) <= upper; zero rows if none.
   std::tuple<const GlobalFunctional&, const Inequality&> GetInequality() const;
   /// @brief The first row of the functional @a functional_id in the equality
   /// constraints; it has GetFunctional(functional_id)->Height() rows.
   /// Returns -1 if the functional is not an active equality constraint.
   int GetEqualityIndex(int functional_id) const;
   /// @brief The first row of the functional @a functional_id in the
   /// inequality constraints; it has GetFunctional(functional_id)->Height()
   /// rows. Returns -1 if the functional is not an active inequality
   /// constraint.
   int GetInequalityIndex(int functional_id) const;
   const Array<int> &GetFieldEssentialDofs(int field_id) const;

   /// @brief Check if the problem layout is finalized.
   /// Any modification after finalization will invalidate the layout.
   bool Finalized() const { return finalized; }

   /// @name Getter and setter for field sizes
   ///@{
   int NumFields() const { return field_sizes.Size(); }
   const Array<int> &GetFieldSizes() const { return field_sizes; }
   void SetFieldSize(int field_idx, int size);
   int NumFunctionals() const { return (int)functionals.size(); }

   void SetFieldSizes(const Array<int> &sizes);
   ///@}

   const GlobalFunctional *GetFunctional(int functional_id) const;

protected:
   /// Delete the owned operators of @a fm that @a next does not use.
   static void DeleteOwned(const FieldMetric &fm, const FieldMetric &next);

   std::unique_ptr<WeightedSumFunctional> MakeSum() const;

   /// The sizes of the fields @a field_ids.
   Array<int> InputSizes(const Array<int> &field_ids) const;
};

} // namespace optimization

} // namespace mfem

#endif // MFEM_OPTIMIZATION_HPP
