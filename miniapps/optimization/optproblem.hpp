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

#ifndef MFEM_OPTPROBLEM_HPP
#define MFEM_OPTPROBLEM_HPP

#include "mfem.hpp"
#include <memory>
#include "functional.hpp"

namespace mfem
{
/// @brief Role of a functional in an optimization problem:
/// objective, equality constraint, or inequality constraint.
/// Use @ref Objective(), @ref Equality(), and @ref Inequality() to construct a FunctionalRole.
class FunctionalRole
{
public:
   enum Role { OBJ, EQ, INEQ };
private:
   Role role;
   real_t lower;
   real_t upper;

   /// @brief Private constructor for FunctionalRole. For public usage,
   /// See @ref Objective(), @ref Equality(), and @ref Inequality()
   FunctionalRole(Role role, real_t lower_, real_t upper_)
      : role(role), lower(lower_), upper(upper_) {}
public:
   static FunctionalRole Objective()
   { return FunctionalRole(Role::OBJ, -infinity(), infinity()); }

   static FunctionalRole Equality(real_t target)
   { return FunctionalRole(Role::EQ, target, target); }

   static FunctionalRole Inequality(real_t lower, real_t upper)
   {
      MFEM_VERIFY(lower <= upper, "Inequality constraint lower bound "
                  << lower << " is greater than upper bound " << upper << ".");
      return FunctionalRole(Role::INEQ, lower, upper);
   }

   bool IsObjective() const { return role == Role::OBJ; }

   bool IsEquality() const { return role == Role::EQ; }
   real_t GetTarget() const
   {
      MFEM_VERIFY(IsEquality(), "FunctionalRole is not an equality constraint.");
      return lower;
   }

   bool IsInequality() const { return role == Role::INEQ; }
   real_t GetLower() const
   {
      MFEM_VERIFY(!IsObjective(), "Objective does not have lower bound.");
      return lower;
   }
   real_t GetUpper() const
   {
      MFEM_VERIFY(!IsObjective(), "Objective does not have upper bound.");
      return upper;
   }
};


class SubOptProblem;
class OptProblem
{
   /// reads the parent's fields and functionals when constructing
   friend class SubOptProblem;
protected:
   Array<int> field_sizes; // size of each field
   /// essential dofs for each field. empty if none (cannot be null)
   /// We use shared_ptr to allow for non-owning references
   std::vector<std::shared_ptr<const Array<int>>> field_ess_dofs;

   /// Registered functionals
   std::vector<std::shared_ptr<Functional>> funcs;
   /// Registered functionals' input field ids.
   std::vector<Array<int>> func_input_ids;
   /// Registered functionals' roles.
   std::vector<FunctionalRole> funcs_roles;

   /// Registered dofwise bounds for each field. Null if none.
   std::vector<std::shared_ptr<const Vector>> field_lower_bounds;
   std::vector<std::shared_ptr<const Vector>> field_upper_bounds;

   bool finalized = false;
   int obj_id = -1;
   /// Constraints [equality, inequality] where equality and inequality
   /// are ordered by the functional registration order.
   Array<const Functional*> constraints;
   Array<int> constraint_ids;
   /// Lower and upper bounds for the constraints. lower[i]==upper[i] for equality
   Vector lower, upper;
   int num_eq = 0;

   mutable std::vector<std::shared_ptr<Operator>> func_derivatives;

   /// Total input size of the i-th functional, checked against its Width().
   virtual int InputWidth(int i) const;
   virtual real_t Eval(int i, const MultiVector &X) const;
   /// Derivative of the i-th functional with respect to its own inputs only:
   /// @a dfdXin has one block per func_input_ids[i], in that order, already
   /// sized (or pointing at the caller's storage).
   virtual void EvalDerivativeCompact(int i, const MultiVector &X,
                                      MultiVector &dfdXin) const;
   /// Derivative of the i-th functional with respect to all fields: sizes
   /// @a dfdX, zeros the fields it does not depend on, and fills the rest with
   /// EvalDerivativeCompact().
   void EvalDerivative(int i, const MultiVector &X, MultiVector &dfdX) const;

public:
   OptProblem() = default;
   virtual ~OptProblem() = default;

   /// Register a field with no essential dofs.
   int RegisterField(int size);

   /// Register a field. If @a copy_ess is false, @a ess_dofs must outlive
   /// this model (and any copy of it).
   int RegisterField(int size, const Array<int> &ess_dofs, bool copy_ess=true);

   /// Resize a field. Essential dofs and dofwise bounds will be cleared.
   void UpdateFieldSize(int field_id, int new_size);

   /// Resize a field and replace its essential dofs (sorted, unique).
   /// Dofwise bounds will be cleared. If @a copy_ess is false,
   /// @a new_ess_dofs must outlive this model (and any copy of it).
   void UpdateFieldSize(int field_id, int new_size,
                        const Array<int> &new_ess_dofs, bool copy_ess=true);

   /// Set dofwise bounds of a field. If @a copy_bounds is false, @a lower and
   /// @a upper must outlive this model (and any copy of it).
   void SetFieldBounds(int field_id, const Vector &lower, const Vector &upper,
                       bool copy_bounds=true);
   const Vector& GetFieldLowerBounds(int field_id)
   {
      MFEM_VERIFY(field_id >= 0 && field_id < field_sizes.Size(),
                  "Field index out of range.");
      return *field_lower_bounds[field_id];
   }
   const Vector& GetFieldUpperBounds(int field_id)
   {
      MFEM_VERIFY(field_id >= 0 && field_id < field_sizes.Size(),
                  "Field index out of range.");
      return *field_upper_bounds[field_id];
   }

   const Array<int> &GetEssDofs(int field_id) const
   {
      MFEM_VERIFY(field_id >= 0 && field_id < field_sizes.Size(),
                  "Field index out of range.");
      return *field_ess_dofs[field_id];
   }

   /// @brief Register a functional of the fields @a field_ids with the given
   /// role. If @a take_ownership is false, @a f must outlive this model (and
   /// any copy of it).
   int RegisterFunctional(FunctionalRole role, const Array<int> &field_ids,
                          Functional *f, bool take_ownership=false);

   /// @brief Change the role of a registered functional.
   void ChangeRole(int func_id, FunctionalRole role);
   std::vector<std::shared_ptr<Functional>> GetFunctionals() const { return funcs; }

   /// @brief Finalize the optimization problem.
   virtual void Finalize();
   bool IsFinalized() const { return finalized; }

   bool HasObjective() const
   {
      MFEM_VERIFY(finalized,
                  "OptProblem must be finalized before checking for objective.");
      return obj_id >= 0;
   }

   /// @brief Get the registered constraints (equalities then inequalities)
   const Array<const Functional*> &GetConstraints() const;
   bool HasConstraints() const
   {
      MFEM_VERIFY(finalized,
                  "OptProblem must be finalized before checking for constraints.");
      return constraints.Size() > 0;
   }
   /// @brief Get the lower bounds of the constraints including equalities
   const Vector &GetConstraintsLowerBounds() const;
   /// @brief Get the upper bounds of the constraints including equalities
   const Vector &GetConstraintsUpperBounds() const;

   /// @brief Number of constraints (equality + inequality)
   int GetNumConstraints() const
   {
      MFEM_VERIFY(finalized, "OptProblem must be finalized before counting "
                  "constraints.");
      return constraints.Size();
   }
   /// @brief Number of equality constraints
   int GetNumEqualities() const
   {
      MFEM_VERIFY(finalized, "OptProblem must be finalized before counting "
                  "constraints.");
      return num_eq;
   }

   real_t EvalObjective(const MultiVector &X) const;
   void EvalObjectiveDerivative(const MultiVector &X, MultiVector &dfdX) const;

   void EvalConstraintDerivative(int i, const MultiVector &X,
                                 MultiVector &dc_dX) const;

   /// @brief Evaluate both equality and inequality Jacobian transpose at X
   /// Constraints are [equality, inequality] and each is ordered by the
   /// functional registration order.
   void EvalConstraintJacobianTranspose(const MultiVector &X,
                                        DenseMatrix &JT, int start=0, int end=-1) const;
   /// @brief Evaluate equality Jacobian transpose at X
   void EvalEqualityConstraintJacobianTranspose(const MultiVector &X,
                                                DenseMatrix &JT) const
   { EvalConstraintJacobianTranspose(X, JT, 0, num_eq); }
   /// @brief Evaluate inequality Jacobian transpose at X
   void EvalInequalityConstraintJacobianTranspose(const MultiVector &X,
                                                  DenseMatrix &JT) const
   { EvalConstraintJacobianTranspose(X, JT, num_eq, constraints.Size()); }

   real_t EvalConstraint(int i, const MultiVector &X) const
   {
      MFEM_VERIFY(finalized,
                  "OptProblem must be finalized before evaluating constraints.");
      MFEM_VERIFY(i >= 0 && i < constraints.Size(),
                  "Constraint index out of range.");
      return Eval(constraint_ids[i], X);
   }

   /// @brief Evaluate both equality and inequality constraints at X
   /// Constraints are [equality, inequality] and each is ordered by the
   /// functional registration order.
   void EvalConstraints(const MultiVector &X, Vector &c) const
   {
      c.SetSize(constraints.Size());
      real_t *c_data = c.HostWrite();
      for (int i=0; i<constraints.Size(); i++)
      { c_data[i] = EvalConstraint(i, X); }
   }

   /// @brief Evaluate equality constraints at X
   void EvalEqualityConstraints(const MultiVector &X, Vector &c) const
   {
      c.SetSize(num_eq);
      real_t *c_data = c.HostWrite();
      for (int i=0; i<num_eq; i++)
      { c_data[i] = EvalConstraint(i, X); }
   }

   /// @brief Evaluate inequality constraints at X
   void EvalInequalityConstraints(const MultiVector &X, Vector &c) const
   {
      MFEM_VERIFY(finalized,
                  "OptProblem must be finalized before evaluating constraints.");
      c.SetSize(constraints.Size()-num_eq);
      real_t *c_data = c.HostWrite();
      for (int i=0; i<constraints.Size()-num_eq; i++)
      { c_data[i] = EvalConstraint(i+num_eq, X); }
   }

   /// @brief Create a finalized subproblem with only the fields in
   /// @a active_field_ids and the functionals in @a active_func_ids, in the
   /// given order. See SubOptProblem.
   SubOptProblem MakeSubProblem(
      const Array<int> &active_field_ids,
      const Array<int> &active_func_ids,
      const MultiVector *X=nullptr, bool copy_X=false,
      bool remove_ess_dofs=false) const;
};

/// @brief Subproblem of an OptProblem over a subset of its fields and
/// functionals. Functionals and roles are shared with the parent (not copied);
/// the parent itself is not referenced after construction.
///
/// Inputs of the subproblem are the active fields only. Inactive fields that
/// an active functional depends on are taken from @a X0, and derivatives
/// with respect to them are dropped.
///
/// If @a remove_ess_dofs is true, each active field's essential dofs are
/// removed from its input: the field size becomes size - #ess, the values at
/// essential dofs are taken from @a X0, and derivatives at essential dofs are
/// dropped. Dofwise bounds are reduced the same way. Host only.
///
/// @a X0 has one block per parent field. It is required if any active
/// functional uses an inactive field, or if @a remove_ess_dofs is true and an
/// active field has essential dofs. If @a copy_X is false, @a X0 (and its
/// blocks) must outlive the subproblem.
class SubOptProblem : public OptProblem
{
   Array<int> parent_field_ids;  ///< sub field -> parent field
   Array<int> sub_field_ids;     ///< parent field -> sub field, -1 if inactive
   Array<int> parent_field_sizes;
   std::vector<std::shared_ptr<const Array<int>>> parent_ess_dofs;
   std::vector<Array<int>> parent_input_ids; ///< per sub functional
   std::shared_ptr<const MultiVector> X0;
   bool remove_ess_dofs;

   /// Storage for scattered input and discarded/gathered derivative blocks
   mutable std::vector<Vector> xwork, dwork;

   bool IsEssRemoved(int p) const
   { return remove_ess_dofs && parent_ess_dofs[p]->Size() > 0; }
   /// Build the input of the i-th functional (one block per
   /// parent_input_ids[i]) from the subproblem input @a X.
   void BuildInput(int i, const MultiVector &X, MultiVector &Xin) const;

protected:
   int InputWidth(int i) const override;
   real_t Eval(int i, const MultiVector &X) const override;
   void EvalDerivativeCompact(int i, const MultiVector &X,
                              MultiVector &dfdXin) const override;

public:
   /// Create a subproblem of @a parent with only the fields in @a active_field_ids
   /// and the functionals in @a active_func_ids, in the given order.
   /// When there are missing fields, @a X0 must be provided.
   /// If @a remove_ess_dofs is true, then @a copy_X must be true,
   /// and scatter and gather are host-only.
   SubOptProblem(const OptProblem &parent,
                 const Array<int> &active_field_ids,
                 const Array<int> &active_func_ids,
                 const MultiVector *X0_=nullptr, bool copy_X=false,
                 bool remove_ess_dofs=false);

   /// Parent field id of the i-th subproblem field.
   int ParentFieldId(int i) const { return parent_field_ids[i]; }
};
}

#endif
