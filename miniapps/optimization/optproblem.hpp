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
#include <tuple>
#include <vector>

namespace mfem
{

/** @brief Operator notified of new iterates by OptProblem::Update().

    Typical use: several quantities share a state solve u(x) and recompute it
    only when the point sequence changes. Update() should be cheap; the actual
    work belongs in Mult() / GetGradient(). */
class UpdatableOperator : public Operator
{
public:
   using Operator::Operator;
   /// The iterate is now @a x, with point sequence @a seq.
   virtual void Update(const Vector &x, long seq) = 0;
};

/** @brief Contraction of a stacked operator: v -> sum_i y_i S_i v, where
    S = [S_1; ...; S_m] is (m n) x n and y has size m.

    Typical use: a multiplier-weighted Hessian from the stacked second
    derivative of an OptProblem view. @a S is not owned; @a y is copied. */
class ContractedOperator : public Operator
{
public:
   ContractedOperator(const Operator &S, const Vector &y);
   /// Hv = sum_i y_i S_i v
   void Mult(const Vector &v, Vector &Hv) const override;
   /// Hv = S^T (y \otimes v) = sum_i y_i S_i^T v
   void MultTranspose(const Vector &v, Vector &Hv) const override;

private:
   const Operator &S;
   Vector y;
   mutable Vector Sv;
};

/** @brief Optimization problem built from registered operators
    q: R^n -> R^{m_q} ("quantities").

    Each quantity has a residual r_q(x) = q(x) - s_q with a copied shift s_q,
    and a role (QuantityType). The problem is

        min  f(x) = sum_{q in OBJ} w_q^T r_q(x)
        s.t. h(x) = [r_q]_{q in EQ} = 0,
             g(x) = [r_q]_{q in LE}, [-r_q]_{q in GE} <= 0,

    where a constraint quantity contributes its m_q rows or, if weighted, the
    single row w_q^T r_q. Constraints are meant to be a few global ones; use
    SetDofBounds() for pointwise bounds. Inactive quantities contribute no
    rows and are not evaluated. Without an objective, f = 0.

    Solvers use three views, f, h and g (see GetObjectiveOperator()). As in
    MFEM, GetGradient() returns the derivative, whose MultTranspose() yields a
    covector; solvers apply the Riesz map (SetRieszMap()) to obtain a
    gradient.

    Roles can be changed at any time. The row layout of h and g is updated
    immediately and GetLayoutSequence() is incremented. Rows are ordered by
    quantity id, then by output index.

    @note OptProblem performs no MPI communication and no caching, and holds
    no communicator: quantities must return globally reduced values from
    Mult() and from the Mult() of their derivatives, and local covectors from
    MultTranspose(); solvers reduce over their own communicator. Roles and
    activity must be identical on all ranks. */
class OptProblem
{
public:
   enum class QuantityType
   {
      /// No role (after AddQuantity, or a replaced objective)
      INVALID,
      /// Part of the objective
      OBJ,
      /// r_q = 0
      EQ,
      /// r_q <= 0
      LE,
      /// r_q >= 0, exposed in g as -r_q <= 0
      GE
   };

   /// @param n Local input size.
   explicit OptProblem(int n);

   /** @brief Register @a op with residual r_q = op(x) - @a shift.
       @note @a op is not owned. The shift is copied.
       @note Second derivatives require that the operator returned by
       op.GetGradient(x) implements GetGradient(x), returning the stacked
       (m_q n) x n operator whose block i is the Hessian of q_i.
       @return Quantity id, initially QuantityType::INVALID and active. */
   int AddQuantity(Operator &op, real_t shift = 0.0);
   /// Same as above with a per-row shift of size op.Height().
   int AddQuantity(Operator &op, const Vector &shift);

   /// Change the shift of quantity @a q (copied); the layout is unchanged.
   void SetShift(int q, real_t s);
   void SetShift(int q, const Vector &s);

   /** @brief Set the objective f = r_q, with m_q = 1.

       Each SetObjective() call replaces the objective. Quantities of the
       previous objective that are not in the new one become
       QuantityType::INVALID; each must be given a constraint type or be
       deactivated before the views are used. The new objective quantities are
       made active and their weights are stored, see
       SetConstraintType(int, QuantityType, bool). */
   void SetObjective(int q);
   /// f = w^T r_q, with |w| = m_q.
   void SetObjective(int q, const Vector &w);
   /// f = sum_k w_k r_{q_k}: one weight per scalar quantity.
   void SetObjective(const Array<int> &q, const Vector &w);
   /// f = sum_k w_k^T r_{q_k}, with |w_k| = m_{q_k}.
   void SetObjective(const Array<int> &q, const std::vector<Vector> &w);

   /** @brief Make quantity @a q an active constraint of type @a t (EQ, LE or
       GE).
       @param use_weights If false, q contributes m_q rows r_q; if true, the
       single row w_q^T r_q with the stored weights. */
   void SetConstraintType(int q, QuantityType t, bool use_weights = false);
   /// Same, with the single row w^T r_q; @a w (|w| = m_q) replaces the
   /// stored weights.
   void SetConstraintType(int q, QuantityType t, const Vector &w);

   /// Switch a quantity on or off, keeping its type. OBJ cannot be switched
   /// off. An inactive quantity may have QuantityType::INVALID.
   void SetActive(int q, bool on);
   void Activate(int q) { SetActive(q, true); }
   void Deactivate(int q) { SetActive(q, false); }

   /** @brief Set the dofwise bounds lo <= x <= hi (copied, keeping the
       host/device side of the input). Use -/+infinity() for an unbounded
       side. The layout is unchanged. */
   void SetDofBounds(const Vector &lo, const Vector &hi);
   /// Uniform bounds, stored on the device if @a use_device is true.
   void SetDofBounds(real_t lo, real_t hi, bool use_device = false);
   bool HasDofBounds() const { return has_dof_bounds; }
   const Vector &GetDofLowerBound() const;
   const Vector &GetDofUpperBound() const;

   /** @brief Set the Riesz map R: covector -> vector (n x n, not owned), e.g.
       the inverse of a mass matrix, which solvers apply to derivatives to
       obtain gradients; without one they use the identity. */
   void SetRieszMap(const Operator &R);
   bool HasRieszMap() const { return riesz != nullptr; }
   const Operator &GetRieszMap() const;

   int NumQuantities() const { return (int)quantities.size(); }
   QuantityType GetType(int q) const { return Get(q).type; }
   bool IsActive(int q) const { return Get(q).active; }
   /// Output size m_q of quantity @a q.
   int GetHeight(int q) const { return Get(q).op->Height(); }

   /** @brief Block offsets of h / g, of size NumQuantities() + 1, with block i
       belonging to quantity i (as used by BlockVector). Block q is empty
       unless q is an active quantity of that view; its size is 1 if weighted
       and m_q otherwise. Valid until GetLayoutSequence() changes. */
   const Array<int> &GetEqOffsets() const { return eq_offsets; }
   const Array<int> &GetIneqOffsets() const { return ineq_offsets; }

   /// Half-open row range [start, end) of quantity @a q in h / g.
   std::tuple<int, int> GetEqRows(int q) const;
   std::tuple<int, int> GetIneqRows(int q) const;

   /// Incremented whenever roles, activity or the row layout change.
   long GetLayoutSequence() const { return sequence; }
   /// Verify that no active quantity has QuantityType::INVALID. Called by
   /// the views.
   void Validate() const;

   int Width() const { return n; }
   int NumEq() const { return eq_offsets.Last(); }
   int NumIneq() const { return ineq_offsets.Last(); }

   /** @brief Views f: R^n -> R^1, h: R^n -> R^{NumEq()} and
       g: R^n -> R^{NumIneq()}.

       Mult(x, y) evaluates the included quantities at @a x. The views are
       owned by the problem and their heights follow the layout.

       D = view.GetGradient(x) is the derivative at @a x (Height() x n): Mult()
       is the JVP and MultTranspose() the VJP. Every included quantity is
       applied, also for a zero seed. D.GetGradient(x) is the second
       derivative, stacked per row: (Height() n) x n, block i being the
       Hessian of row i. Its MultTranspose(Y) returns sum_i H_i^T Y_i; see
       ContractedOperator for a multiplier-weighted Hessian.

       @note D and its second derivative are reused by the view: they are
       invalidated by the next GetGradient() call on the view (or on D) and by
       any role change (checked). They may hold references to @a x, which must
       not change while they are used.
       @note Quantities that cache state (UpdatableOperator) expect Update()
       to be called with each new iterate before the views are evaluated. */
   const Operator &GetObjectiveOperator() const;
   const Operator &GetEqualityConstraintOperator() const;
   const Operator &GetInequalityConstraintOperator() const;

   /** @brief Notify every UpdatableOperator quantity, active or not, of the
       new iterate @a x. A quantity registered twice is notified twice.
       @return The new point sequence. */
   long Update(const Vector &x);
   /// Point sequence of the last Update(), 0 before the first.
   long GetPointSequence() const { return point_sequence; }

   ~OptProblem();
   OptProblem(const OptProblem &) = delete;
   OptProblem &operator=(const OptProblem &) = delete;

private:
   // Convenient classes for the objective, equality and inequality views
   class GroupedOperator;
   class GroupedDerivative;
   class GroupedSecondDerivative;

   // data wrapped by a quantity id
   struct Quantity
   {
      Operator *op = nullptr;
      /// size m_q
      Vector shift;
      /// stored weights, size 0 if none
      Vector w;
      QuantityType type = QuantityType::INVALID;
      bool active = true;
      /// constraint is the single row w^T r_q
      bool weighted = false;
      /// work buffer, size m_q
      mutable Vector val;
      /// work buffer, size m_q n
      mutable Vector stack;
   };

   const Quantity &Get(int q) const;
   Quantity &Get(int q);
   void UpdateLayout();

   int n;
   std::vector<Quantity> quantities;

   Array<int> eq_offsets, ineq_offsets;
   long sequence = 0;
   mutable bool validated = false;

   Vector dof_lo, dof_hi;
   bool has_dof_bounds = false;
   const Operator *riesz = nullptr;

   long point_sequence = 0;

   /// The views f, h and g.
   std::unique_ptr<GroupedOperator> objective, equality, inequality;
   /// work buffer, size n
   /// subject to deprecation after we have scratch space.
   mutable Vector dx_q;
};

} // namespace mfem

#endif // MFEM_OPTPROBLEM_HPP
