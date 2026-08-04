#pragma once
#include "mfem.hpp"
namespace mfem
{

class StackedOperator : public Operator
{
public:
   StackedOperator(int m=0): Operator(0, m), offsets{0} {}

   /// @brief Add an operator to the stack
   /// @param op The operator to add to the stack (ownership is transferred)
   virtual int AddOperator(Operator *op)
   {
      MFEM_VERIFY(!finalized, "Operator is finalized");
      MFEM_VERIFY(op, "Operator is null");
      MFEM_VERIFY(op->Width() == width, "Operator width inconsistent");
      height += op->Height();
      offsets.Append(op->Height());
      ops.emplace_back(op);
      return offsets.Size()-1;
   }

   /// @brief Finalize the stack of operators and create the BlockOperator
   void Finalize()
   {
      MFEM_VERIFY(!finalized, "Operator already been finalized");
      offsets.PartialSum();
      Array<int> col_offset({0, width});
      blk_op.reset(new BlockOperator(offsets, col_offset));
      for (size_t i=0; i<ops.size(); i++)
      {
         blk_op->SetBlock(i, 0, ops[i].get());
      }
      finalized = true;
   }

   bool IsFinalized() const { return finalized; }

   /// @brief Return a reference to the BlockOperator
   BlockOperator &AsBlockOperator() const
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      return *blk_op;
   }

   /// @brief Apply the stacked operator to a vector
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      blk_op->Mult(x, y);
   }

   /// @brief Return derivative
   /// @param x The point at which the gradient is evaluated
   /// @return A reference to the derivative operator
   /// @note The returned operator is assumed to be the derivative.
   /// To get "gradient" (primal vector), apply Riesz map
   Operator &GetGradient(const Vector &x) const override
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      if (!grad_op) { grad_op.reset(new StackedDerivative(*this)); }
      grad_op->SetPoint(x);
      return *grad_op;
   }

   /// @brief Return derivative of the i-th operator in the stack
   /// @param i The index of the operator in the stack
   /// @param x The point at which the gradient is evaluated
   /// @return A reference to the derivative operator of the i-th operator
   /// @note The returned operator is assumed to be the derivative.
   /// To get "gradient" (primal vector), apply Riesz map
   Operator &GetGradient(const int i, const Vector &x) const
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      return ops[i]->GetGradient(x);
   }

private:

   class StackedDerivative : public Operator
   {
   public:
      StackedDerivative(const StackedOperator &prob)
         : Operator(prob.Height(), prob.Width())
         , prob(prob)
      {
         x_.SetSize(prob.Width());
         x_.UseDevice(true);
         dx_.SetSize(prob.Width()); dx_.UseDevice(true);
      }
      void SetPoint(const Vector &x)
      {
         x_ = x;
         grad_ops.clear();
         for (size_t i=0; i<prob.ops.size(); i++)
         {
            grad_ops.push_back(&prob.ops[i]->GetGradient(x_));
         }
      }
      /// @brief Apply Jacobi-vector product (JVP): df = J(x) * dx
      void Mult(const Vector &dx, Vector &df) const override
      {
         MFEM_VERIFY(dx.Size() == prob.Width(), "Input vector size mismatch");
         df.SetSize(prob.Height());
         Vector df_i;
         for (size_t i=0; i<prob.ops.size(); i++)
         {
            df_i.MakeRef(df, prob.offsets[i], prob.ops[i]->Height());
            grad_ops[i]->Mult(dx, df_i);
         }
      }
      /// @brief Apply vector-Jacobi product (VJP): dx = J(x)^T * dy
      /// @note dx is a covector, not a vector
      void MultTranspose(const Vector &dy, Vector &dx) const override
      {
         MFEM_VERIFY(dy.Size() == prob.Height(), "Input vector size mismatch");
         dx.SetSize(prob.Width());
         dx = 0.0;
         Vector dy_i;
         // safe to cast away const since we are not modifying dy, only creating a view
         Vector &dy_view = const_cast<Vector &>(dy);
         for (size_t i=0; i<prob.ops.size(); i++)
         {
            dy_i.MakeRef(dy_view, prob.offsets[i], prob.ops[i]->Height());
            grad_ops[i]->MultTranspose(dy_i, dx_);
            dx += dx_;
         }
      }
   private:
      const StackedOperator &prob; // reference to the parent StackedOperator
      Vector x_; // a copy of the point at which the gradient is evaluated
      std::vector<Operator *> grad_ops; // pointers to gradient operators (not owned)
      mutable Vector dx_;
   };
   mutable std::unique_ptr<StackedDerivative> grad_op;

protected:
   bool finalized = false;
   Array<int> offsets;
   std::vector<std::unique_ptr<Operator>> ops;
   std::unique_ptr<BlockOperator> blk_op;
};

class OptimProblem : public StackedOperator
{
public:
   enum class ConstType
   {
      EQ, // equality constraint
      LE, // less than or equal constraint
      OBJ, // objective function
   };

   OptimProblem(int m=0): StackedOperator(m) {}

   int AddOperator(Operator *op) override
   {
      MFEM_ABORT("Use SetObjective or AddConstraint "
                 "to add operators to the optimization problem");
      return -1;
   }

   /// @brief Add a constraint operator to the optimization problem
   /// @param con The constraint operator to add to the problem (ownership is transferred)
   /// @param type The type of constraint (equality or inequality)
   /// @return The index of the added constraint operator in the stack
   /// @note If the input is the objective, use ConstType::OBJ
   int AddConstraint(Operator *con, ConstType type)
   {
      MFEM_VERIFY(con, "Constraint operator is null");
      MFEM_VERIFY(!finalized, "Operator is finalized");
      constraint_types.push_back(Array<ConstType>(con->Height()));
      constraint_types.back() = type;
      return StackedOperator::AddOperator(con);
   }

   /// @brief Alias for AddConstraint with a single constraint type
   int AddOperator(Operator *con, ConstType type)
   {
      return AddConstraint(con, type);
   }

   /// @brief Add a constraint operator to the optimization problem with multiple constraint types
   /// @param con The constraint operator to add to the problem (ownership is transferred)
   /// @param type The types of constraints (equality or inequality) for each output of
   /// the operator
   /// @return The index of the added constraint operator in the stack
   int AddConstraint(Operator *con, const Array<ConstType> &type)
   {
      MFEM_VERIFY(con, "Constraint operator is null");
      MFEM_VERIFY(!finalized, "Operator is finalized");
      MFEM_VERIFY(type.Size() == con->Height(),
                  "Constraint type count must match operator height");
      constraint_types.push_back(type);
      return StackedOperator::AddOperator(con);
   }
   /// @brief Alias for AddConstraint with multiple constraint types
   int AddOperator(Operator *con, const Array<ConstType> &type)
   {
      return AddConstraint(con, type);
   }

   /// @brief Set the objective operator for the optimization problem
   /// @param obj The objective operator (ownership is transferred)
   /// @return The index of the objective operator in the stack
   /// @note The objective operator must have height 1. If one of the outputs is an objective,
   /// use AddConstraint and ReplaceObjective to specify the objective index.
   int SetObjective(Operator *obj)
   {
      MFEM_VERIFY(obj, "Objective operator is null");
      MFEM_VERIFY(!finalized, "Operator is finalized");
      MFEM_VERIFY(obj->Height() == 1,
                  "Objective operator must have height 1. "
                  "If one of the output is an objective, "
                  "use AddConstraint and ReplaceObjective to specify the objective index.");
      obj_blk_idx = AddOperator(obj, ConstType::OBJ);
      obj_loc_idx = 0;
      return obj_blk_idx;
   }

   /// @brief Move the objective designation to a different stacked output
   /// @param org_obj_block Block index currently tagged ConstType::OBJ
   /// @param org_obj_loc_idx Output index within that block currently tagged OBJ
   /// @param obj_block Block index of the new objective output
   /// @param obj_loc_idx_ Output index within the new block (default is 0)
   /// @param obj_new_type Type assigned to the demoted original output
   /// (default ConstType::LE)
   void ReplaceObjective(int org_obj_block, int org_obj_loc_idx,
                         int obj_block, int obj_loc_idx_,
                         ConstType obj_new_type = ConstType::LE)
   {
      MFEM_VERIFY(org_obj_block >= 0 && org_obj_block < constraint_types.size(),
                  "Original objective block index out of bounds");
      MFEM_VERIFY(org_obj_loc_idx >= 0 &&
                  org_obj_loc_idx < constraint_types[org_obj_block].Size(),
                  "Original objective index out of bounds");
      MFEM_VERIFY(obj_block >= 0 && obj_block < ops.size(),
                  "Objective block index out of bounds");
      MFEM_VERIFY(obj_loc_idx_ >= 0, "Objective index must be non-negative");
      MFEM_VERIFY(obj_loc_idx_ < ops[obj_block]->Height(),
                  "Objective index out of bounds");
      MFEM_VERIFY(constraint_types[org_obj_block][org_obj_loc_idx] == ConstType::OBJ,
                  "Original objective index does not correspond to an objective");
      constraint_types[org_obj_block][org_obj_loc_idx] = obj_new_type;
      constraint_types[obj_block][obj_loc_idx_] = ConstType::OBJ;
      obj_blk_idx = obj_block;
      obj_loc_idx = obj_loc_idx_;
   }

   /// @brief Evaluate the objective function at a given point x
   /// @note The objective block is tagged ConstType::OBJ; obj_blk_idx and
   /// obj_loc_idx select which stacked output holds the objective value.
   real_t Objective(const Vector &x) const
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      aux_y.SetSize(ops[obj_blk_idx]->Height());
      ops[obj_blk_idx]->Mult(x, aux_y);
      return aux_y(obj_loc_idx);
   }

   /// @brief Evaluate the energy (objective function) at a given point x
   /// @note This replicates the NonlinearForm::GetEnergy interface.
   real_t GetEnergy(const Vector &x) const
   {
      return Objective(x);
   }

   const Array<ConstType> &GetConstraintType(int con_block) const
   {
      MFEM_VERIFY(con_block >= 0 && con_block < constraint_types.size(),
                  "Constraint block index out of bounds");
      return constraint_types[con_block];
   }

   /// @brief Classify every stacked output by its ConstType into global-row
   /// index lists (row = position in the stacked Mult output).
   /// @param eq_rows Filled with the global rows tagged ConstType::EQ
   /// @param le_rows Filled with the global rows tagged ConstType::LE
   /// @param obj_row Set to the global row tagged ConstType::OBJ (-1 if none)
   void ClassifyRows(Array<int> &eq_rows, Array<int> &le_rows,
                     int &obj_row) const
   {
      eq_rows.SetSize(0);
      le_rows.SetSize(0);
      obj_row = -1;
      int base = 0;
      for (size_t b = 0; b < constraint_types.size(); b++)
      {
         const Array<ConstType> &t = constraint_types[b];
         for (int k = 0; k < t.Size(); k++)
         {
            switch (t[k])
            {
               case ConstType::EQ:  eq_rows.Append(base + k); break;
               case ConstType::LE:  le_rows.Append(base + k); break;
               case ConstType::OBJ: obj_row = base + k; break;
            }
         }
         base += t.Size();
      }
   }

   /// @brief Set the lower bound for the optimization variables (dof)
   /// @param lb The lower bound vector (will be copied)
   void SetDofLowerBound(const Vector &lb)
   {
      MFEM_VERIFY(lb.Size() == width, "Lower bound size mismatch");
      dof_lb.SetSize(width);
      dof_lb = lb;
   }

   /// @brief Set the upper bound for the optimization variables (dof)
   /// @param ub The upper bound vector (will be copied)
   void SetDofUpperBound(const Vector &ub)
   {
      MFEM_VERIFY(ub.Size() == width, "Upper bound size mismatch");
      dof_ub.SetSize(width);
      dof_ub = ub;
   }

   const Vector &GetDofLowerBound() const { return dof_lb; }
   const Vector &GetDofUpperBound() const { return dof_ub; }

   /// @brief Set the upper and lower bounds for the optimization variables (dof)
   /// @param lb The lower bound vector (will be copied)
   /// @param ub The upper bound vector (will be copied)
   void SetDofBounds(const Vector &lb, const Vector &ub)
   {
      MFEM_VERIFY(lb.Size() == width, "Lower bound size mismatch");
      MFEM_VERIFY(ub.Size() == width, "Upper bound size mismatch");
      dof_lb.SetSize(width);
      dof_lb = lb;
      dof_ub.SetSize(width);
      dof_ub = ub;
   }
   bool HasDofLowerBound() const { return dof_lb.Size() > 0; }
   bool HasDofUpperBound() const { return dof_ub.Size() > 0; }
   bool HasDofBounds() const { return dof_lb.Size() > 0 && dof_ub.Size() > 0; }


   /// @brief Check if the inner product operator is set
   /// @return true if the inner product operator is set, false otherwise
   /// @note The inner product operator is used to define the inner product in the optimization problem.
   bool HasInnerProduct() const { return dot_prod != nullptr; }
   InnerProductOperator &GetInnerProduct() const
   {
      MFEM_VERIFY(dot_prod, "Inner product operator not set");
      return *dot_prod;
   }
   /// @brief Set the inner product operator for the optimization problem
   /// @param dot The inner product operator (ownership is transferred)
   /// @note The inner product operator is used to define the inner product in the optimization problem.
   void SetInnerProduct(InnerProductOperator *dot)
   {
      dot_prod.reset(dot);
   }
   /// @brief Check if the Riesz map operator is set
   /// @return true if the Riesz map operator is set, false otherwise
   /// @note The Riesz map operator is used to map the derivative of the objective function to the primal space.
   bool HasRieszMap() const
   {
      return riesz_map != nullptr;
   }
   /// @brief Set the Riesz map operator for the optimization problem
   /// @param riesz The Riesz map operator (ownership is transferred)
   /// @note The Riesz map operator is used to map the derivative of the objective function to the primal space.
   void SetRieszMap(Operator *riesz) { riesz_map.reset(riesz); }
   Operator &GetRieszMap() const
   {
      MFEM_VERIFY(riesz_map, "Riesz map not set");
      return *riesz_map;
   }

private:
   std::vector<Array<ConstType>> constraint_types;
   int obj_blk_idx = -1;
   int obj_loc_idx = -1;
   mutable Vector aux_y;

   Vector dof_lb;
   Vector dof_ub;

   // Inner product operator
   std::unique_ptr<InnerProductOperator> dot_prod;
   // Riesz map operator (if any)
   std::unique_ptr<Operator> riesz_map;
};

/// @brief Adapter exposing an OptimProblem as an mfem::OptimizationProblem, so
/// it can be handed to an OptimizationSolver (SLBQP, HiOp, ...).
///
/// The stacked outputs of the OptimProblem are split by ConstType:
///   - OBJ row  -> objective F(x)     (CalcObjective / CalcObjectiveGrad)
///   - EQ  rows -> equality operator   C(x) = 0
///   - LE  rows -> inequality operator -infinity <= D(x) <= 0
/// EQ and LE are relative to a zero right-hand side, hence c_e = 0, d_hi = 0,
/// and d_lo = -infinity.
///
/// @note Gradients of C and D are assembled as (n_c x m) DenseMatrices (as
/// required by HiOp); this is cheap because the number of constraints is small.
/// Derivatives are returned as raw sensitivities (Euclidean); any inner product
/// / Riesz map is left to the solver, consistent with OptimProblem.
/// @note This is meant to be owned by the caller (e.g. a stack object);
/// OptimizationProblem has no virtual destructor, so do not delete via a base
/// pointer.
class StackedOptimizationProblem : public OptimizationProblem
{
public:
   StackedOptimizationProblem(OptimProblem &prob)
      : OptimizationProblem(prob.Width(), nullptr, nullptr), prob(prob)
   {
      MFEM_VERIFY(prob.IsFinalized(),
                  "OptimProblem must be finalized before wrapping");

      Array<int> eq_rows, le_rows;
      int obj_row = -1;
      prob.ClassifyRows(eq_rows, le_rows, obj_row);
      MFEM_VERIFY(obj_row >= 0, "No objective set on the OptimProblem");

      // Unit seed selecting the objective row for the VJP in CalcObjectiveGrad.
      obj_seed.SetSize(prob.Height());
      obj_seed.UseDevice(true);
      obj_seed = 0.0;
      {
         Array<int> idx(1);
         idx[0] = obj_row;
         obj_seed.SetSubVector(idx, 1.0);
      }

      // Equality operator: C(x) = 0.
      if (eq_rows.Size() > 0)
      {
         eq_op.reset(new SelectedRows(prob, eq_rows));
         C = eq_op.get();
         c_e_vec.SetSize(eq_rows.Size());
         c_e_vec = 0.0;
         SetEqualityConstraint(c_e_vec);
      }

      // Inequality operator: -infinity <= D(x) <= 0.
      if (le_rows.Size() > 0)
      {
         ineq_op.reset(new SelectedRows(prob, le_rows));
         D = ineq_op.get();
         d_lo_vec.SetSize(le_rows.Size());
         d_lo_vec = -infinity();
         d_hi_vec.SetSize(le_rows.Size());
         d_hi_vec = 0.0;
         SetInequalityConstraint(d_lo_vec, d_hi_vec);
      }

      // Variable bounds x_lo <= x <= x_hi (unset side filled with +/-infinity).
      if (prob.HasDofLowerBound() || prob.HasDofUpperBound())
      {
         x_lo_vec.SetSize(prob.Width());
         x_hi_vec.SetSize(prob.Width());
         if (prob.HasDofLowerBound()) { x_lo_vec = prob.GetDofLowerBound(); }
         else { x_lo_vec = -infinity(); }
         if (prob.HasDofUpperBound()) { x_hi_vec = prob.GetDofUpperBound(); }
         else { x_hi_vec = infinity(); }
         SetSolutionBounds(x_lo_vec, x_hi_vec);
      }
   }

   real_t CalcObjective(const Vector &x) const override
   {
      return prob.Objective(x);
   }

   void CalcObjectiveGrad(const Vector &x, Vector &grad) const override
   {
      // grad = J_obj^T e_{obj} : objective-row derivative (dual/sensitivity).
      prob.GetGradient(x).MultTranspose(obj_seed, grad);
   }

private:
   /// Operator returning a subset of the stacked outputs, selected by row index.
   /// Mult() gathers the selected rows; GetGradient() assembles the dense
   /// (n_rows x m) Jacobian of those rows via the adjoint (VJP).
   class SelectedRows : public Operator
   {
   public:
      SelectedRows(OptimProblem &prob, const Array<int> &rows)
         : Operator(rows.Size(), prob.Width()), prob(prob), rows(rows)
      {
         full.SetSize(prob.Height()); full.UseDevice(true);
         seed.SetSize(prob.Height()); seed.UseDevice(true);
         col.SetSize(prob.Width());   col.UseDevice(true);
         jac.SetSize(rows.Size(), prob.Width());
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         prob.Mult(x, full);
         full.GetSubVector(rows, y);
      }

      Operator &GetGradient(const Vector &x) const override
      {
         Operator &der = prob.GetGradient(x);   // (n x m) derivative operator
         Array<int> idx(1);
         for (int i = 0; i < rows.Size(); i++)
         {
            seed = 0.0;
            idx[0] = rows[i];
            seed.SetSubVector(idx, 1.0);         // e_{rows[i]}
            der.MultTranspose(seed, col);        // col = J^T e   (size m)
            col.HostRead();
            jac.SetRow(i, col);                  // row i = gradient of output i
         }
         return jac;
      }

   private:
      OptimProblem &prob;
      Array<int> rows;
      mutable Vector full, seed, col;
      mutable DenseMatrix jac;
   };

   OptimProblem &prob;
   std::unique_ptr<Operator> eq_op, ineq_op;
   Vector obj_seed;
   Vector c_e_vec, d_lo_vec, d_hi_vec, x_lo_vec, x_hi_vec;
};

}

