#include "mfem.hpp"

namespace mfem
{

class StackedOperator : public Operator
{
public:
   StackedOperator(int m=0): Operator(0, m), offset{0} {}

   virtual int AddOperator(Operator &op)
   {
      MFEM_VERIFY(!finalized, "Operator is finalized");
      MFEM_VERIFY(op.Width() == width, "Operator width inconsistent");
      offset.Append(op.Height());
      ops.Append(&op);
      return ops.Size()-1;
   }

   void Finalize()
   {
      MFEM_VERIFY(!finalized, "Operator already been finalized");
      offset.PartialSum();
      Array<int> col_offset({0, width});
      blk_op.reset(new BlockOperator(offset, col_offset));
      for (int i=0; i<ops.Size(); i++)
      {
         blk_op->SetBlock(i, 0, ops[i]);
      }
   }

   bool IsFinalized() const { return finalized; }

   BlockOperator &AsBlockOperator() const
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      return *blk_op;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      blk_op->Mult(x, y);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      if (!grad_op) { grad_op.reset(new ProblemGradient(*this)); }
      grad_op->SetPoint(x);
      return *grad_op;
   }

   Operator &GetGradient(const int i, const Vector &x) const
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      return ops[i]->GetGradient(x);
   }

private:

   class ProblemGradient : public Operator
   {
   public:
      ProblemGradient(const StackedOperator &prob)
         : Operator(prob.Width(), prob.Width())
         , prob(prob)
      {}
      void SetPoint(const Vector &x) { x_ = x; }
      void Mult(const Vector &x, Vector &y) const override
      {
         //
      }
   private:
      const StackedOperator &prob;
      Vector x_;
   };

protected:
   bool finalized = false;
   Array<int> offset;
   Array<Operator *> ops;
   std::unique_ptr<BlockOperator> blk_op;
   mutable std::unique_ptr<ProblemGradient> grad_op;
};

class OptimProblem : public StackedOperator
{
   enum class ConstType
   {
      EQ, // equality constraint
      LE, // less than or equal constraint
   };

   int AddOperator(Operator &op) override
   {
      MFEM_ABORT("Use SetObjective or AddConstraint to add operators to the optimization problem");
      return -1;
   }

   int SetObjective(Operator &obj, int obj_idx=0)
   {
      MFEM_VERIFY(!finalized, "Operator is finalized");
      MFEM_VERIFY(obj_blk_idx == -1, "Objective already set");
      MFEM_VERIFY(obj_idx >= 0, "Objective index must be non-negative");
      MFEM_VERIFY(obj_idx < ops.Size(), "Objective index out of bounds");
      obj_loc_idx = obj_idx;
      obj_blk_idx = StackedOperator::AddOperator(obj);
      return obj_blk_idx;
   }

   int AddConstraint(Operator &con, ConstType type, int con_idx=0)
   {
      MFEM_VERIFY(!finalized, "Operator is finalized");
      MFEM_VERIFY(con_idx >= 0, "Constraint index must be non-negative");
      MFEM_VERIFY(con_idx < ops.Size(), "Constraint index out of bounds");
      constraint_types.Append(type);
      return StackedOperator::AddOperator(con);
   }

   void UpdateObjectiveIndex(int obj_block, int obj_loc_idx_=0)
   {
      MFEM_VERIFY(obj_block >= 0 && obj_block < ops.Size(),
                  "Objective block index out of bounds");
      MFEM_VERIFY(obj_loc_idx_ >= 0, "Objective index must be non-negative");
      MFEM_VERIFY(obj_loc_idx_ < ops[obj_block]->Height(),
                  "Objective index out of bounds");
      obj_blk_idx = obj_block;
      obj_loc_idx = obj_loc_idx_;
   }

   real_t GetEnergy(const Vector &x) const
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      aux_y.SetSize(ops[obj_blk_idx]->Height());
      ops[obj_blk_idx]->Mult(x, aux_y);
      return aux_y(obj_loc_idx);
   }

   real_t Objective(const Vector &x) const
   {
      return GetEnergy(x);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(finalized, "Operator not finalized");
      blk_op->Mult(x, y);
   }

   ConstType GetConstraintType(int con_block) const
   {
      MFEM_VERIFY(con_block >= 0 && con_block < constraint_types.Size(),
                  "Constraint block index out of bounds");
      return constraint_types[con_block];
   }

   // @brief Set the lower bound for the optimization variables (dof)
   // @param lb The lower bound vector (will be copied)
   void SetDofLowerBound(const Vector &lb)
   {
      MFEM_VERIFY(lb.Size() == width, "Lower bound size mismatch");
      dof_lb.UseDevice(true);
      dof_lb.SetSize(width);
      dof_lb = lb;
   }

   // @brief Set the upper bound for the optimization variables (dof)
   // @param ub The upper bound vector (will be copied)
   void SetDofUpperBound(const Vector &ub)
   {
      MFEM_VERIFY(ub.Size() == width, "Upper bound size mismatch");
      dof_ub.SetSize(width);
      dof_ub = ub;
   }

   // @brief Set the upper and lower bounds for the optimization variables (dof)
   // @param lb The lower bound vector (will be copied)
   // @param ub The upper bound vector (will be copied)
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

private:
   Array<ConstType> constraint_types;
   int obj_blk_idx = -1;
   int obj_loc_idx = -1;
   mutable Vector aux_y;

   Vector dof_lb;
   Vector dof_ub;
};

}

