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

#ifndef MFEM_OPT_CONTACT_PROBLEM
#define MFEM_OPT_CONTACT_PROBLEM

#include "elastoperator.hpp"
#include "axom/slic.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include <memory>

namespace mfem
{

/// Boundary attribute sets for one contact interface.
struct ContactInterface
{
   /// Boundary attributes of the mortar surface.
   std::set<int> mortar_attrs;

   /// Boundary attributes of the non-mortar surface.
   std::set<int> nonmortar_attrs;
};

/** @class OptContactProblem
 *  @brief Contact optimization problem with mortar and non-mortar interfaces.
 *
 *  This class formulates and manages a parallel finite element contact problem
 *  built on top of an `ElasticityOperator`. It sets up the contact system using
 *  Tribol and provides operators for objective evaluation, gradients, Hessians,
 *  and constraints.
 *
 *  The gap constraints are affine within each contact solve:
 *  g(u) = J*(u - xref) + gapv. FormContactSystem() rebuilds J through Tribol
 *  and invalidates the cached constraint operators. Nonlinear elasticity
 *  does not change this constraint linearization.
 *
 *  Features include:
 * - Construction of gap constraints and Jacobians through Tribol.
 * - Optional bound constraints.
 * - Objective and gradient evaluation for optimization solvers.
 */
class OptContactProblem
{
public:
   /// Local constraint blocks and their lumped integration weights. There is
   /// one slack per constraint; blocks are gap, then optional lower/upper bounds.
   struct ConstraintLayout
   {
      enum Block { Gap = 0, LowerBound = 1, UpperBound = 2 };
      Array<int> block_offsets;
      Vector lumped_weights;

      int Size() const { return block_offsets.Last(); }
   };

private:
   /// MPI communicator for the problem.
   MPI_Comm comm;

   /// Underlying elasticity problem (not owned).
   ElasticityOperator * problem = nullptr;

   /// Finite element space for displacements
   ParFiniteElementSpace * vfes = nullptr;

   /// Local displacement and gap dimensions.
   int dimU, dimG;

   /// Owned layout/weights precede work vectors that reference its offsets.
   ConstraintLayout constraint_layout;

   /// Global number of constraints.
   int num_constraints;

   /// Energy value at reference configuration.
   real_t energy_ref;

   /// Energy gradient at reference configuration.
   Vector  grad_ref;

   /// Reference configuration displacement vector.
   Vector xref;

   /// Reference configuration displacement vector with updated BCs.
   Vector xrefbc;

   /// Reusable displacement difference, matrix action, and constraint buffers.
   Vector displacement_work, energy_work;
   BlockVector constraint_work;

   /// Gap vector on contact interface.
   Vector gapv;

   /// Contact surfaces, one entry per interface.
   /// Each interface is handled by its own Tribol coupling scheme so that
   /// faces of different interfaces are never paired with each other.
   std::vector<ContactInterface> contact_interfaces;

   /// Reference stiffness (Hessian) matrix, owned by the elasticity problem.
   HypreParMatrix * Kref=nullptr;

   /// Coordinates of mesh nodes (grid function).
   ParGridFunction * coords = nullptr;

   /// Free allocated matrices/vectors.
   void ReleaseMemory();

   /// Clear constraint operators when the contact system or bounds change.
   void ResetConstraintJacobians();

   /// Rebuild partitions, local constraint blocks and weights from active bounds.
   void UpdateConstraintLayout();

   /// Compute gap and its Jacobian using Tribol.
   void ComputeGapJacobian();

   /// Constraint partition offsets (for distributed data).
   Array<HYPRE_BigInt> constraints_starts;

   /// DOF partition offsets (for distributed data).
   Array<HYPRE_BigInt> dof_starts;

   // with additional constraints
   //         [ g ]
   // g_new = [ eps + (d - dl) ]
   //         [ eps - (d - dl) ]
   // there are additional components to the Jacobian
   //         [ J ]
   // J_new = [ I ]
   //         [-I ]

   /// Lumped volume mass vector.
   Vector Mvlump;

   /// Lumped contact surface mass (full).
   Vector Mcslumpfull;

   /// Lumped contact surface mass (reduced, one entry per gap constraint).
   Vector Mcslump;

   /// Bound displacement vector for constraints.
   Vector dl;

   /// Epsilon vector (slack/bounds).
   Vector eps;

   /// Minimum epsilon value (>0).
   real_t eps_min = 1.e-4;

   /// Proximity ratio for Tribol binning.
   real_t tribol_ratio;

   /// Flag: whether bound constraints are enabled.
   bool bound_constraints;

   /// Flag: whether bound constraints have been activated.
   bool bound_constraints_activated = false;

   // Owned operators follow the partition storage they may reference, so they
   // are destroyed before that storage.
   /// Jacobian of the gap function.
   std::unique_ptr<HypreParMatrix> J;
   /// Transpose of gap Jacobian.
   std::unique_ptr<HypreParMatrix> Jt;
   /// Transfer operator from contact space to displacement space.
   std::unique_ptr<HypreParMatrix> Pc;
   /// Identity matrices for bound constraints.
   std::unique_ptr<HypreParMatrix> Iu;
   std::unique_ptr<HypreParMatrix> negIu;
   /// Cached constraint Jacobian with bounds.
   std::unique_ptr<HypreParMatrix> dcdu;
   /// Cached transpose of the full constraint Jacobian. Jt is reserved for
   /// constructing the filtered contact subspace.
   std::unique_ptr<HypreParMatrix> dcduT;

public:
   OptContactProblem(ElasticityOperator * problem_,
                     const std::set<int> & mortar_attrs_,
                     const std::set<int> & nonmortar_attrs_,
                     real_t tribol_ratio_,
                     bool bound_constraints_);

   /// Construct with several contact interfaces.
   OptContactProblem(ElasticityOperator * problem_,
                     const std::vector<ContactInterface> &contact_interfaces_,
                     real_t tribol_ratio_,
                     bool bound_constraints_);

   /// Build contact system, assemble gap Jacobian and mass matrices.
   void FormContactSystem(ParGridFunction * coords_, const Vector & xref);

   /// Return displacement space dimension.
   int GetDimU() const {return dimU;}

   /// Layout and weights are owned here and rebuilt by FormContactSystem() and
   /// ActivateBoundConstraints(). Construct IPSolver after choosing the layout
   /// for the load step. Existing vector views must not outlive a layout rebuild.
   const ConstraintLayout &GetConstraintLayout() const
   {
      MFEM_VERIFY(constraint_layout.block_offsets.Size() >= 2,
                  "Build the contact system before requesting its layout.");
      return constraint_layout;
   }

   /// Fixed displacement-space weights, updated by FormContactSystem().
   const Vector &GetDisplacementWeights() const { return Mvlump; }

   /// Get MPI communicator.
   MPI_Comm GetComm() {return comm ;}

   /// Get distributed constraint partition offsets.
   HYPRE_BigInt * GetConstraintsStarts() {return constraints_starts.GetData();}

   /// Return global number of constraints.
   HYPRE_BigInt GetGlobalNumConstraints() {return num_constraints;}

   /// Get distributed DOF partition offsets.
   HYPRE_BigInt * GetDofStarts() {return dof_starts.GetData();}

   /// Return global number of DOFs (from Jacobian).
   HYPRE_BigInt GetGlobalNumDofs() {return J->GetGlobalNumCols();}

   /// Return underlying elasticity operator.
   ElasticityOperator * GetElasticityOperator() {return problem;}

   /// Borrowed Jacobian of the affine gap-minus-slack constraints with respect
   /// to displacement. It is independent of the iterate and cached until
   /// FormContactSystem() or ActivateBoundConstraints() invalidates it.
   HypreParMatrix *GetConstraintJacobian();

   /// Borrowed cached transpose, invalidated along with GetConstraintJacobian().
   /// An iterate-dependent constraint model would require a new linearization
   /// and invalidation of both operators before use by the optimizer.
   HypreParMatrix *GetConstraintJacobianTranspose();

   /// Return transfer operator from contact to displacement subspace.
   HypreParMatrix * GetContactSubspaceTransferOperator();

   /// Evaluate the affine gap function at the displacement.
   void EvaluateGap(const Vector &, Vector &);

   /// Evaluate gap and optional bounds minus slack, in GetConstraintLayout order.
   void EvaluateConstraints(const BlockVector &, Vector &);

   /// Potential energy including applied loads. For nonlinear elasticity this
   /// is the fixed quadratic approximation refreshed by FormContactSystem().
   real_t GetEnergy(const Vector &displacement, int &eval_err);

   /// Displacement gradient of GetEnergy; the objective is independent of slack.
   void GetEnergyGradient(const Vector &displacement, Vector &gradient);

   /// Borrowed displacement Hessian of GetEnergy. The elasticity problem owns
   /// it; FormContactSystem() refreshes the nonlinear reference approximation.
   HypreParMatrix *GetEnergyHessian(const Vector &displacement);

   /// Update displacement and eps for bound constraints.
   void SetDisplacement(const Vector & dx, bool active_constraints);

   /// Activate bound constraints (if enabled).
   void ActivateBoundConstraints();

   /// Get Gap, its Jacobian and the lumped contact surface mass from Tribol
   /// for a single contact interface.
   HypreParMatrix *  SetupTribol(ParMesh * pmesh, ParGridFunction * coords,
                                 const Array<int> & ess_tdofs,
                                 const std::set<int> & mortar_attrs,
                                 const std::set<int> & non_mortar_attrs,
                                 Vector &gap, Vector &mcslump,
                                 real_t tribol_ratio);

   ~OptContactProblem() = default;
};

}

#endif
