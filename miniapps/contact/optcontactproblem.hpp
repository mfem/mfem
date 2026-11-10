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
private:
   /// MPI communicator for the problem.
   MPI_Comm comm;

   /// Underlying elasticity problem (not owned).
   ElasticityOperator * problem = nullptr;

   /// Finite element space for displacements
   ParFiniteElementSpace * vfes = nullptr;

   /// Dimensions: displacement, slack, constraint and gap variables.
   int dimU, dimM, dimC, dimG;

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

   /// Offsets for block vector partitioning of constraints.
   Array<int> block_offsetsg;

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
   int GetDimU() {return dimU;}

   /// Return slack variable dimension.
   int GetDimM() {return dimM;}

   /// Return constraint space dimension.
   int GetDimC() {return dimC;}

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

   /// Hessian of the objective wrt displacement
   HypreParMatrix * Duuf(const BlockVector &x) {return DddE(x.GetBlock(0));}

   /// Jacobian of the affine constraints wrt displacement; independent of x
   /// until FormContactSystem() or ActivateBoundConstraints() is called.
   HypreParMatrix * Duc(const BlockVector &);

   /// Cached transpose of Duc(), owned here and invalidated when the contact
   /// system is rebuilt or bound constraints are activated. If constraints
   /// become iterate-dependent, Duc() must call ResetConstraintJacobians()
   /// before updating its Jacobian for a new linearization.
   HypreParMatrix * DucTranspose(const BlockVector &);

   /// Return transfer operator from contact to displacement subspace.
   HypreParMatrix * GetContactSubspaceTransferOperator();

   /// Evaluate gap function
   void g(const Vector &, Vector &);

   /// Evaluate contact constraints
   void c(const BlockVector &, Vector &);

   /// Compute objective functional value.
   real_t CalcObjective(const BlockVector &, int &);

   /// Compute gradient of objective functional.
   void CalcObjectiveGrad(const BlockVector &, BlockVector &);

   /// Evaluate elastic energy functional.
   real_t E(const Vector & d, int & eval_err);

   /// Evaluate gradient of energy functional.
   void DdE(const Vector & d, Vector & gradE);

   /// Return Hessian of energy functional.
   HypreParMatrix * DddE(const Vector & d);

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

   /// Get lumped mass weights for contact and volume spaces.
   void GetLumpedMassWeights(Vector & Mcslump_, Vector & Mvlump_)
   {
      Mcslump_.SetSize(Mcslump.Size()); Mcslump_ = 0.0;
      Mcslump_.Set(1.0, Mcslump);
      Mvlump_.SetSize(Mvlump.Size()); Mvlump_ = 0.0;
      Mvlump_.Set(1.0, Mvlump);
   };
   ~OptContactProblem() = default;
};

}

#endif
