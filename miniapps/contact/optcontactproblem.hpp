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

namespace mfem
{

/** @class OptContactProblem
 *  @brief Contact optimization problem with mortar and non-mortar interfaces.
 *
 *  This class formulates and manages a parallel finite element contact problem
 *  built on top of an `ElasticityOperator`. It sets up the contact system using
 *  Tribol and provides operators for objective evaluation, gradients, Hessians,
 *  and constraints.
 *
 *  The optional Coulomb mode supplies coupled residuals and Jacobians to a
 *  separate nonlinear solver, using frozen mortar geometry per increment.
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

   /// Underlying elasticity problem.
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

   /// Gap vector on contact interface.
   Vector gapv;

   /// Mortar and non-mortar attribute sets.
   std::set<int> mortar_attrs;
   std::set<int> nonmortar_attrs;

   /// Negative identity matrix (used in constraints).
   HypreParMatrix * NegId = nullptr;

   /// Reference stiffness (Hessian) matrix.
   HypreParMatrix * Kref=nullptr;

   /// Jacobian of the gap function.
   HypreParMatrix * J = nullptr;

   /// Transpose of gap Jacobian.
   HypreParMatrix * Jt = nullptr;

   /// Transfer operator from contact space to displacement space.
   HypreParMatrix * Pc = nullptr;

   /// Coordinates of mesh nodes (grid function).
   ParGridFunction * coords = nullptr;

   /// Free allocated matrices/vectors.
   void ReleaseMemory();

   /// Persistent vector mortar data; the IP path keeps its scalar data above.
   struct CoulombData;
   std::unique_ptr<CoulombData> coulomb_data;

   /// Freeze vector mortar geometry at the accepted displacement.
   void FormCoulombSystem(const Vector &displacement);

   /// Owned nodal Coulomb equations, diagnostics, and selected-branch partials.
   struct CoulombLawResult
   {
      Vector equation, gap, pressure, slip; ///< Local equations and diagnostics.
      Array<int> state; ///< Open=0, stick=1, slip=2.
      /// Projection derivatives with respect to kinematics and multipliers.
      std::unique_ptr<HypreParMatrix> dk, dl;
   };
   /// Calculate valid projection equations; nonzero values are unconverged.
   CoulombLawResult EvaluateCoulombLaw(const Vector &displacement,
                                      bool derivatives) const;

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

   /// Identity matrices for bound constraints.
   HypreParMatrix * Iu = nullptr;
   HypreParMatrix * negIu = nullptr;

   /// Cached constraint Jacobian with bounds.
   HypreParMatrix * dcdu = nullptr;

   /// Mass matrix in the volume.
   HypreParMatrix * Mv = nullptr;

   /// Mass matrix on the contact surface.
   HypreParMatrix * Mcs = nullptr;

   /// Lumped volume mass vector.
   Vector Mvlump;

   /// Lumped contact surface mass (full).
   Vector Mcslumpfull;

   /// Lumped contact surface mass (reduced).
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

public:
   OptContactProblem(ElasticityOperator * problem_,
                     const std::set<int> & mortar_attrs_,
                     const std::set<int> & nonmortar_attrs_,
                     real_t tribol_ratio_,
                     bool bound_constraints_);

   /** @brief Select vector Coulomb mortar before forming contact.
       @param coefficient Finite nonnegative friction coefficient.
       Scalar IP constraints remain the default. Coulomb uses linear bulk
       mechanics and does not use the IP slack or bound-constraint data. */
   void EnableCoulomb(real_t coefficient);
   /// Whether vector Coulomb mortar is selected instead of scalar IP contact.
   bool UsesCoulomb() const { return bool(coulomb_data); }

   /// Mutable Cartesian multiplier input on the dual submesh true DOFs.
   Vector & GetContactMultipliers();
   /// Coupled elasticity and Coulomb equations at a trial displacement.
   Vector CoulombResidual(const Vector &displacement) const;
   /// Owned Jacobian of CoulombResidual, with essential increments eliminated.
   std::unique_ptr<BlockOperator> CoulombJacobian(const Vector &displacement) const;
   /// Check pressure, linearized gap, complementarity, cone, and dissipation.
   bool CheckCoulombContact(const Vector &displacement) const;

   /// Build scalar IP constraints or freeze vector Coulomb mortar kinematics.
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

   /// Return the global number of displacement true DOFs.
   HYPRE_BigInt GetGlobalNumDofs() {return vfes->GlobalTrueVSize();}

   /// Return underlying elasticity operator.
   ElasticityOperator * GetElasticityOperator() {return problem;}

   /// Hessian of the objective wrt displacement
   HypreParMatrix * Duuf(const BlockVector &x) {return DddE(x.GetBlock(0));}

   /// Hessian of the objective wrt the slack variables
   HypreParMatrix * Dmmf(const BlockVector &) {return nullptr;}

   /// Jacobian of the constraints wrt displacement
   HypreParMatrix * Duc(const BlockVector &);

   /// Jacobian of the constraints wrt slack variables
   HypreParMatrix * Dmc(const BlockVector &);

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

   /// Form scalar constraints using Tribol's frozen-geometry mortar formulation.
   HypreParMatrix *  SetupTribol(ParMesh * pmesh, ParGridFunction * coords,
                                 const Array<int> & ess_tdofs,
                                 const std::set<int> & mortar_attrs,
                                 const std::set<int> & non_mortar_attrs,
                                 Vector &gap,  real_t tribol_ratio);

   /// Get lumped mass weights for contact and volume spaces.
   void GetLumpedMassWeights(Vector & Mcslump_, Vector & Mvlump_)
   {
      Mcslump_.SetSize(Mcslump.Size()); Mcslump_ = 0.0;
      Mcslump_.Set(1.0, Mcslump);
      Mvlump_.SetSize(Mvlump.Size()); Mvlump_ = 0.0;
      Mvlump_.Set(1.0, Mvlump);
   };
   ~OptContactProblem();
};

}

#endif
