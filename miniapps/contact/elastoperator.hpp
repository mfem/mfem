// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ELASTICITY_OPERATOR
#define MFEM_ELASTICITY_OPERATOR

#include "mfem.hpp"

namespace mfem
{

/**
 *  @class ElasticityOperator
 *  @brief Parallel finite element operator for linear and nonlinear elasticity.
 *
 *  This class sets up and evaluates finite element operators for elasticity on a parallel mesh.
 *  It supports both linear and nonlinear (Neo-Hookean) formulations.
 *  Features include evaluation of energy, gradient, and Hessian for use in optimization solvers.
 */
class ElasticityOperator
{
private:
   MPI_Comm comm;

   /// Toggle for nonlinear formulation (true = Neo-Hookean, false = linear elasticity).
   bool nonlinear = false;

   /// Tracks whether the linear system has been formed.
   bool formsystem = false;

   ParMesh * pmesh = nullptr;

   /// Essential boundary attribute markers.
   Array<int> ess_bdr, ess_bdr_attr;

   /// Essential DOFs (true DOF list) and component-based attributes.
   Array<int> ess_tdof_list, ess_bdr_attr_comp;

   /// Polynomial order of the FE basis (default = 1).
   int order=1;

   /// Global number of true DOFs in the FE space.
   int globalntdofs;

   /// Finite element collection (H1).
   FiniteElementCollection * fec = nullptr;

   /// Parallel finite element space for displacement.
   ParFiniteElementSpace * fes = nullptr;

   /// Underlying operator: bilinear form (linear) or nonlinear form (nonlinear).
   Operator * op = nullptr;

   /// Linear form for RHS assembly.
   ParLinearForm * b = nullptr;

   /// Current solution
   ParGridFunction x;

   // System matrix
   HypreParMatrix *K=nullptr;

   /// Right-hand side (B) and solution vector (X).
   Vector B, X;

   /// Neumann pressure coefficient (for traction BCs).
   ConstantCoefficient pressure_cf;

   /// Material parameters:
   /// - Linear: c1 = λ (1ˢᵗ Lame parameter), c2 = μ (2ⁿᵈ Lame parameter or shear modulus)
   /// - Nonlinear: c1 = G (shear modulus), c2 = K (bulk modulus)
   Vector c1, c2;
   PWConstCoefficient c1_cf, c2_cf;

   /// Hyperelastic material model (only for nonlinear case).
   NeoHookeanModel * material_model = nullptr;

   /// Reference configuration displacement
   Vector xref;

   /// Internal setup functions
   void Init();
   void SetEssentialBC();
   void SetUpOperator();

public:
   /**  @brief Construct an ElasticityOperator.
    *   @param pmesh_ Parallel mesh.
    *   @param ess_bdr_attr_ Array of essential boundary attributes.
    *   @param ess_bdr_attr_comp_ Component index for each essential boundary attribute.
    *   @param E Vector of Young’s modulus values (per attribute).
    *   @param nu Vector of Poisson’s ratio values (per attribute).
    *   @param nonlinear_ If true, setup nonlinear hyperelasticity.
    */
   ElasticityOperator(ParMesh * pmesh_, Array<int> & ess_bdr_attr_,
                      Array<int> & ess_bdr_attr_comp_,
                      const Vector & E, const Vector & nu, bool nonlinear_ = false);

   /// Set material parameters from vectors of Young’s modulus (E) and Poisson’s ratio (ν).
   void SetParameters(const Vector & E, const Vector & nu);

   /// Apply Neumann (pressure) boundary condition on a set of boundary markers.
   void SetNeumanPressureData(ConstantCoefficient &f, Array<int> & bdr_marker);

   /// Apply Dirichlet (displacement) boundary condition on a set of boundary markers.
   void SetDisplacementDirichletData(const Vector & delta, Array<int> essbdr);

   /// Assemble and form the linear system (matrix and RHS).
   void FormLinearSystem();

   /// Reset and reassemble the RHS linear form.
   void UpdateRHS();

   ParMesh * GetMesh() const { return pmesh; };
   MPI_Comm GetComm() const { return comm; };

   ParFiniteElementSpace * GetFESpace() const { return fes; };

   int GetGlobalNumDofs() const { return globalntdofs; };

   const Array<int> & GetEssentialDofs() const { return ess_tdof_list; };

   /// Get the displacement with essential boundary conditions applied.
   void Getxrefbc(Vector & xrefbc) const {x.GetTrueDofs(xrefbc);}

   /// Compute the elastic energy functional at a given displacement vector.
   real_t GetEnergy(const Vector & u) const;

   /// Compute the gradient of the energy functional at a given displacement vector.
   void GetGradient(const Vector & u, Vector & gradE) const;

   /// Get the Hessian (stiffness matrix) at a given displacement vector.
   HypreParMatrix * GetHessian(const Vector & u);

   /// Check if the operator is nonlinear.
   bool IsNonlinear() { return nonlinear; }

   /// Destructor (cleans up FE space, operator, and material model).
   ~ElasticityOperator();
};

}

#endif
