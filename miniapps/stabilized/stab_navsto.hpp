// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_STAB_NAVSTO_HPP
#define MFEM_STAB_NAVSTO_HPP

#include "mfem.hpp"
#include "stab_tau.hpp"

namespace mfem
{

/** Stabilized incompressible Navier-Stokes integrator
    Start with Galerkin for stokes - done
    Add convection - done

    Modify diffusion

    Add supg
    Add pspg
    Add lsq

    Add correct inverse estimate

    Leopoldo P. Franca, Sérgio L. Frey
    Stabilized finite element methods:
    II. The incompressible Navier-Stokes equations.
    Computer Methods in Applied Mechanics and Engineering, 99(2-3), 209-233.

    https://doi.org/10.1016/0045-7825(92)90041-H
    https://www.sciencedirect.com/science/article/pii/004578259290041H

*/
class StabInNavStoIntegrator : public BlockNonlinearFormIntegrator
{
private:
   Coefficient *c_mu;
   Vector u;
   DenseMatrix sigma;

   DenseMatrix elf_u, elv_u;
   DenseMatrix elf_p, elv_p;
   Vector sh_u, ushg_u, sh_p;
   DenseMatrix shg_u, grad_u;

   /// The stabilization parameter
   StabType stab;
   Tau *tau = nullptr;
   Tau *delta = nullptr;

   /// The advection field
   VectorCoefficient *adv = nullptr;
public:
   StabInNavStoIntegrator(Coefficient &mu_,
                          Tau &t, Tau &d,
                          StabType s = GALERKIN);

   virtual real_t GetElementEnergy(const Array<const FiniteElement *>&el,
                                   ElementTransformation &Tr,
                                   const Array<const Vector *> &elfun);

   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<Vector *> &elvec);

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array2D<DenseMatrix *> &elmats);
};

class GeneralResidualMonitor : public IterativeSolverMonitor
{
public:
   GeneralResidualMonitor(const std::string& prefix_, int print_lvl)
      : prefix(prefix_)
   {
      print_level = print_lvl;
   }

   virtual void MonitorResidual(int it, real_t norm, const Vector &r, bool final);

private:
   const std::string prefix;
   int print_level;
   mutable real_t norm0;
};

class SystemResidualMonitor : public IterativeSolverMonitor
{
public:
   SystemResidualMonitor(const std::string& prefix_,
                          int print_lvl,
                          Array<int> &offsets)
      : prefix(prefix_), bOffsets(offsets)
   {
      print_level = print_lvl;
      nvar = bOffsets.Size()-1;
      norm0.SetSize(nvar);
   }

   virtual void MonitorResidual(int it, real_t norm, const Vector &r, bool final);

private:
   const std::string prefix;
   int print_level, nvar;
   mutable Vector norm0;
  // Offsets for extracting block vector segments
   Array<int> &bOffsets;
};

// Custom block preconditioner for the Jacobian of the incompressible nonlinear
// elasticity operator. It has the form
//
// P^-1 = [ K^-1 0 ][ I -B^T ][ I  0           ]
//        [ 0    I ][ 0  I   ][ 0 -\gamma S^-1 ]
//
// where the original Jacobian has the form
//
// J = [ K B^T ]
//     [ B 0   ]
//
// and K^-1 is an approximation of the inverse of the displacement part of the
// Jacobian and S^-1 is an approximation of the inverse of the Schur
// complement S = B K^-1 B^T. The Schur complement is approximated using
// a mass matrix of the pressure variables.
class JacobianPreconditioner : public Solver
{
protected:
   // Finite element spaces for setting up preconditioner blocks
   Array<FiniteElementSpace *> spaces;

   // Offsets for extracting block vector segments
   Array<int> &block_trueOffsets;

   // Jacobian for block access
   BlockOperator *jacobian;

   // Scaling factor for the pressure mass matrix in the block preconditioner
   real_t gamma;

   // Objects for the block preconditioner application
   SparseMatrix *pressure_mass;
   Solver *mass_pcg;
   Solver *mass_prec;
   Solver *stiff_pcg;
   Solver *stiff_prec;

public:
   JacobianPreconditioner(Array<FiniteElementSpace *> &fes,
                          SparseMatrix &mass, Array<int> &offsets);

   virtual void Mult(const Vector &k, Vector &y) const;
   virtual void SetOperator(const Operator &op);

   virtual ~JacobianPreconditioner();
};

// After spatial discretization, the rubber model can be written as:
//     0 = H(x)
// where x is the block vector representing the deformation and pressure and
// H(x) is the nonlinear incompressible neo-Hookean operator.
class StabInNavStoOperator : public Operator
{
protected:
   // Finite element spaces
   Array<FiniteElementSpace *> spaces;

   // Block nonlinear form
   BlockNonlinearForm *Hform;

   // Pressure mass matrix for the preconditioner
   SparseMatrix *pressure_mass;

   // Newton solver for the hyperelastic operator
   NewtonSolver newton_solver;
   SystemResidualMonitor newton_monitor;

   // Solver for the Jacobian solve in the Newton method
   Solver *j_solver;
   GeneralResidualMonitor j_monitor;

   // Preconditioner for the Jacobian
   Solver *j_prec;

   // Shear modulus coefficient
   Coefficient &mu;

   //
   Tau *tau, *delta;
   GridFunction *adv_gf;
   VectorCoefficient *adv;
   

   // Block offsets for variable access
   Array<int> &block_trueOffsets;

public:
   StabInNavStoOperator(Array<FiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets, real_t rel_tol, real_t abs_tol,
                  int iter, Coefficient &mu);

   // Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &xp) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   // Driver for the newton solver
   void Solve(Vector &xp) const;

   virtual ~StabInNavStoOperator();
};


} // namespace mfem

#endif
