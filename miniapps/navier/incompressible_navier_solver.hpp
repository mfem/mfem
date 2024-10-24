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

#ifndef MFEM_INCOMP_NAVIER_SOLVER_HPP
#define MFEM_INCOMP_NAVIER_SOLVER_HPP

#define INCOMP_NAVIER_VERSION 0.1

#include "mfem.hpp"

namespace mfem
{
namespace incompressible_navier
{
using VecFuncT = void(const Vector &x, real_t t, Vector &u);
using ScalarFuncT = real_t(const Vector &x, real_t t);

class UnitVectorGridFunctionCoeff : public VectorCoefficient
{
public:
   UnitVectorGridFunctionCoeff( int dim)
   : VectorCoefficient(dim*dim)
   { }

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t coeffVal = gridfunc_->GetValue(T, ip);

      V.SetSize(vdim); V = 0.0;                           // FIXME
      V[0] = coeffVal;
      V[3] = coeffVal;
   }

   void SetGridFunction( GridFunction * gridfunc )
   {
      gridfunc_ = gridfunc;
   }

   GridFunction *gridfunc_ = nullptr;
};

class PrevVelVectorGridFunctionCoeff : public VectorCoefficient
{
public:
   PrevVelVectorGridFunctionCoeff( int dim)
   : VectorCoefficient(dim)
   { }

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      V.SetSize(vdim);
      gridFuncCoeff->Eval(V, T, ip);

      V *= 1.0/dt_;
   }

   void SetGridFunction( GridFunction * gridfunc, real_t dt )
   {
      gridfunc_ = gridfunc;
      dt_ = dt;
      delete gridFuncCoeff;
      gridFuncCoeff = new VectorGridFunctionCoefficient( gridfunc );
   }

   GridFunction *gridfunc_ = nullptr;
   VectorGridFunctionCoefficient *gridFuncCoeff = nullptr;
   real_t dt_;
};

class NonLinTermVectorGridFunctionCoeff : public VectorCoefficient
{
public:
   NonLinTermVectorGridFunctionCoeff( int dim)
   : VectorCoefficient(dim)
   { }

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector val(vdim);
      Vector resultVal(vdim);
      DenseMatrix vecGrad;
      V.SetSize(vdim);
      gridFuncCoeff->Eval(val, T, ip);

      gridfunc_->GetVectorGradient(T, vecGrad);

      vecGrad.MultTranspose( val, V );

      V *= -1.0;
   }

   void SetGridFunction( ParGridFunction * gridfunc )
   {
      delete gridFuncCoeff;
      gridfunc_ = gridfunc;
      gridFuncCoeff = new VectorGridFunctionCoefficient( gridfunc );
   }

   VectorGridFunctionCoefficient *gridFuncCoeff = nullptr;
   ParGridFunction *gridfunc_ = nullptr;
};

/// Container for a Dirichlet boundary condition of the velocity field.
class VelDirichletBC_T
{
public:
   VelDirichletBC_T(Array<int> attr, VectorCoefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   VelDirichletBC_T(VelDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~VelDirichletBC_T() { delete coeff; }

   Array<int> attr;
   VectorCoefficient *coeff;
};

/// Transient incompressible Navier Stokes solver in a split scheme formulation.
/**
 * This implementation of a transient incompressible Navier Stokes solver uses
 * the non-dimensionalized formulation. The coupled momentum and
 * incompressibility equations are decoupled using the split scheme described in
 * [1]. This leads to three solving steps.
 *
 */
class IncompressibleNavierSolver
{
public:
   /// Initialize data structures, set FE space order and kinematic viscosity.
   /**
    * The ParMesh @a mesh can be a linear or curved parallel mesh. The @a order
    * of the finite element spaces is
    */
   IncompressibleNavierSolver(ParMesh *mesh, int velorder, int porder, int tOrder, real_t kin_vis);

   /// Initialize forms, solvers and preconditioners.
   void Setup(real_t dt);

   /// Compute solution at the next time step t+dt.
   /**
    * This method can 
    */
   void Step(real_t &time, real_t dt, int cur_step, bool provisional = false);

   /// Return a pointer to the provisional velocity ParGridFunction.
   ParGridFunction *GetProvisionalVelocity() { return velGF[1]; }

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelocity() { return velGF[0]; }

   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPressure() { return pGF[0]; }

   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPsi() { return &psiGF ; }


   /// Add a Dirichlet boundary condition to the velocity field.
   void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);

   void AddVelDirichletBC(VecFuncT *f, Array<int> &attr);

   /// Add a Dirichlet boundary condition to the pressure field.
   // void AddPresDirichletBC(Coefficient *coeff, Array<int> &attr);

   // void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr);

   /// Enable partial assembly for every operator.
   void EnablePA(bool pa) { partial_assembly = pa; }

   /// Enable numerical integration rules. This means collocated quadrature at
   /// the nodal points.
   void EnableNI(bool ni) { numerical_integ = ni; }

   /// Print timing summary of the solving routine.
   /**
    * The summary shows the timing in seconds in the first row of
    *
    */
   void PrintTimingData();

   ~IncompressibleNavierSolver();

   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(real_t dt);


   /// Compute CFL
   real_t ComputeCFL(ParGridFunction &u, real_t dt);

protected:

   /// Eliminate essential BCs in an Operator and apply to RHS.
   void EliminateRHS(Operator &A,
                     ConstrainedOperator &constrainedA,
                     const Array<int> &ess_tdof_list,
                     Vector &x,
                     Vector &b,
                     Vector &X,
                     Vector &B,
                     int copy_interior = 0);

   /// Enable/disable debug output.
   bool debug = false;

   /// Enable/disable verbose output.
   bool verbose = true;

   /// Enable/disable partial assembly of forms.
   bool partial_assembly = false;

   /// Enable/disable numerical integration rules of forms.
   bool numerical_integ = false;

   /// The parallel mesh.
   ParMesh *pmesh = nullptr;

   /// The order of the velocity and pressure space.
   int velorder;
   int porder;
   int torder;

   /// Kinematic viscosity (dimensionless).
   real_t kin_vis;
   Coefficient * kinvisCoeff = nullptr;

   Coefficient *dtCoeff = nullptr;

   IntegrationRules gll_rules;

   /// Velocity $H^1$ finite element collection.
   FiniteElementCollection *vfec = nullptr;

   /// Psi $H^1$ finite element collection.
   FiniteElementCollection *psifec = nullptr;

   /// Pressure $H^1$ finite element collection.
   FiniteElementCollection *pfec = nullptr;

   /// Velocity $(H^1)^d$ finite element space.
   ParFiniteElementSpace *vfes = nullptr;

   /// Psi $(H^1)^d$ finite element space.
   ParFiniteElementSpace *psifes = nullptr;

   /// Pressure $H^1$ finite element space.
   ParFiniteElementSpace *pfes = nullptr;

   ParBilinearForm *velBForm = nullptr;
   ParBilinearForm *psiBForm = nullptr;
   ParBilinearForm *pBForm = nullptr;

   ParLinearForm *velLForm = nullptr;
   ParLinearForm *psiLForm = nullptr;
   ParLinearForm *pLForm = nullptr;

   std::vector<ParGridFunction*> velGF;
   std::vector<ParGridFunction*> pGF;
   ParGridFunction psiGF;

   ParGridFunction DvGF, divVelGF, pRHS;
   VectorGridFunctionCoefficient * DvelCoeff = nullptr;
   DivergenceGridFunctionCoefficient * divVelCoeff = nullptr;
   GridFunctionCoefficient * pRHSCoeff = nullptr;
   UnitVectorGridFunctionCoeff * pUnitVectorCoeff = nullptr;
   NonLinTermVectorGridFunctionCoeff * nonlinTermCoeff = nullptr;
   PrevVelVectorGridFunctionCoeff * prevVelLoadCoeff = nullptr;
   

   // VectorGridFunctionCoefficient *FText_gfcoeff = nullptr;

   // ParLinearForm *FText_bdr_form = nullptr;

   // ParLinearForm *f_form = nullptr;

   // ParLinearForm *g_bdr_form = nullptr;

   // /// Linear form to compute the mass matrix in various subroutines.
   // ParLinearForm *mass_lf = nullptr;
   // ConstantCoefficient onecoeff;
   // real_t volume = 0.0;

   // ConstantCoefficient nlcoeff;
   // ConstantCoefficient Sp_coeff;
   // ConstantCoefficient H_lincoeff;
   // ConstantCoefficient H_bdfcoeff;

   OperatorHandle vOp;
   OperatorHandle psiOp;
   OperatorHandle pOp;

   Solver *velInvPC = nullptr;
   CGSolver *velInv = nullptr;

   Solver *psiInvPC = nullptr;
   OrthoSolver *SpInvOrthoPC = nullptr;
   CGSolver *psiInv = nullptr;

   Solver *pInvPC = nullptr;
   CGSolver *pInv = nullptr;

   // Vector fn, un, un_next, unm1, unm2, Nun, Nunm1, Nunm2, Fext, FText, Lext,
   //        resu;
   // Vector tmp1;

   Vector velLF, psiLF, pLF;

   // ParGridFunction un_gf, un_next_gf, curlu_gf, curlcurlu_gf, Lext_gf, FText_gf,
   //                 resu_gf;

   // ParGridFunction pn_gf, resp_gf;

   // All essential attributes.
   Array<int> vel_ess_attr;
   Array<int> pres_ess_attr;

   // All essential true dofs.
   Array<int> vel_ess_tdof;
   Array<int> pres_ess_tdof;

   // Bookkeeping for velocity dirichlet bcs.
   std::vector<VelDirichletBC_T> vel_dbcs;

   // Bookkeeping for pressure dirichlet bcs.
   //std::vector<PresDirichletBC_T> pres_dbcs;

   // Print levels.
   int pl_psolve = 0;
   int pl_psisolve = 0;
   int pl_velsolve = 0;
   int pl_amg = 0;

   #if defined(MFEM_USE_DOUBLE)
   real_t rtol_psolve = 1e-10;
   real_t rtol_psisolve = 1e-10;
   real_t rtol_velsolve = 1e-12;
#elif defined(MFEM_USE_SINGLE)
   real_t rtol_psolve = 1e-9;
   real_t rtol_psisolve = 1e-5;
   real_t rtol_velsolve = 1e-7;
#else
#error "Only single and double precision are supported!"
   real_t rtol_psolve = 1e-12;
   real_t rtol_psisolve = 1e-6;
   real_t rtol_velsolve = 1e-8;
#endif

   // Iteration counts.
   int iter = 1, iter_vsolve = 0, iter_psolve = 0, iter_psisolve = 0;

   // Residuals.
   real_t res_vsolve = 0.0, res_psolve = 0.0, res_psisolve = 0.0;

};

} // namespace incompressible_navier

} // namespace mfem

#endif
