// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ADV_DIFF_SOLVER
#define MFEM_ADV_DIFF_SOLVER

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace thermal
{

class AdvectionTDO : public TimeDependentOperator
{
public:
   AdvectionTDO(ParFiniteElementSpace &H1_FES, VectorCoefficient &velCoef);
   ~AdvectionTDO();

   /** @brief Perform the action of the operator: @a q = f(@a y, t), where
       q solves the algebraic equation F(@a y, q, t) = G(@a y, t) and t is the
       current time. */
   virtual void Mult(const Vector &y, Vector &q) const;

private:

   void initMult() const;

   ParFiniteElementSpace & H1_FESpace_;
   VectorCoefficient & velCoef_;

   Array<int> ess_bdr_tdofs_;

   mutable ParBilinearForm m1_;
   ParBilinearForm adv1_;

   mutable HypreParMatrix   M1_;
   mutable HyprePCG       * M1Inv_;
   mutable HypreDiagScale * M1Diag_;
   mutable ParGridFunction dydt_gf_;
   mutable Vector SOL_;
   mutable Vector RHS_;
   mutable Vector rhs_;
};

/**
   The thermal diffusion equation can be written:

      dcT/dt = Div (sigma Grad T) + Q_s

   where

     T     is the temperature.
     Div   is the divergence operator,
     grad  is the gradient operator,
     sigma is the thermal conductivity,
     c     is the heat capacity,
     Q_s   is the heat source

   Class ThermalDiffusionOperator represents the right-hand side of the above
   system of ODEs.

     f(t, T) = -M_0(c)^{-1}(S_0(sigma)T - M_0 Q_s)

   where

     M_0(c)     is an H_1 mass matrix
     S_0(sigma) is the diffusion operator

   The implicit solve method will solve

     (M_0(c)+dt S_0(sigma))k = -S_0(sigma)T + M_0 Q_s
*/
class AdvectionDiffusionTDO : public TimeDependentOperator
{
public:
   AdvectionDiffusionTDO(ODESolver & exp_solver, double dt_exp,
                         ParFiniteElementSpace &H1_FES,
                         Coefficient & dTdtBdr,
                         Array<int> & bdr_attr,
                         Coefficient & c, bool td_c,
                         Coefficient & k, bool td_k,
                         Coefficient & Q, bool td_Q,
                         VectorCoefficient & v, bool td_v);
   AdvectionDiffusionTDO(ODESolver & exp_solver, double dt_exp,
                         ParFiniteElementSpace &H1_FES,
                         Coefficient & dTdtBdr,
                         Array<int> & bdr_attr,
                         Coefficient & c, bool td_c,
                         MatrixCoefficient & K, bool td_k,
                         Coefficient & Q, bool td_Q,
                         VectorCoefficient & v, bool td_v);

   void SetTime(const double time);
   /*
    void SetHeatSource(Coefficient & Q, bool time_dep = false);

    void SetConductivityCoefficient(Coefficient & k,
             bool time_dep = false);

    void SetConductivityCoefficient(MatrixCoefficient & K,
             bool time_dep = false);

    void SetSpecificHeatCoefficient(
             bool time_dep = false);
   */
   /** @brief Perform the action of the operator: @a q = f(@a y, t), where
       q solves the algebraic equation F(@a y, q, t) = G(@a y, t) and t is the
       current time. */
   virtual void Mult(const Vector &y, Vector &q) const;

   /** @brief Solve the equation: @a q = f(@a y + @a dt @a q, t), for the
       unknown @a q at the current time t.

       For general F and G, the equation for @a q becomes:
       F(@a y + @a dt @a q, @a q, t) = G(@a y + @a dt @a q, t).

       The input vector @a y corresponds to time index (or cycle) n, while the
       currently set time, #t, and the result vector @a q correspond to time
       index n+1. The time step @a dt corresponds to the time interval between
       cycles n and n+1.

       This method allows for the abstract implementation of some time
       integration methods, including diagonal implicit Runge-Kutta (DIRK)
       methods and the backward Euler method in particular.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const double dt, const Vector &y, Vector &q);

   virtual ~AdvectionDiffusionTDO();

private:

   void init();

   void initMult() const;
   void initA(double dt);
   void initImplicitSolve();

   int myid_;
   bool init_;
   // bool initA_;
   // bool initAInv_;
   bool newTime_;

   mutable int multCount_;
   int solveCount_;

   ODESolver & exp_solver_;
   AdvectionTDO  exp_oper_;
   mutable Vector T_exp_;
   mutable Vector dTdt_exp_;
   double t0_exp_;
   double t1_exp_;
   double dt_exp_;

   ParFiniteElementSpace * H1_FESpace_;

   ParBilinearForm * mC_;
   ParBilinearForm * sK_;
   ParBilinearForm * a_;

   ParGridFunction * dTdt_gf_;
   ParLinearForm * Qs_;

   mutable HypreParMatrix   MC_;
   mutable HyprePCG       * MCInv_;
   mutable HypreDiagScale * MCDiag_;

   HypreParMatrix    A_;
   HyprePCG        * AInv_;
   HypreBoomerAMG  * APrecond_;

   // HypreParVector * T_;
   mutable Vector dTdt_;
   mutable Vector RHS_;
   Vector * rhs_;

   Array<int> * bdr_attr_;
   Array<int>   ess_bdr_tdofs_;

   Coefficient       * dTdtBdrCoef_;

   bool tdQ_;
   bool tdC_;
   bool tdK_;
   /*
   bool ownsQ_;
   bool ownsC_;
   bool ownsK_;
   */
   Coefficient       * QCoef_;
   Coefficient       * CCoef_;
   Coefficient       * kCoef_;
   MatrixCoefficient * KCoef_;
   // Coefficient       * CInvCoef_;
   // Coefficient       * kInvCoef_;
   // MatrixCoefficient * KInvCoef_;
   Coefficient       * dtkCoef_;
   MatrixCoefficient * dtKCoef_;
};

} // namespace thermal

class InverseCoefficient : public Coefficient
{
public:
   InverseCoefficient(Coefficient & c) : c_(&c) {}

   void SetTime(double t) { time = t; c_->SetTime(t); }

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   { return 1.0 / c_->Eval(T, ip); }

private:
   Coefficient * c_;
};

class MatrixInverseCoefficient :public MatrixCoefficient
{
public:
   MatrixInverseCoefficient(MatrixCoefficient & M)
      : MatrixCoefficient(M.GetWidth()), M_(&M) {}

   void SetTime(double t) { time = t; M_->SetTime(t); }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   MatrixCoefficient * M_;
};

class ScaledCoefficient : public Coefficient
{
public:
   ScaledCoefficient(double a, Coefficient & c) : a_(a), c_(&c) {}

   void SetTime(double t) { time = t; c_->SetTime(t); }

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   { return a_ * c_->Eval(T, ip); }

private:
   double a_;
   Coefficient * c_;
};

class ScaledMatrixCoefficient :public MatrixCoefficient
{
public:
   ScaledMatrixCoefficient(double a, MatrixCoefficient & M)
      : MatrixCoefficient(M.GetWidth()), a_(a), M_(&M) {}

   void SetTime(double t) { time = t; M_->SetTime(t); }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   double a_;
   MatrixCoefficient * M_;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_ADV_DIFF_SOLVER
