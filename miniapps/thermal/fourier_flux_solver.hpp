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

#ifndef MFEM_FOURIER_FLUX_SOLVER
#define MFEM_FOURIER_FLUX_SOLVER

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace thermal
{

/**
   The thermal diffusion equation can be written:

      dcT/dt = Div (chi Grad T) + Q_s

   We would like to rewrite this using the flux formulation which solves for
   the heat flux vector q.  The primary equations are:

      q = chi Grad T
      u = c T
      du/dt + Div q = Q_s

   Which lead to:

      dq/dt = chi Grad (c^{-1} Div q) - Grad(c^{-1} Q_s)

   where

      T     is the temperature.
      q     is the heat flux
      u     is the thermal energy density
      Div   is the divergence operator,
      Grad  is the gradient operator,
      chi   is the thermal conductivity,
      c     is the heat capacity,
      Q_s   is the heat source

   Class ThermalDiffusionFluxOperator represents the right-hand side of
   the above system of ODEs.

      f(t, T) = -M_0(c)^{-1}(S_0(chi)T - M_0 Q_s)

   where

      M_0(c)     is an H_1 mass matrix
      S_0(sigma) is the diffusion operator

   The implicit solve method will solve

      (M_0(c)+dt S_0(sigma))k = -S_0(sigma)T + M_0 Q_s
*/
class ThermalDiffusionFluxOperator : public TimeDependentOperator
{
public:
   ThermalDiffusionFluxOperator(ParMesh & pmesh,
                                ParFiniteElementSpace &HDiv_FES,
                                ParFiniteElementSpace &L2_FES,
                                VectorCoefficient & dqdtBdr,
                                Array<int> & bdr_attr,
                                Coefficient & c, bool td_c,
                                Coefficient & k, bool td_k,
                                Coefficient & Q, bool td_Q);
   ThermalDiffusionFluxOperator(ParMesh & pmesh,
                                ParFiniteElementSpace &HDiv_FES,
                                ParFiniteElementSpace &L2_FES,
                                VectorCoefficient & dqdtBdr,
                                Array<int> & bdr_attr,
                                Coefficient & c, bool td_c,
                                MatrixCoefficient & K, bool td_k,
                                Coefficient & Q, bool td_Q);

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

   virtual ~ThermalDiffusionFluxOperator();

private:

   void init();

   void initMult() const;
   void initA(double dt);
   void initImplicitSolve();

   bool init_;
   // bool initA_;
   // bool initAInv_;
   bool newTime_;

   int dim_;
   mutable int multCount_;
   int solveCount_;

   ParFiniteElementSpace * HDiv_FESpace_;
   ParFiniteElementSpace * L2_FESpace_;

   ParBilinearForm * mK_;
   ParBilinearForm * sC_;
   ParMixedBilinearForm * dC_;
   ParBilinearForm * a_;

   ParDiscreteLinearOperator * Div_;

   ParGridFunction * dqdt_gf_;
   ParGridFunction * Qs_;

   mutable HypreParMatrix   MK_;
   mutable HyprePCG       * MKInv_;
   mutable HypreDiagScale * MKDiag_;

   HypreParMatrix    A_;
   HyprePCG        * AInv_;
   HypreSolver     * APrecond_;

   // HypreParVector * T_;
   mutable ParGridFunction q_;
   mutable ParGridFunction u_;
   mutable ParGridFunction dqdt_;
   mutable ParGridFunction dudt_;
   mutable Vector X_;
   mutable Vector RHS_;
   mutable Vector rhs_;
   mutable Vector dQs_;
   mutable Vector tmp_;

   Array<int> * bdr_attr_;
   Array<int>   ess_bdr_tdofs_;

   VectorCoefficient * dqdtBdrCoef_;

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
   Coefficient       * CInvCoef_;
   Coefficient       * kInvCoef_;
   MatrixCoefficient * KInvCoef_;
   Coefficient       * dtCInvCoef_;
   // MatrixCoefficient * dtKCoef_;
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

#endif // MFEM_FOURIER_FLUX_SOLVER
