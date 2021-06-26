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

#ifndef MFEM_FOURIER_NL_SOLVER
#define MFEM_FOURIER_NL_SOLVER

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{
/*
class UnitVectorField : public VectorCoefficient
{
private:
 int prob_;
 int unit_vec_type_;
 double a_;
 double b_;

public:
 UnitVectorField(int prob, int unit_vec_type, double a = 0.4, double b = 0.8)
   : VectorCoefficient(2), prob_(prob), unit_vec_type_(unit_vec_type),
     a_(a), b_(b) {}

 void Eval(Vector &V, ElementTransformation &T,
           const IntegrationPoint &ip);
};
*/
class ChiParaCoef : public MatrixCoefficient
{
private:
   MatrixCoefficient * bbT_;
   GridFunctionCoefficient * T_;
   double chi_para_;
   bool nonlin_;

public:
   ChiParaCoef(MatrixCoefficient &bbT, GridFunctionCoefficient &T,
               double chi_para, bool nonlin = false)
      : MatrixCoefficient(2), bbT_(&bbT), T_(&T),
        chi_para_(chi_para), nonlin_(nonlin)
   {}

   void SetTemp(GridFunction & T) { T_->SetGridFunction(&T); }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ChiPerpCoef : public MatrixCoefficient
{
private:
   MatrixCoefficient * bbT_;
   GridFunctionCoefficient * T_;
   double chi_perp_;
   bool nonlin_;

public:
   ChiPerpCoef(MatrixCoefficient &bbT, GridFunctionCoefficient &T,
               double chi_perp, bool nonlin = false)
      : MatrixCoefficient(2), bbT_(&bbT), T_(&T),
        chi_perp_(chi_perp), nonlin_(nonlin)
   {}

   void SetTemp(GridFunction & T) { T_->SetGridFunction(&T); }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ChiCoef : public MatrixSumCoefficient
{
private:
   ChiPerpCoef * chiPerpCoef_;
   ChiParaCoef * chiParaCoef_;

public:
   ChiCoef(ChiPerpCoef & chiPerp, ChiParaCoef & chiPara)
      : MatrixSumCoefficient(chiPerp, chiPara),
        chiPerpCoef_(&chiPerp), chiParaCoef_(&chiPara) {}

   void SetTemp(GridFunction & T)
   { chiPerpCoef_->SetTemp(T); chiParaCoef_->SetTemp(T); }
};

class dChiCoef : public MatrixCoefficient
{
private:
   MatrixCoefficient * bbT_;
   GridFunctionCoefficient * T_;
   double chi_perp_;
   double chi_para_;

public:
   dChiCoef(MatrixCoefficient &bbT, GridFunctionCoefficient &T,
            double chi_perp, double chi_para)
      : MatrixCoefficient(2), bbT_(&bbT), T_(&T),
        chi_perp_(chi_perp), chi_para_(chi_para)
   {}

   void SetTemp(GridFunction & T) { T_->SetGridFunction(&T); }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

namespace thermal
{

class ImplicitDiffOp : public Operator
{
public:
   ImplicitDiffOp(ParFiniteElementSpace & H1_FESpace,
                  Coefficient & dTdtBdr, bool tdBdr,
                  Array<int> & bdr_attr,
                  Coefficient & heatCap, bool tdCp,
                  ChiCoef & chi, bool tdChi,
                  dChiCoef & dchi, bool tdDChi,
                  Coefficient & heatSource, bool tdQ,
                  bool nonlinear = false);
   ~ImplicitDiffOp();

   void SetState(ParGridFunction & T, double t, double dt);

   void Mult(const Vector &x, Vector &y) const;

   Operator & GetGradient(const Vector &x) const;

   Solver & GetGradientSolver() const;

   const Vector & GetRHS() const { return RHS_; }

private:

   bool first_;
   bool tdBdr_;
   bool tdCp_;
   bool tdChi_;
   bool tdDChi_;
   bool tdQ_;
   bool nonLinear_;
   bool newTime_;
   bool newTimeStep_;

   double t_;
   double dt_;

   Array<int> & ess_bdr_attr_;
   Array<int>   ess_bdr_tdofs_;

   Coefficient * bdrCoef_;
   Coefficient * cpCoef_;
   ChiCoef     * chiCoef_;
   dChiCoef    * dChiCoef_;
   Coefficient * QCoef_;
   ScalarMatrixProductCoefficient dtChiCoef_;

   mutable ParGridFunction T0_;
   mutable ParGridFunction T1_;
   mutable ParGridFunction dT_;

   mutable GradientGridFunctionCoefficient gradTCoef_;
   ScalarVectorProductCoefficient dtGradTCoef_;
   MatVecCoefficient dtdChiGradTCoef_;

   ParBilinearForm m0cp_;
   mutable ParBilinearForm s0chi_;
   mutable ParBilinearForm a0_;

   mutable HypreParMatrix A_;
   mutable ParGridFunction dTdt_;
   mutable ParLinearForm Q_;
   mutable ParLinearForm Qs_;
   mutable ParLinearForm rhs_;

   mutable  Vector SOL_;
   mutable  Vector RHS_;
   // Vector RHS0_; // Dummy RHS vector which hase length zero

   mutable Solver         * AInv_;
   mutable HypreBoomerAMG * APrecond_;
};

/**
   The thermal diffusion equation can be written:

      dcT/dt = Div (chi Grad T) + Q_s

   where

     T     is the temperature.
     Div   is the divergence operator,
     grad  is the gradient operator,
     chi   is the thermal conductivity tensor,
     c     is the heat capacity,
     Q_s   is the heat source

   Class ThermalDiffusionTDO represents the right-hand side of the above
   system of ODEs.

     f(t, T) = -M_0(c)^{-1}(S_0(chi)T - M_0 Q_s)

   where

     M_0(c)     is an H_1 mass matrix
     S_0(chi) is the diffusion operator

   The implicit solve method will solve

     (M_0(c)+dt S_0(chi))k = -S_0(chi)T + M_0 Q_s
*/
class ThermalDiffusionTDO : public TimeDependentOperator
{
public:
   ThermalDiffusionTDO(ParFiniteElementSpace &H1_FES,
                       Coefficient & dTdtBdr,
                       Array<int> & bdr_attr,
                       double chi_perp,
                       double chi_para,
                       int prob,
                       int coef_type,
                       VectorCoefficient & UnitB,
                       Coefficient & c, bool td_c,
                       Coefficient & Q, bool td_Q);

   void SetTime(const double time);

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

   virtual ~ThermalDiffusionTDO();

private:

   void init();

   bool init_;
   bool newTime_;
   bool nonLinear_;
   bool testGradient_;

   mutable int multCount_;
   int solveCount_;

   mutable ParGridFunction T_;

   GridFunctionCoefficient TCoef_;
   VectorCoefficient * unitBCoef_;
   // IdentityMatrixCoefficient ICoef_;
   OuterProductCoefficient bbTCoef_;
   ChiPerpCoef chiPerpCoef_;
   ChiParaCoef chiParaCoef_;
   ChiCoef     chiCoef_;
   dChiCoef    dChiCoef_;

   ImplicitDiffOp impOp_;
   NewtonSolver   newton_;
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

#endif // MFEM_FOURIER_NL_SOLVER
