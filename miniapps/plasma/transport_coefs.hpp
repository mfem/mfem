// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TRANSPORT_COEFS
#define MFEM_TRANSPORT_COEFS

#include "../common/fem_extras.hpp"
#include "plasma.hpp"
#include "pgridfuncarray.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

namespace transport
{

enum FieldType {INVALID_FIELD_TYPE   = -1,
                NEUTRAL_DENSITY      = 0,
                ION_DENSITY          = 1,
                ION_PARA_VELOCITY    = 2,
                ION_TEMPERATURE      = 3,
                ELECTRON_TEMPERATURE = 4
               };

std::string FieldSymbol(FieldType t);

class StateVariableFunc
{
public:

   virtual bool NonTrivialValue(FieldType deriv) const = 0;

   void SetDerivType(FieldType deriv) { derivType_ = deriv; }
   FieldType GetDerivType() const { return derivType_; }

protected:
   StateVariableFunc(FieldType deriv = INVALID_FIELD_TYPE)
      : derivType_(deriv) {}

   FieldType derivType_;
};


class StateVariableCoef : public StateVariableFunc, public Coefficient
{
public:

   virtual StateVariableCoef * Clone() const = 0;

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      switch (derivType_)
      {
         case INVALID_FIELD_TYPE:
            return Eval_Func(T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(T, ip);
         case ION_DENSITY:
            return Eval_dNi(T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(T, ip);
         default:
            return 0.0;
      }
   }

   /// Implementation of the coefficient function
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip) { return 0.0; }

   /// Implementation of the genderalized derivative of the function
   virtual double Eval_dFunc(FieldType deriv,
                             ElementTransformation &T,
                             const IntegrationPoint &ip) { return 0.0; }

   /// The following can be overridden for efficiency when appropriate
   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(NEUTRAL_DENSITY, T, ip);
   }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ION_DENSITY, T, ip);
   }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ION_PARA_VELOCITY, T, ip);
   }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ION_TEMPERATURE, T, ip);
   }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ELECTRON_TEMPERATURE, T, ip);
   }

protected:
   StateVariableCoef(FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv) {}
};

class StateVariableVecCoef : public StateVariableFunc,
   public VectorCoefficient
{
public:
   virtual StateVariableVecCoef * Clone() const = 0;

   virtual void Eval(Vector &V,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      V.SetSize(vdim);

      switch (derivType_)
      {
         case INVALID_FIELD_TYPE:
            return Eval_Func(V, T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(V, T, ip);
         case ION_DENSITY:
            return Eval_dNi(V, T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(V, T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(V, T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(V, T, ip);
         default:
            V = 0.0;
            return;
      }
   }

   virtual void Eval_Func(Vector &V,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dNn(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dNi(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dVi(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dTi(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dTe(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

protected:
   StateVariableVecCoef(int dim, FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv), VectorCoefficient(dim) {}
};

class StateVariableMatCoef : public StateVariableFunc,
   public MatrixCoefficient
{
public:

   virtual StateVariableMatCoef * Clone() const = 0;

   virtual void Eval(DenseMatrix &M,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      M.SetSize(height, width);

      switch (derivType_)
      {
         case INVALID_FIELD_TYPE:
            return Eval_Func(M, T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(M, T, ip);
         case ION_DENSITY:
            return Eval_dNi(M, T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(M, T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(M, T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(M, T, ip);
         default:
            M = 0.0;
            return;
      }
   }

   virtual void Eval_Func(DenseMatrix &M,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dNn(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dNi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dVi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dTi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dTe(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

protected:
   StateVariableMatCoef(int dim, FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv), MatrixCoefficient(dim) {}

   StateVariableMatCoef(int h, int w, FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv), MatrixCoefficient(h, w) {}
};

class StateVariableStandardCoef : public StateVariableCoef
{
private:
   Coefficient & c_;

public:
   StateVariableStandardCoef(Coefficient & c)
      : c_(c)
   {}

   StateVariableStandardCoef(const StateVariableStandardCoef &other)
      : c_(other.c_)
   {}

   virtual StateVariableStandardCoef * Clone() const
   {
      return new StateVariableStandardCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   {
      return c_.Eval(T, ip);
   }
};

class StateVariableStandardVecCoef : public StateVariableVecCoef
{
private:
   VectorCoefficient & V_;

public:
   StateVariableStandardVecCoef(VectorCoefficient & V)
      : StateVariableVecCoef(V.GetVDim()), V_(V)
   {}

   StateVariableStandardVecCoef(const StateVariableStandardVecCoef &other)
      : StateVariableVecCoef(other.vdim), V_(other.V_)
   {}

   virtual StateVariableStandardVecCoef * Clone() const
   {
      return new StateVariableStandardVecCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual void Eval_Func(Vector & V, ElementTransformation &T,
                          const IntegrationPoint &ip)
   {
      V_.Eval(V, T, ip);
   }
};

class StateVariableStandardMatCoef : public StateVariableMatCoef
{
private:
   MatrixCoefficient & M_;

public:
   StateVariableStandardMatCoef(MatrixCoefficient & M)
      : StateVariableMatCoef(M.GetHeight(), M.GetWidth()), M_(M)
   {}

   StateVariableStandardMatCoef(const StateVariableStandardMatCoef &other)
      : StateVariableMatCoef(other.height, other.width), M_(other.M_)
   {}

   virtual StateVariableStandardMatCoef * Clone() const
   {
      return new StateVariableStandardMatCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual void Eval_Func(DenseMatrix & M, ElementTransformation &T,
                          const IntegrationPoint &ip)
   {
      M_.Eval(M, T, ip);
   }
};

class StateVariableConstantCoef : public StateVariableCoef
{
private:
   ConstantCoefficient c_;

public:
   StateVariableConstantCoef(double c)
      : c_(c)
   {}

   StateVariableConstantCoef(const StateVariableConstantCoef &other)
      : c_(other.c_)
   {}

   virtual StateVariableConstantCoef * Clone() const
   {
      return new StateVariableConstantCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   {
      return c_.Eval(T, ip);
   }
};

class StateVariableGridFunctionCoef : public StateVariableCoef
{
private:
   GridFunctionCoefficient gfc_;
   FieldType fieldType_;

public:
   StateVariableGridFunctionCoef(FieldType field)
      : fieldType_(field)
   {}

   StateVariableGridFunctionCoef(GridFunction *gf, FieldType field)
      : gfc_(gf), fieldType_(field)
   {}

   StateVariableGridFunctionCoef(const StateVariableGridFunctionCoef &other)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      gfc_       = other.gfc_;
   }

   virtual StateVariableGridFunctionCoef * Clone() const
   {
      return new StateVariableGridFunctionCoef(*this);
   }

   void SetGridFunction(GridFunction *gf) { gfc_.SetGridFunction(gf); }
   const GridFunction * GetGridFunction() const
   { return gfc_.GetGridFunction(); }

   void SetFieldType(FieldType field) { fieldType_ = field; }
   FieldType GetFieldType() const { return fieldType_; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == fieldType_);
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   {
      return gfc_.Eval(T, ip);
   }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == NEUTRAL_DENSITY) ? 1.0 : 0.0; }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ION_DENSITY) ? 1.0 : 0.0; }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ION_PARA_VELOCITY) ? 1.0 : 0.0; }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ION_TEMPERATURE) ? 1.0 : 0.0; }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ELECTRON_TEMPERATURE) ? 1.0 : 0.0; }
};

class StateVariableSumCoef : public StateVariableCoef
{
private:
   StateVariableCoef *a;
   StateVariableCoef *b;

   double alpha;
   double beta;

public:
   // Result is _alpha * A + _beta * B
   StateVariableSumCoef(StateVariableCoef &A, StateVariableCoef &B,
                        double _alpha = 1.0, double _beta = 1.0)
      : a(A.Clone()), b(B.Clone()), alpha(_alpha), beta(_beta) {}

   ~StateVariableSumCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableSumCoef * Clone() const
   {
      return new StateVariableSumCoef(*a, *b, alpha, beta);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableCoef &B) { b = &B; }
   StateVariableCoef * GetBCoef() const { return b; }

   void SetAlpha(double _alpha) { alpha = _alpha; }
   double GetAlpha() const { return alpha; }

   void SetBeta(double _beta) { beta = _beta; }
   double GetBeta() const { return beta; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) || b->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   { return alpha * a->Eval_Func(T, ip) + beta * b->Eval_Func(T, ip); }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dNn(T, ip) + beta * b->Eval_dNn(T, ip); }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dNi(T, ip) + beta * b->Eval_dNi(T, ip); }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dVi(T, ip) + beta * b->Eval_dVi(T, ip); }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dTi(T, ip) + beta * b->Eval_dTi(T, ip); }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dTe(T, ip) + beta * b->Eval_dTe(T, ip); }
};

class StateVariableProductCoef : public StateVariableCoef
{
private:
   StateVariableCoef *a;
   StateVariableCoef *b;

public:
   // Result is A * B
   StateVariableProductCoef(StateVariableCoef &A, StateVariableCoef &B)
      : a(A.Clone()), b(B.Clone()) {}

   ~StateVariableProductCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableProductCoef * Clone() const
   {
      return new StateVariableProductCoef(*a, *b);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableCoef &B) { b = &B; }
   StateVariableCoef * GetBCoef() const { return b; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) || b->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   { return a->Eval_Func(T, ip) * b->Eval_Func(T, ip); }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dNn(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dNn(T, ip);
   }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dNi(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dNi(T, ip);
   }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dVi(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dVi(T, ip);
   }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dTi(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dTi(T, ip);
   }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dTe(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dTe(T, ip);
   }
};

class StateVariablePowerCoef : public StateVariableCoef
{
private:
   StateVariableCoef *a;
   int p;

public:
   // Result is A^p
   StateVariablePowerCoef(StateVariableCoef &A, int p)
      : a(A.Clone()), p(p) {}

   ~StateVariablePowerCoef()
   {
      if (a != NULL) { delete a; }
   }

   virtual StateVariablePowerCoef * Clone() const
   {
      return new StateVariablePowerCoef(*a, p);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetPower(int new_p) { p = new_p; }
   int GetPower() const { return p; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   { return pow(a->Eval_Func(T, ip), p); }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dNn(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dNn(T, ip);
   }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dNi(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dNi(T, ip);
   }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dVi(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dVi(T, ip);
   }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dTi(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dTi(T, ip);
   }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dTe(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dTe(T, ip);
   }
};

class StateVariableScalarVectorProductCoef : public StateVariableVecCoef
{
private:
   StateVariableCoef    *a;
   StateVariableVecCoef *b;

   VectorCoefficient *b_std;

   mutable Vector dV_;

public:
   // Result is A * B
   StateVariableScalarVectorProductCoef(StateVariableCoef &A,
                                        StateVariableVecCoef &B)
      : StateVariableVecCoef(B.GetVDim()),
        a(A.Clone()), b(B.Clone()), b_std(NULL), dV_(B.GetVDim()) {}

   StateVariableScalarVectorProductCoef(StateVariableCoef &A,
                                        VectorCoefficient &B)
      : StateVariableVecCoef(B.GetVDim()),
        a(A.Clone()), b(NULL), b_std(&B) {}

   ~StateVariableScalarVectorProductCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableScalarVectorProductCoef * Clone() const
   {
      if (b != NULL)
      {
         return new StateVariableScalarVectorProductCoef(*a, *b);
      }
      else
      {
         return new StateVariableScalarVectorProductCoef(*a, *b_std);
      }
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableVecCoef &B) { b = &B; b_std = NULL; }
   StateVariableVecCoef * GetBCoef() const { return b; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) ||
             (b ? b->NonTrivialValue(deriv) : false);
   }

   /// Evaluate the coefficient
   virtual void Eval_Func(Vector &V, ElementTransformation &T,
                          const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_Func(T, ip);
   }

   virtual void Eval_dNn(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dNn(T, ip);
      if (b)
      {
         b->Eval_dNn(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dNi(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dNi(T, ip);
      if (b)
      {
         b->Eval_dNi(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dVi(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dVi(T, ip);
      if (b)
      {
         b->Eval_dVi(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dTi(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dTi(T, ip);
      if (b)
      {
         b->Eval_dTi(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dTe(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dTe(T, ip);
      if (b)
      {
         b->Eval_dTe(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }
};

class StateVariableScalarMatrixProductCoef : public StateVariableMatCoef
{
private:
   StateVariableCoef    *a;
   StateVariableMatCoef *b;

   mutable DenseMatrix dM_;

public:
   // Result is A * B
   StateVariableScalarMatrixProductCoef(StateVariableCoef &A,
                                        StateVariableMatCoef &B)
      : StateVariableMatCoef(B.GetHeight(), B.GetWidth()),
        a(A.Clone()), b(B.Clone()), dM_(B.GetHeight(), B.GetWidth()) {}

   ~StateVariableScalarMatrixProductCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableScalarMatrixProductCoef * Clone() const
   {
      return new StateVariableScalarMatrixProductCoef(*a, *b);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableMatCoef &B) { b = &B; }
   StateVariableMatCoef * GetBCoef() const { return b; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) || b->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual void Eval_Func(DenseMatrix &M, ElementTransformation &T,
                          const IntegrationPoint &ip)
   { b->Eval_Func(M, T, ip); M *= a->Eval_Func(T, ip); }

   virtual void Eval_dNn(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dNn(T, ip);
      b->Eval_dNn(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dNi(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dNi(T, ip);
      b->Eval_dNi(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dVi(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dVi(T, ip);
      b->Eval_dVi(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dTi(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dTi(T, ip);
      b->Eval_dTi(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dTe(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dTe(T, ip);
      b->Eval_dTe(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }
};

class CoulombLogEICoef : public StateVariableCoef
{
private:
   int zi_;
   Coefficient *TeCoef_;
   Coefficient *neCoef_;

public:
   CoulombLogEICoef(Coefficient &TeCoef, Coefficient &neCoef, int zi)
      : zi_(zi), TeCoef_(&TeCoef), neCoef_(&neCoef) {}

   CoulombLogEICoef(const CoulombLogEICoef &other)
   {
      derivType_ = other.derivType_;
      zi_        = other.zi_;
      TeCoef_    = other.TeCoef_;
      neCoef_    = other.neCoef_;
   }

   virtual CoulombLogEICoef * Clone() const
   {
      return new CoulombLogEICoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);
      double ne = std::max(neCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(ne > 0.0,
      //          "CoulombLogEICoef::Eval_Func: "
      //          "Electron density (" << ne << ") "
      //          "less than or equal to zero.");
      // MFEM_VERIFY(Te > 0.0,
      //          "CoulombLogEICoef::Eval_Func: "
      //          "Electron temperature (" << Te << ") "
      //          "less than or equal to zero.");

      return lambda_ei(Te, ne, (double)zi_);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ne = std::max(neCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(ne > 0.0,
      //           "CoulombLogEICoef::Eval_dNi: "
      //           "Electron density (" << ne << ") "
      //           "less than or equal to zero.");

      return dlambda_ei_dne(1.0, ne, (double)zi_) * (double)zi_;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te > 0.0,
      //           "CoulombLogEICoef::Eval_dTe: "
      //           "Electron temperature (" << Te << ") "
      //           "less than or equal to zero.");

      return dlambda_ei_dne(Te, 1.0, (double)zi_);
   }
};

class IonCollisionTimeCoef : public StateVariableCoef
{
private:
   int z_i_;
   double m_i_;
   Coefficient &niCoef_;
   Coefficient &TiCoef_;
   StateVariableCoef &lnLambda_;

public:
   IonCollisionTimeCoef(int zi, double mi,
                        Coefficient &niCoef,
                        Coefficient &TiCoef,
                        StateVariableCoef &lnLambda)
      : z_i_(zi), m_i_(mi), niCoef_(niCoef), TiCoef_(TiCoef),
        lnLambda_(lnLambda) {}

   IonCollisionTimeCoef(const IonCollisionTimeCoef &other)
      : niCoef_(other.niCoef_),
        TiCoef_(other.TiCoef_),
        lnLambda_(other.lnLambda_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_i_       = other.m_i_;
   }

   virtual IonCollisionTimeCoef * Clone() const
   {
      return new IonCollisionTimeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_Func: "
      //          "Negative temperature found");

      return tau_i(m_i_, (double)z_i_, ni, Ti, lnLambda);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dNi(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_dNi: "
      //          "Negative temperature found");

      double dtau_dn = dtau_i_dni(m_i_, (double)z_i_, ni, Ti, lnLambda);
      double dtau_dl = dtau_i_dlambda(m_i_, (double)z_i_, ni, Ti, lnLambda);

      return dtau_dn + dtau_dl * dl;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_dTi: "
      //          "Negative temperature found");

      return dtau_i_dTi(m_i_, (double)z_i_, ni, Ti, lnLambda);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dTe(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_dTe: "
      //          "Negative temperature found");

      double dtau_dl = dtau_i_dlambda(m_i_, (double)z_i_, ni, Ti, lnLambda);

      return dtau_dl * dl;
   }
};

class ElectronCollisionTimeCoef : public StateVariableCoef
{
private:
   int z_i_;
   Coefficient &niCoef_;
   Coefficient &TeCoef_;
   StateVariableCoef &lnLambda_;

public:
   ElectronCollisionTimeCoef(int zi,
                             Coefficient &niCoef,
                             Coefficient &TeCoef,
                             StateVariableCoef &lnLambda)
      : z_i_(zi), niCoef_(niCoef), TeCoef_(TeCoef),
        lnLambda_(lnLambda) {}

   ElectronCollisionTimeCoef(const ElectronCollisionTimeCoef &other)
      : niCoef_(other.niCoef_),
        TeCoef_(other.TeCoef_),
        lnLambda_(other.lnLambda_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
   }

   virtual ElectronCollisionTimeCoef * Clone() const
   {
      return new ElectronCollisionTimeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);

      // MFEM_VERIFY(Te >= 0.0, "ElectronCollisionTimeCoef::Eval_Func: "
      //          "Negative temperature found");

      return tau_e(Te, (double)z_i_, ni, lnLambda);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dNi(T, ip);

      // MFEM_VERIFY(Te >= 0.0, "ElectronCollisionTimeCoef::Eval_dNi: "
      //          "Negative temperature found");

      double dtau_dn = dtau_e_dni(Te, (double)z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, (double)z_i_, ni, lnLambda);

      return dtau_dn + dtau_dl * dl;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dTe(T, ip);

      // MFEM_VERIFY(Te >= 0.0, "ElectronCollisionTimeCoef::Eval_dTe: "
      //          "Negative temperature found");

      double dtau_dT = dtau_e_dTe(Te, (double)z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, (double)z_i_, ni, lnLambda);

      return dtau_dT + dtau_dl * dl;
   }
};

/** Given the ion and electron temperatures in eV this coefficient returns an
    approximation to the sound speed in m/s.
*/
class SoundSpeedCoef : public StateVariableCoef
{
private:
   double mi_kg_;
   Coefficient *TiCoef_;
   Coefficient *TeCoef_;

public:
   SoundSpeedCoef(double ion_mass_kg, Coefficient &TiCoef, Coefficient &TeCoef)
      : mi_kg_(ion_mass_kg), TiCoef_(&TiCoef), TeCoef_(&TeCoef) {}

   SoundSpeedCoef(const SoundSpeedCoef &other)
   {
      derivType_ = other.derivType_;
      mi_kg_     = other.mi_kg_;
      TiCoef_    = other.TiCoef_;
      TeCoef_    = other.TeCoef_;
   }

   virtual SoundSpeedCoef * Clone() const
   {
      return new SoundSpeedCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Ti = TiCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);

      // MFEM_VERIFY(Ti + Te >= 0.0,
      //          "SoundSpeedCoef::Eval_Func: Negative temperature found");

      return sqrt(J_per_eV_ * std::max(Ti + Te, 0.0) / mi_kg_);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = TiCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);

      // MFEM_VERIFY(Ti + Te >= 0.0,
      //          "SoundSpeedCoef::Eval_dTi: Negative temperature found");

      return 0.5 * sqrt(J_per_eV_ / (mi_kg_ * std::max(Ti + Te, 1.0)));
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = TiCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);

      // MFEM_VERIFY(Ti + Te >= 0.0,
      //          "SoundSpeedCoef::Eval_dTe: Negative temperature found");

      return 0.5 * sqrt(J_per_eV_ / (mi_kg_ * std::max(Ti + Te, 1.0)));
   }

};

/** Given the electron temperature in eV this coefficient returns an
    approximation to the expected ionization rate in m^3/s.
*/
class ApproxIonizationRate : public StateVariableCoef
{
private:
   Coefficient *TeCoef_;

public:
   ApproxIonizationRate(Coefficient &TeCoef)
      : TeCoef_(&TeCoef) {}

   ApproxIonizationRate(const ApproxIonizationRate &other)
   {
      derivType_ = other.derivType_;
      TeCoef_    = other.TeCoef_;
   }

   virtual ApproxIonizationRate * Clone() const
   {
      return new ApproxIonizationRate(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te2 = pow(TeCoef_->Eval(T, ip), 2);

      return 3.0e-16 * Te2 / (3.0 + 0.01 * Te2);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = TeCoef_->Eval(T, ip);

      return 2.0 * 3.0 * 3.0e-16 * Te / pow(3.0 + 0.01 * Te * Te, 2);
   }

};

/** Given the electron temperature in eV this coefficient returns an
    approximation to the expected recombination rate in m^3/s.
*/
class ApproxRecombinationRate : public StateVariableCoef
{
private:
   Coefficient *TeCoef_;

public:
   ApproxRecombinationRate(Coefficient &TeCoef)
      : TeCoef_(&TeCoef) {}

   ApproxRecombinationRate(const ApproxRecombinationRate &other)
   {
      derivType_ = other.derivType_;
      TeCoef_    = other.TeCoef_;
   }

   virtual ApproxRecombinationRate * Clone() const
   {
      return new ApproxRecombinationRate(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);

      int    pqn_c = 1;
      int    xsi_c = 2;
      double Ry    = 13.6;
      double chi_c = 13.61;
      double E_ion = Ry;

      // MFEM_VERIFY(Te >= 0.0, "ApproxRecombinationRate::Eval_Func: "
      //          "Negative temperature found");

      return 5.2e-20 / 2.0 * sqrt(E_ion / Te)
             * (1.0 - exp(-chi_c / Te
                          * (1.0 + 1.0 / pqn_c * (xsi_c / pow(pqn_c, 2) - 1.0))))
             * sqrt(pow(log(chi_c / Te), 2) + 2.0);

      // double Te2 = pow(Te, 2);
      // return 3.0e-19 * Te2 / (3.0 + 0.01 * Te2);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "ApproxRecombinationRate::Eval_dTe: "
      //          "Negative temperature found");

      int    pqn_c = 1;
      int    xsi_c = 2;
      double Ry    = 13.6;
      double chi_c = 13.61;
      double E_ion = Ry;

      double a = log(chi_c / Te);
      double b = 2.0 + pow(a, 2);
      double c = exp(-chi_c / Te
                     * (1.0 + 1.0 / pqn_c * (xsi_c / pow(pqn_c, 2) - 1.0)));

      return -5.2e-20  * 0.5 * sqrt(E_ion / (b * pow(Te, 3))) *
             ((a + 0.5 * b) * (1.0 - c)
              + c * (xsi_c - b * chi_c * pqn_c * pqn_c
                     * (1.0 - pqn_c) / (pow(pqn_c, 3) * Te)));

      // return 2.0 * 3.0 * 3.0e-19 * Te / pow(3.0 + 0.01 * Te * Te, 2);
   }

};

/** Given the ion temperature in eV this coefficient returns an
    approximation to the expected charge exchange rate in m^3/s.
*/
class ApproxChargeExchangeRate : public StateVariableCoef
{
private:
   Coefficient *TiCoef_;

public:
   ApproxChargeExchangeRate(Coefficient &TiCoef)
      : TiCoef_(&TiCoef) {}

   ApproxChargeExchangeRate(const ApproxChargeExchangeRate &other)
   {
      derivType_ = other.derivType_;
      TiCoef_    = other.TiCoef_;
   }

   virtual ApproxChargeExchangeRate * Clone() const
   {
      return new ApproxChargeExchangeRate(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_->Eval(T, ip), 0.0);

      // MFEM_VERIFY(Ti >= 0.0, "ApproxChargeExchangeRate::Eval_Func: "
      //            "Negative temperature found");

      return 1.0e-14 * pow(Ti, 0.318);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_->Eval(T, ip), 0.0);

      // MFEM_VERIFY(Ti >= 0.0, "ApproxChargeExchangeRate::Eval_dTi: "
      //            "Negative temperature found");

      return 0.318 * 1.0e-14 * pow(Ti, 0.318 - 1.0);
   }

};

/** Diffusion coefficient used in the Neutral density equation

   Dn = v_n^2 / [3 n_e (<sigma nu>_{iz} + <sigma nu>_{cx})]

   Where:
      v_n is the average speed of the Neutrals often labeled \bar{v}_n and
          given by the formula \bar{v}_n = sqrt{8 T_n / (pi m_n)}
      n_e is the electron density
      <sigma nu>_{iz} is the ionization rate
      <sigma nu>_{cx} is the charge exchange rate
*/
class NeutralDiffusionCoef : public StateVariableCoef
{
private:
   Coefficient       * ne_;
   Coefficient       * vn_;
   Coefficient       * iz_;
   Coefficient       * cx_;

   StateVariableCoef * ne_sv_;
   StateVariableCoef * vn_sv_;
   StateVariableCoef * iz_sv_;
   StateVariableCoef * cx_sv_;

public:
   NeutralDiffusionCoef(Coefficient &neCoef, Coefficient &vnBarCoef,
                        Coefficient &izCoef, Coefficient &cxCoef)
      : ne_(&neCoef), vn_(&vnBarCoef), iz_(&izCoef), cx_(&cxCoef)
   {
      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      vn_sv_ = dynamic_cast<StateVariableCoef*>(vn_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
      cx_sv_ = dynamic_cast<StateVariableCoef*>(cx_);
   }

   NeutralDiffusionCoef(const NeutralDiffusionCoef &other)
   {
      derivType_ = other.derivType_;
      ne_ = other.ne_;
      vn_ = other.vn_;
      iz_ = other.iz_;
      cx_ = other.cx_;

      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      vn_sv_ = dynamic_cast<StateVariableCoef*>(vn_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
      cx_sv_ = dynamic_cast<StateVariableCoef*>(cx_);
   }

   virtual NeutralDiffusionCoef * Clone() const
   {
      return new NeutralDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool dvn = (vn_sv_) ? vn_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;
      bool dcx = (cx_sv_) ? cx_sv_->NonTrivialValue(deriv) : false;

      return (deriv == INVALID_FIELD_TYPE || dne || dvn || diz || dcx);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ne = ne_->Eval(T, ip);
      double vn = vn_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);
      double cx = cx_->Eval(T, ip);

      return vn * vn / (3.0 * ne * (iz + cx));
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool dvn = (vn_sv_) ? vn_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;
      bool dcx = (cx_sv_) ? cx_sv_->NonTrivialValue(deriv) : false;

      if (!dne && !dvn && !diz && !dcx)
      {
         return 0.0;
      }

      double ne = ne_->Eval(T, ip);
      double vn = vn_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);
      double cx = cx_->Eval(T, ip);

      if (ne_sv_) { ne_sv_->SetDerivType(deriv); }
      if (vn_sv_) { vn_sv_->SetDerivType(deriv); }
      if (iz_sv_) { iz_sv_->SetDerivType(deriv); }
      if (cx_sv_) { cx_sv_->SetDerivType(deriv); }

      double dne_df = (ne_sv_) ? ne_sv_->Eval(T, ip) : 0.0;
      double dvn_df = (vn_sv_) ? vn_sv_->Eval(T, ip) : 0.0;
      double diz_df = (iz_sv_) ? iz_sv_->Eval(T, ip) : 0.0;
      double dcx_df = (cx_sv_) ? cx_sv_->Eval(T, ip) : 0.0;

      if (ne_sv_) { ne_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (vn_sv_) { vn_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (iz_sv_) { iz_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (cx_sv_) { cx_sv_->SetDerivType(INVALID_FIELD_TYPE); }

      // vn * vn / (3.0 * ne * (iz + cx));
      return (2.0 * dvn_df - vn * (dne_df / ne + (diz_df + dcx_df) / (iz + cx)))
             * vn / (3.0 * ne * (iz + cx));
   }
};

class IonDiffusionCoef : public StateVariableMatCoef
{
private:
   Coefficient       * Dperp_;
   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   IonDiffusionCoef(Coefficient &DperpCoef, VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2), Dperp_(&DperpCoef), B3_(&B3Coef), B_(3) {}

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(DenseMatrix & M,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      M.SetSize(2);

      double Dperp = Dperp_->Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;

      M(0,0) = (B_[1] * B_[1] + B_[2] * B_[2]) * Dperp / Bmag2;
      M(0,1) = -B_[0] * B_[1] * Dperp / Bmag2;
      M(1,0) = M(0,1);
      M(1,1) = (B_[0] * B_[0] + B_[2] * B_[2]) * Dperp / Bmag2;
   }

};

struct ArtViscParams
{
   double log_width;
   double art_visc_amp;
   double osc_thresh;
};

/**
*/
class IonDensityParaDiffusionCoef : public StateVariableCoef
{
private:
   StateVariableCoef & CsCoef_;

   Coefficient       * elOrdCoef_;
   Coefficient       * OscCoef_;
   Coefficient       * hCoef_;
   const double width_;
   const double avisc_;

public:
   IonDensityParaDiffusionCoef(StateVariableCoef &CsCoef,
                               Coefficient * elOrdCoef,
                               Coefficient * OscCoef,
                               Coefficient * hCoef,
                               const ArtViscParams &av)
      : CsCoef_(CsCoef),
        elOrdCoef_(elOrdCoef),
        OscCoef_(OscCoef),
        hCoef_(hCoef),
        width_(av.log_width),
        avisc_(av.art_visc_amp)
   {}

   IonDensityParaDiffusionCoef(const IonDensityParaDiffusionCoef &other)
      : CsCoef_(other.CsCoef_),
        elOrdCoef_(other.elOrdCoef_),
        OscCoef_(other.OscCoef_),
        hCoef_(other.hCoef_),
        width_(other.width_),
        avisc_(other.avisc_)
   {
      derivType_ = other.derivType_;
   }

   virtual IonDensityParaDiffusionCoef * Clone() const
   {
      return new IonDensityParaDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double DPara = 0.0;

      if (OscCoef_)
      {
         double elemOrder = elOrdCoef_->Eval(T, ip);
         double s0 = -4.0 * log10(elemOrder);

         double Se = OscCoef_->Eval(T, ip);
         double se = log10(Se);

         double eps = 0.0;
         if (se >= s0 - width_)
         {
            double Cs = CsCoef_.Eval(T, ip);
            double h  = hCoef_->Eval(T, ip);

            double eps0 = avisc_ * Cs * h;

            if (se > s0 + width_)
            {
               eps = eps0;
            }
            else
            {
               eps = 0.5 * eps0 * (1.0 + sin(0.5 * M_PI * (se - s0) / width_));
            }
         }

         DPara += eps;
      }
      return DPara;
   }
};

class IonAdvectionCoef : public StateVariableVecCoef
{
private:
   // double dt_;

   StateVariableCoef &vi_;
   // GridFunctionCoefficient vi0_;
   // GridFunctionCoefficient dvi0_;

   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   IonAdvectionCoef(StateVariableCoef &vi,
                    VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        vi_(vi),
        B3_(&B3Coef), B_(3) {}

   IonAdvectionCoef(const IonAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        vi_(other.vi_),
        B3_(other.B3_),
        B_(3)
   {}

   virtual IonAdvectionCoef * Clone() const
   {
      return new IonAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_PARA_VELOCITY);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double vi = vi_.Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = vi * B_[0] / Bmag;
      V[1] = vi * B_[1] / Bmag;
   }

   void Eval_dVi(Vector &V, ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
   }
};

class IonizationSourceCoef : public StateVariableCoef
{
private:
   Coefficient        * ne_;
   Coefficient        * nn_;
   Coefficient        * iz_;

   StateVariableCoef  * ne_sv_;
   StateVariableCoef  * iz_sv_;

   double nn0_;

public:
   IonizationSourceCoef(Coefficient &neCoef, Coefficient &nnCoef,
                        Coefficient &izCoef,
                        double nn0 = 1e10)
      : ne_(&neCoef), nn_(&nnCoef), iz_(&izCoef), nn0_(nn0)
   {
      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
   }

   IonizationSourceCoef(const IonizationSourceCoef &other)
   {
      derivType_ = other.derivType_;
      ne_  = other.ne_;
      nn_  = other.nn_;
      iz_  = other.iz_;
      nn0_ = other.nn0_;

      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
   }

   virtual IonizationSourceCoef * Clone() const
   {
      return new IonizationSourceCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;

      return (deriv == INVALID_FIELD_TYPE || deriv == NEUTRAL_DENSITY || dne || diz);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nn_->Eval(T, ip);
      if (nn < nn0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);

      return ne * nn * iz;
   }

   double Eval_dFunc(FieldType deriv, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dnn = (deriv == NEUTRAL_DENSITY);
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;

      if (!dnn && !dne && !diz)
      {
         return 0.0;
      }

      double nn = nn_->Eval(T, ip);
      if (nn < nn0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);

      if (ne_sv_) { ne_sv_->SetDerivType(deriv); }
      if (iz_sv_) { iz_sv_->SetDerivType(deriv); }

      double dnn_df = (deriv == NEUTRAL_DENSITY) ? 1.0 : 0.0;
      double dne_df = (ne_sv_) ? ne_sv_->Eval(T, ip) : 0.0;
      double diz_df = (iz_sv_) ? iz_sv_->Eval(T, ip) : 0.0;

      if (ne_sv_) { ne_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (iz_sv_) { iz_sv_->SetDerivType(INVALID_FIELD_TYPE); }

      //  s_ * ne * nn * iz;
      return dne_df * nn * iz + ne * dnn_df * iz + ne * nn * diz_df;
   }
};

class RecombinationSinkCoef : public StateVariableCoef
{
private:
   Coefficient        * ne_;
   Coefficient        * ni_;
   Coefficient        * rc_;

   StateVariableCoef  * ne_sv_;
   StateVariableCoef  * rc_sv_;

   double ni0_;

public:
   RecombinationSinkCoef(Coefficient &neCoef, Coefficient &niCoef,
                         Coefficient &rcCoef, double ni0 = 1e10)
      : ne_(&neCoef), ni_(&niCoef), rc_(&rcCoef), ni0_(ni0)
   {
      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      rc_sv_ = dynamic_cast<StateVariableCoef*>(rc_);
   }

   RecombinationSinkCoef(const RecombinationSinkCoef &other)
   {
      derivType_ = other.derivType_;
      ne_ = other.ne_;
      ni_ = other.ni_;
      rc_ = other.rc_;

      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      rc_sv_ = dynamic_cast<StateVariableCoef*>(rc_);

      ni0_ = other.ni0_;
   }

   virtual RecombinationSinkCoef * Clone() const
   {
      return new RecombinationSinkCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool drc = (rc_sv_) ? rc_sv_->NonTrivialValue(deriv) : false;

      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || dne || drc);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = ni_->Eval(T, ip);
      if (ni < ni0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double rc = rc_->Eval(T, ip);

      return ne * ni * rc;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dni = (deriv == ION_DENSITY);
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool drc = (rc_sv_) ? rc_sv_->NonTrivialValue(deriv) : false;

      if (!dni && !dne && !drc)
      {
         return 0.0;
      }

      double ni = ni_->Eval(T, ip);
      if (ni < ni0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double rc = rc_->Eval(T, ip);

      if (ne_sv_) { ne_sv_->SetDerivType(deriv); }
      if (rc_sv_) { rc_sv_->SetDerivType(deriv); }

      double dni_df = (deriv == ION_DENSITY) ? 1.0 : 0.0;
      double dne_df = (ne_sv_) ? ne_sv_->Eval(T, ip) : 0.0;
      double drc_df = (rc_sv_) ? rc_sv_->Eval(T, ip) : 0.0;

      if (ne_sv_) { ne_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (rc_sv_) { rc_sv_->SetDerivType(INVALID_FIELD_TYPE); }

      /// ne * ni * rc
      return dne_df * ni * rc + ne * dni_df * rc + ne * ni * drc_df;
   }
};

class ChargeExchangeSinkCoef : public StateVariableCoef
{
private:
   Coefficient        & nn_;
   Coefficient        & ni_;
   Coefficient        & cx_;

   StateVariableCoef  * cx_sv_;

public:
   ChargeExchangeSinkCoef(Coefficient &nnCoef, Coefficient &niCoef,
                          Coefficient &cxCoef)
      : nn_(nnCoef), ni_(niCoef), cx_(cxCoef)
   {
      cx_sv_ = dynamic_cast<StateVariableCoef*>(&cx_);
   }

   ChargeExchangeSinkCoef(const ChargeExchangeSinkCoef &other)
      : nn_(other.nn_), ni_(other.ni_), cx_(other.cx_)
   {
      derivType_ = other.derivType_;
      cx_sv_ = dynamic_cast<StateVariableCoef*>(&cx_);
   }

   virtual ChargeExchangeSinkCoef * Clone() const
   {
      return new ChargeExchangeSinkCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dcx = cx_sv_->NonTrivialValue(deriv);

      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || dcx);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nn_.Eval(T, ip);
      double ni = ni_.Eval(T, ip);
      double cx = cx_.Eval(T, ip);

      return nn * ni * cx;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dni = (deriv == ION_DENSITY);
      bool dcx = cx_sv_->NonTrivialValue(deriv);

      if (!dni && !dcx)
      {
         return 0.0;
      }

      double nn = nn_.Eval(T, ip);
      double ni = ni_.Eval(T, ip);
      double cx = cx_.Eval(T, ip);

      double dni_df = (deriv == ION_DENSITY) ? 1.0 : 0.0;
      double dcx_df = cx_sv_->Eval_dFunc(deriv, T, ip);

      /// nn * ni * cx
      return nn * (dni_df * cx + ni * dcx_df);
   }
};

class IonMomentumParaCoef : public StateVariableCoef
{
private:
   double m_i_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;

public:
   IonMomentumParaCoef(double m_i,
                       StateVariableCoef &niCoef,
                       StateVariableCoef &viCoef)
      : m_i_(m_i), niCoef_(niCoef), viCoef_(viCoef) {}

   IonMomentumParaCoef(const IonMomentumParaCoef &other)
      : niCoef_(other.niCoef_),
        viCoef_(other.viCoef_)
   {
      derivType_ = other.derivType_;
      m_i_ = other.m_i_;
   }

   virtual IonMomentumParaCoef * Clone() const
   {
      return new IonMomentumParaCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == ION_PARA_VELOCITY);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval_Func(T, ip);
      double vi = viCoef_.Eval_Func(T, ip);

      return m_i_ * ni * vi;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval_Func(T, ip);

      return m_i_ * vi;
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval_Func(T, ip);

      return m_i_ * ni;
   }
};

/**
   The momentum equation uses a mass specified in kg and particle
   density in particles per meter cubed. Therefore, the diffusion
   coefficient must have units of kg / (meter * second).
*/
class IonMomentumParaDiffusionCoef : public StateVariableCoef
{
private:
   double z_i_;
   double m_i_kg_;
   const double a_;

   StateVariableCoef & lnLambda_;
   StateVariableCoef & TiCoef_;

   StateVariableCoef & niCoef_;
   StateVariableCoef & CsCoef_;

   Coefficient       * elOrdCoef_;
   Coefficient       * OscCoef_;
   Coefficient       * hCoef_;
   const double width_;
   const double avisc_;

public:
   IonMomentumParaDiffusionCoef(int ion_charge_number, double ion_mass_kg,
                                StateVariableCoef &lnLambda,
                                StateVariableCoef &TiCoef,
                                StateVariableCoef &niCoef,
                                StateVariableCoef &CsCoef,
                                Coefficient * elOrdCoef,
                                Coefficient * OscCoef,
                                Coefficient * hCoef,
                                const ArtViscParams &av)
      : z_i_((double)ion_charge_number), m_i_kg_(ion_mass_kg),
        a_(0.96 * tau_i(m_i_kg_, z_i_, 1.0, 1.0/J_per_eV_, 1.0)),
        lnLambda_(lnLambda),
        TiCoef_(TiCoef),
        niCoef_(niCoef),
        CsCoef_(CsCoef),
        elOrdCoef_(elOrdCoef),
        OscCoef_(OscCoef),
        hCoef_(hCoef),
        width_(av.log_width),
        avisc_(av.art_visc_amp)
   {}

   IonMomentumParaDiffusionCoef(const IonMomentumParaDiffusionCoef &other)
      : z_i_(other.z_i_), m_i_kg_(other.m_i_kg_),
        a_(0.96 * tau_i(m_i_kg_, z_i_, 1.0, 1.0/J_per_eV_, 1.0)),
        lnLambda_(other.lnLambda_),
        TiCoef_(other.TiCoef_),
        niCoef_(other.niCoef_),
        CsCoef_(other.CsCoef_),
        elOrdCoef_(other.elOrdCoef_),
        OscCoef_(other.OscCoef_),
        hCoef_(other.hCoef_),
        width_(other.width_),
        avisc_(other.avisc_)
   {
      derivType_ = other.derivType_;
   }

   virtual IonMomentumParaDiffusionCoef * Clone() const
   {
      return new IonMomentumParaDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval(T, ip);
      double Ti_J = std::max(TiCoef_.Eval_Func(T, ip), 1.0) * J_per_eV_;

      double EtaPara = a_ * sqrt(pow(Ti_J, 5)) / lnLambda;

      if (OscCoef_)
      {
         double elemOrder = elOrdCoef_->Eval(T, ip);
         double s0 = -4.0 * log10(elemOrder);

         double Se = OscCoef_->Eval(T, ip);
         double se = log10(Se);

         double eps = 0.0;
         if (se >= s0 - width_)
         {
            double ni = std::max(niCoef_.Eval(T, ip), 1.0);
            double Cs = CsCoef_.Eval(T, ip);
            double h  = hCoef_->Eval(T, ip);

            double eps0 = avisc_ * m_i_kg_ * ni * Cs * h;

            if (se > s0 + width_)
            {
               eps = eps0;
            }
            else
            {
               eps = 0.5 * eps0 * (1.0 + sin(0.5 * M_PI * (se - s0) / width_));
            }
         }

         EtaPara += eps;
      }
      return EtaPara;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval(T, ip);
      double Ti_J = std::max(TiCoef_.Eval_Func(T, ip), 1.0) * J_per_eV_;

      double dEtaPara = 2.5 * a_ * sqrt(pow(Ti_J, 3)) * J_per_eV_ / lnLambda;
      return dEtaPara;
   }

};

class IonMomentumPerpDiffusionCoef : public StateVariableCoef
{
private:
   double Dperp_;
   double m_i_kg_;

   StateVariableCoef * niCoef_;

public:
   IonMomentumPerpDiffusionCoef(double Dperp, double ion_mass_kg,
                                StateVariableCoef &niCoef)
      : Dperp_(Dperp), m_i_kg_(ion_mass_kg),
        niCoef_(&niCoef)
   {}

   IonMomentumPerpDiffusionCoef(const IonMomentumPerpDiffusionCoef &other)
      : Dperp_(other.Dperp_), m_i_kg_(other.m_i_kg_),
        niCoef_(other.niCoef_)
   {
      derivType_ = other.derivType_;
   }

   virtual IonMomentumPerpDiffusionCoef * Clone() const
   {
      return new IonMomentumPerpDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_DENSITY);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_->Eval_Func(T, ip);
      double EtaPerp = Dperp_ * m_i_kg_ * ni;
      return EtaPerp;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double dEtaPerp = Dperp_ * m_i_kg_;
      return dEtaPerp;
   }

};
/*
class IonMomentumDiffusionCoef : public StateVariableMatCoef
{
private:
   double zi_;
   double mi_;
   const double lnLambda_;
   const double a_;

   Coefficient       * Dperp_;
   Coefficient       * niCoef_;
   Coefficient       * TiCoef_;
   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   IonMomentumDiffusionCoef(int ion_charge, double ion_mass,
                            Coefficient &DperpCoef,
                            Coefficient &niCoef, Coefficient &TiCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2), zi_((double)ion_charge), mi_(ion_mass),
        lnLambda_(17.0),
        a_(0.96 * 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
           sqrt(mi_ * kg_per_amu_ * pow(J_per_eV_, 5) / M_PI) /
           (lnLambda_ * pow(q_ * zi_, 4))),
        Dperp_(&DperpCoef), niCoef_(&niCoef), TiCoef_(&TiCoef),
        B3_(&B3Coef), B_(3) {}

   bool NonTrivialValue(DerivType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(DenseMatrix & M,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      M.SetSize(2);

      double Dperp = Dperp_->Eval(T, ip);

      double ni = niCoef_->Eval(T, ip);
      double Ti = TiCoef_->Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;

      double EtaPerp = mi_ * ni * Dperp;

      M(0,0) = (B_[1] * B_[1] + B_[2] * B_[2]) * EtaPerp / Bmag2;
      M(0,1) = -B_[0] * B_[1] * EtaPerp / Bmag2;
      M(1,0) = M(0,1);
      M(1,1) = (B_[0] * B_[0] + B_[2] * B_[2]) * EtaPerp / Bmag2;

      double EtaPara = a_ * sqrt(pow(Ti, 5));

      M(0,0) += B_[0] * B_[0] * EtaPara / Bmag2;
      M(0,1) += B_[0] * B_[1] * EtaPara / Bmag2;
      M(1,0) += B_[0] * B_[1] * EtaPara / Bmag2;
      M(1,1) += B_[1] * B_[1] * EtaPara / Bmag2;
   }

};
*/
class IonMomentumAdvectionCoef : public StateVariableVecCoef
{
private:
   double mi_;

   StateVariableCoef &ni_;
   StateVariableCoef &vi_;

   GradientGridFunctionCoefficient grad_ni0_;
   GradientGridFunctionCoefficient grad_dni0_;
   double dt_;

   Coefficient       * Dperp_;
   VectorCoefficient * B3_;

   mutable Vector gni_;
   mutable Vector gdni_;

   mutable Vector B_;

public:
   IonMomentumAdvectionCoef(StateVariableCoef &ni,
                            StateVariableCoef &vi,
                            double ion_mass,
                            Coefficient &DperpCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        mi_(ion_mass),
        ni_(ni),
        vi_(vi),
        grad_ni0_(dynamic_cast<StateVariableGridFunctionCoef*>
                  (dynamic_cast<StateVariableSumCoef&>(ni).GetACoef())->
                  GetGridFunction()),
        grad_dni0_(dynamic_cast<StateVariableGridFunctionCoef*>
                   (dynamic_cast<StateVariableSumCoef&>(ni).GetBCoef())->
                   GetGridFunction()),
        dt_(dynamic_cast<StateVariableSumCoef&>(ni).GetBeta()),
        Dperp_(&DperpCoef),
        B3_(&B3Coef), B_(3)
   {}

   IonMomentumAdvectionCoef(const IonMomentumAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        mi_(other.mi_),
        ni_(other.ni_),
        vi_(other.vi_),
        grad_ni0_(other.grad_ni0_.GetGridFunction()),
        grad_dni0_(other.grad_dni0_.GetGridFunction()),
        dt_(other.dt_),
        B3_(other.B3_),
        B_(3)
   {}

   virtual IonMomentumAdvectionCoef * Clone() const
   {
      return new IonMomentumAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni = ni_.Eval(T, ip);
      double vi = vi_.Eval(T, ip);

      // double dni0 = dni0_.Eval(T, ip);
      // double dvi0 = dvi0_.Eval(T, ip);

      // double ni1 = ni0 + dt_ * dni0;
      // double vi1 = vi0 + dt_ * dvi0;

      grad_ni0_.Eval(gni_, T, ip);
      grad_dni0_.Eval(gdni_, T, ip);
      // grad_dni0_.Eval(gdni0_, T, ip);
      gni_.Add(dt_, gdni_);

      // gni1_.SetSize(gni0_.Size());
      // add(gni0_, dt_, gdni0_, gni1_);

      double Dperp = Dperp_->Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = mi_ * (ni * vi * B_[0] / Bmag +
                    Dperp * ((B_[1] * B_[1] + B_[2] * B_[2]) * gni_[0] -
                             B_[0] * B_[1] * gni_[1]) / Bmag2);
      V[1] = mi_ * (ni * vi * B_[1] / Bmag +
                    Dperp * ((B_[0] * B_[0] + B_[2] * B_[2]) * gni_[1] -
                             B_[0] * B_[1] * gni_[0]) / Bmag2);
   }
};

class IonMomentumIonizationCoef : public StateVariableCoef
{
private:
   int zi_;
   double mi_;
   Coefficient &nnCoef_;
   Coefficient &niCoef_;
   Coefficient &vnCoef_;
   StateVariableCoef &izCoef_;

public:
   IonMomentumIonizationCoef(int z_i, double m_i_kg,
                             Coefficient &nnCoef,
                             Coefficient &niCoef,
                             Coefficient &vnCoef,
                             StateVariableCoef &izCoef)
      : zi_(z_i), mi_(m_i_kg),
        nnCoef_(nnCoef), niCoef_(niCoef), vnCoef_(vnCoef), izCoef_(izCoef)
   {}

   IonMomentumIonizationCoef(const IonMomentumIonizationCoef &other)
      : nnCoef_(other.nnCoef_), niCoef_(other.niCoef_),
        vnCoef_(other.vnCoef_), izCoef_(other.izCoef_)
   {
      derivType_ = other.derivType_;
      zi_        = other.zi_;
      mi_        = other.mi_;
   }

   virtual IonMomentumIonizationCoef * Clone() const
   {
      return new IonMomentumIonizationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == NEUTRAL_DENSITY ||
              deriv == ION_DENSITY ||
              izCoef_.NonTrivialValue(deriv) );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double iz = izCoef_.Eval_Func(T, ip);

      return mi_ * zi_ * nn * ni * vn * iz;
   }

   double Eval_dNn(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double iz = izCoef_.Eval_Func(T, ip);
      double diz = izCoef_.Eval_dNn(T, ip);

      return mi_ * zi_ * ni * vn * (iz + nn * diz);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double iz = izCoef_.Eval_Func(T, ip);
      double diz = izCoef_.Eval_dNi(T, ip);

      return mi_ * zi_ * vn * nn * (iz + ni * diz);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double diz = izCoef_.Eval_dVi(T, ip);

      return mi_ * zi_ * nn * ni * vn * diz;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double diz = izCoef_.Eval_dTi(T, ip);

      return mi_ * zi_ * nn * ni * vn * diz;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double diz = izCoef_.Eval_dTe(T, ip);

      return mi_ * zi_ * nn * ni * vn * diz;
   }
};

class IonMomentumRecombinationCoef : public StateVariableCoef
{
private:
   int zi_;
   double mi_;
   Coefficient &niCoef_;
   Coefficient &viCoef_;
   StateVariableCoef &rcCoef_;

public:
   IonMomentumRecombinationCoef(int z_i, double m_i_kg,
                                Coefficient &niCoef,
                                Coefficient &viCoef,
                                StateVariableCoef &rcCoef)
      : zi_(z_i), mi_(m_i_kg),
        niCoef_(niCoef), viCoef_(viCoef), rcCoef_(rcCoef)
   {}

   IonMomentumRecombinationCoef(const IonMomentumRecombinationCoef &other)
      : niCoef_(other.niCoef_), viCoef_(other.viCoef_), rcCoef_(other.rcCoef_)
   {
      derivType_ = other.derivType_;
      zi_        = other.zi_;
      mi_        = other.mi_;
   }

   virtual IonMomentumRecombinationCoef * Clone() const
   {
      return new IonMomentumRecombinationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY ||
              rcCoef_.NonTrivialValue(deriv) );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double rc = rcCoef_.Eval_Func(T, ip);

      return mi_ * zi_ * ni * ni * vi * rc;
   }

   double Eval_dNn(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double drc = rcCoef_.Eval_dNn(T, ip);

      return mi_ * zi_ * ni * ni * vi * drc;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double rc = rcCoef_.Eval_Func(T, ip);
      double drc = rcCoef_.Eval_dNi(T, ip);

      return mi_ * zi_ * vi * ni * (2.0 * rc + ni * drc);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double rc = rcCoef_.Eval_Func(T, ip);
      double drc = rcCoef_.Eval_dVi(T, ip);

      return mi_ * zi_ * ni * ni * (rc + vi * drc);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double drc = rcCoef_.Eval_dTi(T, ip);

      return mi_ * zi_ * ni * ni * vi * drc;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double drc = rcCoef_.Eval_dTe(T, ip);

      return mi_ * zi_ * ni * ni * vi * drc;
   }
};

class IonMomentumChargeExchangeCoef : public StateVariableCoef
{
private:
   double mi_;
   Coefficient &nnCoef_;
   Coefficient &niCoef_;
   Coefficient &vnCoef_;
   Coefficient &viCoef_;
   StateVariableCoef &cxCoef_;

public:
   IonMomentumChargeExchangeCoef(double m_i_kg,
                                 Coefficient &nnCoef,
                                 Coefficient &niCoef,
                                 Coefficient &vnCoef,
                                 Coefficient &viCoef,
                                 StateVariableCoef &cxCoef)
      : mi_(m_i_kg), nnCoef_(nnCoef), niCoef_(niCoef),
        vnCoef_(vnCoef), viCoef_(viCoef), cxCoef_(cxCoef)
   {}

   IonMomentumChargeExchangeCoef(const IonMomentumChargeExchangeCoef &other)
      : nnCoef_(other.nnCoef_), niCoef_(other.niCoef_),
        vnCoef_(other.vnCoef_), viCoef_(other.viCoef_),
        cxCoef_(other.cxCoef_)
   {
      derivType_ = other.derivType_;
      mi_        = other.mi_;
   }

   virtual IonMomentumChargeExchangeCoef * Clone() const
   {
      return new IonMomentumChargeExchangeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == NEUTRAL_DENSITY ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY ||
              cxCoef_.NonTrivialValue(deriv) );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);

      return mi_ * nn * ni * (vn - vi) * cx;
   }

   double Eval_dNn(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);
      double dcx = cxCoef_.Eval_dNn(T, ip);

      return mi_ * ni * (vn - vi) * (cx + nn * dcx);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);
      double dcx = cxCoef_.Eval_dNi(T, ip);

      return mi_ * nn * (vn - vi) * (cx + ni * dcx);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);
      double dcx = cxCoef_.Eval_dVi(T, ip);

      return mi_ * nn * ni * ((vn - vi) * dcx - cx);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double dcx = cxCoef_.Eval_dTi(T, ip);

      return mi_ * nn * ni * (vn - vi) * dcx;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double dcx = cxCoef_.Eval_dTe(T, ip);

      return mi_ * nn * ni * (vn - vi) * dcx;
   }
};

class StaticPressureCoef : public StateVariableCoef
{
private:
   FieldType fieldType_;
   int z_i_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &TCoef_;

public:
   StaticPressureCoef(StateVariableCoef &niCoef,
                      StateVariableCoef &TCoef)
      : fieldType_(ION_TEMPERATURE),
        z_i_(1), niCoef_(niCoef), TCoef_(TCoef) {}

   StaticPressureCoef(int z_i,
                      StateVariableCoef &niCoef,
                      StateVariableCoef &TCoef)
      : fieldType_(ELECTRON_TEMPERATURE),
        z_i_(z_i), niCoef_(niCoef), TCoef_(TCoef) {}

   StaticPressureCoef(const StaticPressureCoef &other)
      : niCoef_(other.niCoef_),
        TCoef_(other.TCoef_)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      z_i_       = other.z_i_;
   }

   virtual StaticPressureCoef * Clone() const
   {
      return new StaticPressureCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == fieldType_);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double Ts = TCoef_.Eval(T, ip) * J_per_eV_;

      return 1.5 * z_i_ * ni * Ts;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ts = TCoef_.Eval(T, ip) * J_per_eV_;

      return 1.5 * z_i_ * Ts;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ION_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ELECTRON_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }
};

class StaticPressureAdvectionCoef : public StateVariableVecCoef
{
private:
   double a_;

   StateVariableCoef &ni_;
   StateVariableCoef &vi_;

   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   StaticPressureAdvectionCoef(StateVariableCoef &ni,
                               StateVariableCoef &vi,
                               double a,
                               VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        a_(a),
        ni_(ni),
        vi_(vi),
        B3_(&B3Coef), B_(3)
   {}

   StaticPressureAdvectionCoef(const StaticPressureAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        a_(other.a_),
        ni_(other.ni_),
        vi_(other.vi_),
        B3_(other.B3_),
        B_(3)
   {}

   virtual StaticPressureAdvectionCoef * Clone() const
   {
      return new StaticPressureAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni = ni_.Eval(T, ip);
      double vi = vi_.Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = a_ * (ni * vi * B_[0] / Bmag);
      V[1] = a_ * (ni * vi * B_[1] / Bmag);
   }
};

class TotalEnergyCoef : public StateVariableCoef
{
private:
   FieldType fieldType_;
   int z_i_;
   double m_kg_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;
   StateVariableCoef &TCoef_;

public:
   TotalEnergyCoef(double m_kg,
                   StateVariableCoef &niCoef,
                   StateVariableCoef &viCoef,
                   StateVariableCoef &TCoef)
      : fieldType_(ION_TEMPERATURE),
        z_i_(1), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef)
   {}

   TotalEnergyCoef(int z_i,
                   double m_kg,
                   StateVariableCoef &niCoef,
                   StateVariableCoef &viCoef,
                   StateVariableCoef &TCoef)
      : fieldType_(ELECTRON_TEMPERATURE),
        z_i_(z_i), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef)
   {}

   TotalEnergyCoef(const TotalEnergyCoef &other)
      : niCoef_(other.niCoef_),
        viCoef_(other.viCoef_),
        TCoef_(other.TCoef_)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      z_i_       = other.z_i_;
      m_kg_      = other.m_kg_;
   }

   virtual TotalEnergyCoef * Clone() const
   {
      return new TotalEnergyCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY || deriv == fieldType_);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni  = niCoef_.Eval(T, ip);
      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      return 0.5 * z_i_ * ni * (3.0 * T_J + m_kg_ * vi * vi);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      return 0.5 * z_i_ * (3.0 * T_J + m_kg_ * vi * vi);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);

      return z_i_ * ni * m_kg_ * vi;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ION_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ELECTRON_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }
};

class KineticEnergyCoef : public StateVariableCoef
{
private:
   int z_i_;
   double m_kg_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;

public:
   KineticEnergyCoef(double m_kg,
                     StateVariableCoef &niCoef,
                     StateVariableCoef &viCoef)
      : z_i_(1), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef) {}

   KineticEnergyCoef(int z_i,
                     double m_kg,
                     StateVariableCoef &niCoef,
                     StateVariableCoef &viCoef)
      : z_i_(z_i), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef) {}

   KineticEnergyCoef(const KineticEnergyCoef &other)
      : niCoef_(other.niCoef_),
        viCoef_(other.viCoef_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_kg_      = other.m_kg_;
   }

   virtual KineticEnergyCoef * Clone() const
   {
      return new KineticEnergyCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);

      return 0.5 * z_i_ * ni * m_kg_ * vi * vi;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);

      return 0.5 * z_i_ * m_kg_ * vi * vi;
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);

      return z_i_ * ni * m_kg_ * vi;
   }
};

class TotalEnergyAdvectionCoef : public StateVariableVecCoef
{
private:
   FieldType fieldType_;
   int z_i_;
   double m_kg_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;
   StateVariableCoef &TCoef_;

   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   TotalEnergyAdvectionCoef(double m_kg,
                            StateVariableCoef &niCoef,
                            StateVariableCoef &viCoef,
                            StateVariableCoef &TCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        fieldType_(ION_TEMPERATURE),
        z_i_(1), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef),
        B3_(&B3Coef), B_(3)
   {}

   TotalEnergyAdvectionCoef(int z_i,
                            double m_kg,
                            StateVariableCoef &niCoef,
                            StateVariableCoef &viCoef,
                            StateVariableCoef &TCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        fieldType_(ELECTRON_TEMPERATURE),
        z_i_(z_i), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef),
        B3_(&B3Coef), B_(3)
   {}

   TotalEnergyAdvectionCoef(const TotalEnergyAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        niCoef_(other.niCoef_),
        viCoef_(other.viCoef_),
        TCoef_(other.TCoef_),
        B3_(other.B3_),
        B_(3)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      z_i_       = other.z_i_;
      m_kg_      = other.m_kg_;
   }

   virtual TotalEnergyAdvectionCoef * Clone() const
   {
      return new TotalEnergyAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY || deriv == fieldType_);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni  = niCoef_.Eval(T, ip);
      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
      V *= 0.5 * z_i_ * ni * vi * (5.0 * T_J + m_kg_ * vi * vi);
   }

   void Eval_dNi(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
      V *= 0.5 * z_i_ * vi * (5.0 * T_J + m_kg_ * vi * vi);
   }

   void Eval_dVi(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni  = niCoef_.Eval(T, ip);
      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
      V *= 0.5 * z_i_ * ni * (5.0 * T_J + 3.0 * m_kg_ * vi * vi);
   }

   void Eval_dTi(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      if (fieldType_ == ION_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);
         double vi = viCoef_.Eval(T, ip);

         B3_->Eval(B_, T, ip);

         double Bmag2 = B_ * B_;
         double Bmag = sqrt(Bmag2);

         V[0] = B_[0] / Bmag;
         V[1] = B_[1] / Bmag;
         V *= 2.5 * z_i_ * ni * vi * J_per_eV_;
      }
      else
      {
         V = 0.0;
      }
   }

   void Eval_dTe(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      if (fieldType_ == ELECTRON_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);
         double vi = viCoef_.Eval(T, ip);

         B3_->Eval(B_, T, ip);

         double Bmag2 = B_ * B_;
         double Bmag = sqrt(Bmag2);

         V[0] = B_[0] / Bmag;
         V[1] = B_[1] / Bmag;
         V *= 2.5 * z_i_ * ni * vi * J_per_eV_;
      }
      else
      {
         V = 0.0;
      }
   }
};

class IonElectronHeatExchangeCoef : public StateVariableCoef
{
private:
   int z_i_;
   double m_i_kg_;
   StateVariableCoef &lnLambda_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &TiCoef_;
   StateVariableCoef &TeCoef_;

public:
   IonElectronHeatExchangeCoef(int z_i, double m_i_kg,
                               StateVariableCoef &lnLambda,
                               StateVariableCoef &niCoef,
                               StateVariableCoef &TiCoef,
                               StateVariableCoef &TeCoef)
      : z_i_(z_i), m_i_kg_(m_i_kg), lnLambda_(lnLambda),
        niCoef_(niCoef), TiCoef_(TiCoef), TeCoef_(TeCoef) {}

   IonElectronHeatExchangeCoef(const IonElectronHeatExchangeCoef &other)
      : lnLambda_(other.lnLambda_),
        niCoef_(other.niCoef_),
        TiCoef_(other.TiCoef_),
        TeCoef_(other.TeCoef_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_i_kg_    = other.m_i_kg_;
   }

   virtual IonElectronHeatExchangeCoef * Clone() const
   {
      return new IonElectronHeatExchangeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval_Func(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_Func: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);

      return 3.0 * (Te - Ti) * J_per_eV_ * me_kg_ * z_i_ * ni / (m_i_kg_ * tau);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval_Func(T, ip);
      double dl = lnLambda_.Eval_dNi(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_dNi: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);
      double dtau_dn = dtau_e_dni(Te, z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, z_i_, ni, lnLambda);

      double a = 3.0 * (Te - Ti) * J_per_eV_ * me_kg_ * z_i_ / m_i_kg_;

      return a * (tau - ni * (dtau_dn + dtau_dl * dl)) / (tau * tau);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_dTi: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);

      return -3.0 * J_per_eV_ * me_kg_ * z_i_ * ni / (m_i_kg_ * tau);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval_Func(T, ip);
      double dl = lnLambda_.Eval_dTe(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_dTe: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);
      double dtau_dT = dtau_e_dTe(Te, z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, z_i_, ni, lnLambda);

      double a = 3.0 * me_kg_ * z_i_ * ni * J_per_eV_ / m_i_kg_;

      return a * (tau - (Te - Ti) * (dtau_dT + dtau_dl * dl)) / (tau * tau);
   }

};

class IonThermalParaDiffusionCoef : public StateVariableCoef
{
private:
   // double lnLambda_;
   double z_i_;
   double m_i_kg_;

   Coefficient * lnLambda_;
   Coefficient * niCoef_;
   Coefficient * TiCoef_;

public:
   IonThermalParaDiffusionCoef(double ion_charge,
                               double ion_mass_kg,
                               Coefficient &lnLambda,
                               Coefficient &niCoef,
                               Coefficient &TiCoef)
      : StateVariableCoef(),
        z_i_(ion_charge),
        m_i_kg_(ion_mass_kg),
        lnLambda_(&lnLambda), niCoef_(&niCoef), TiCoef_(&TiCoef)
   {}

   IonThermalParaDiffusionCoef(const IonThermalParaDiffusionCoef &other)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_i_kg_    = other.m_i_kg_;
      lnLambda_  = other.lnLambda_;
      niCoef_    = other.niCoef_;
      TiCoef_    = other.TiCoef_;

   }

   virtual IonThermalParaDiffusionCoef * Clone() const
   {
      return new IonThermalParaDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni  = std::max(niCoef_->Eval(T, ip), 1.0);
      double Ti_eV = std::max(TiCoef_->Eval(T, ip), 1.0);
      // MFEM_VERIFY(ni > 0.0,
      //          "Ion density (" << ni << ") "
      //          "less than or equal to zero in "
      //          "IonThermalParaDiffusionCoef.");
      // MFEM_VERIFY(Ti_eV > 0.0,
      //          "Ion temperature (" << Ti_eV << ") "
      //          "less than or equal to zero in "
      //          "IonThermalParaDiffusionCoef.");

      double lnLambda = lnLambda_->Eval(T, ip);
      double tau = tau_i(m_i_kg_, z_i_, ni, Ti_eV, lnLambda);
      // std::cout << "Chi_e parallel: " << 3.16 * ne * Te * J_per_eV_ * tau / me_kg_
      // << ", n_e: " << ne << ", T_e: " << Te << std::endl;
      return 3.9 * Ti_eV * J_per_eV_ * tau / m_i_kg_;
   }

};

class IonEnergyIonizationCoef : public StateVariableCoef
{
private:
   double mn_;
   double mi_;
   Coefficient &vnCoef_;
   Coefficient &TnCoef_;
   Coefficient &SizCoef_;
   StateVariableCoef *Siz_sv_;

public:
   IonEnergyIonizationCoef(double m_n_kg, double m_i_kg,
                           Coefficient &vnCoef,
                           Coefficient &TnCoef,
                           Coefficient &SizCoef)
      : mn_(m_n_kg), mi_(m_i_kg),
        vnCoef_(vnCoef), TnCoef_(TnCoef), SizCoef_(SizCoef)
   {
      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   IonEnergyIonizationCoef(const IonEnergyIonizationCoef &other)
      : vnCoef_(other.vnCoef_), TnCoef_(other.TnCoef_),
        SizCoef_(other.SizCoef_)
   {
      derivType_ = other.derivType_;
      mn_        = other.mn_;
      mi_        = other.mi_;

      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   virtual IonEnergyIonizationCoef * Clone() const
   {
      return new IonEnergyIonizationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE || diz);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double Siz = SizCoef_.Eval(T, ip);

      return 0.5 * mi_ * Siz * (3.0 * Tn / mn_ + vn * vn);
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      if (!diz) { return 0.0; }

      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double dSiz = Siz_sv_->Eval_dFunc(deriv, T, ip);

      return 0.5 * mi_ * dSiz * (3.0 * Tn / mn_ + vn * vn);
   }
};

class IonEnergyRecombinationCoef : public StateVariableCoef
{
private:
   double mi_;
   Coefficient &viCoef_;
   Coefficient &SrcCoef_;
   StateVariableCoef *Src_sv_;

public:
   IonEnergyRecombinationCoef(double m_i_kg,
                              Coefficient &viCoef,
                              Coefficient &SrcCoef)
      : mi_(m_i_kg),
        viCoef_(viCoef), SrcCoef_(SrcCoef)
   {
      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   IonEnergyRecombinationCoef(const IonEnergyRecombinationCoef &other)
      : viCoef_(other.viCoef_), SrcCoef_(other.SrcCoef_)
   {
      derivType_ = other.derivType_;
      mi_        = other.mi_;

      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   virtual IonEnergyRecombinationCoef * Clone() const
   {
      return new IonEnergyRecombinationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_PARA_VELOCITY ||
              drc);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);
      double Src = SrcCoef_.Eval(T, ip);

      return 0.5 * mi_ * Src * vi * vi;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      if (!drc || deriv == ION_PARA_VELOCITY)
      { return 0.0; }

      double vi = viCoef_.Eval(T, ip);
      double dvi = (deriv == ION_PARA_VELOCITY) ? 1.0 : 0.0;
      double Src = SrcCoef_.Eval(T, ip);
      double dSrc = (Src_sv_) ? Src_sv_->Eval_dFunc(deriv, T, ip) : 0.0;

      return 0.5 * mi_ * vi * (dSrc * vi + 2.0 * Src * dvi);
   }
};

class IonEnergyChargeExchangeCoef : public StateVariableCoef
{
private:
   double mi_;
   Coefficient &vnCoef_;
   Coefficient &viCoef_;
   Coefficient &ScxCoef_;
   StateVariableCoef *Scx_sv_;

public:
   IonEnergyChargeExchangeCoef(double m_i_kg,
                               Coefficient &vnCoef,
                               Coefficient &viCoef,
                               Coefficient &ScxCoef)
      : mi_(m_i_kg),
        vnCoef_(vnCoef), viCoef_(viCoef), ScxCoef_(ScxCoef)
   {
      Scx_sv_ = dynamic_cast<StateVariableCoef*>(&ScxCoef_);
   }

   IonEnergyChargeExchangeCoef(const IonEnergyChargeExchangeCoef &other)
      : vnCoef_(other.vnCoef_), viCoef_(other.viCoef_), ScxCoef_(other.ScxCoef_)
   {
      derivType_ = other.derivType_;
      mi_        = other.mi_;

      Scx_sv_ = dynamic_cast<StateVariableCoef*>(&ScxCoef_);
   }

   virtual IonEnergyChargeExchangeCoef * Clone() const
   {
      return new IonEnergyChargeExchangeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dcx = (Scx_sv_) ? Scx_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_PARA_VELOCITY ||
              dcx );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double Scx = ScxCoef_.Eval(T, ip);

      return 0.5 * mi_ * Scx * (vn * vn - vi * vi);
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dcx = (Scx_sv_) ? Scx_sv_->NonTrivialValue(deriv) : false;
      if (!dcx || deriv == ION_PARA_VELOCITY)
      { return 0.0; }

      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double dvi = (deriv == ION_PARA_VELOCITY) ? 1.0 : 0.0;
      double Scx = ScxCoef_.Eval(T, ip);
      double dScx = (Scx_sv_) ? Scx_sv_->Eval_dFunc(deriv, T, ip) : 0.0;

      return 0.5 * mi_ * (dScx * (vn * vn - vi * vi) - 2.0 * Scx * vi * dvi);
   }
};

class ElectronThermalParaDiffusionCoef : public StateVariableCoef
{
private:
   double z_i_;

   Coefficient *lnLambda_;
   Coefficient *neCoef_;
   Coefficient *TeCoef_;

public:
   ElectronThermalParaDiffusionCoef(double ion_charge,
                                    Coefficient &lnLambda,
                                    Coefficient &neCoef,
                                    Coefficient &TeCoef,
                                    FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableCoef(deriv),
        z_i_(ion_charge), lnLambda_(&lnLambda),
        neCoef_(&neCoef), TeCoef_(&TeCoef)
   {}

   ElectronThermalParaDiffusionCoef(
      const ElectronThermalParaDiffusionCoef &other)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      lnLambda_  = other.lnLambda_;
      neCoef_    = other.neCoef_;
      TeCoef_    = other.TeCoef_;
   }

   virtual ElectronThermalParaDiffusionCoef * Clone() const
   {
      return new ElectronThermalParaDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ne    = std::max(neCoef_->Eval(T, ip), 1.0);
      double Te_eV = std::max(TeCoef_->Eval(T, ip), 1.0);
      // MFEM_VERIFY(ne > 0.0,
      //          "ElectronThermalParaDiffusionCoef::Eval_Func: "
      //          "Electron density (" << ne << ") "
      //          "less than or equal to zero.");
      // MFEM_VERIFY(Te_eV > 0.0,
      //          "ElectronThermalParaDiffusionCoef::Eval_Func: "
      //          "Electron temperature (" << Te_eV << ") "
      //          "less than or equal to zero.");

      double lnLambda = lnLambda_->Eval(T, ip);
      double tau = tau_e(Te_eV, z_i_, ne, lnLambda);
      // std::cout << "Chi_e parallel: " << 3.16 * ne * Te * J_per_eV_ * tau / me_kg_
      // << ", n_e: " << ne << ", T_e: " << Te << std::endl;
      return 3.16 * Te_eV * J_per_eV_ * tau / me_kg_;
   }
   /*
   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ne = neCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);
      MFEM_VERIFY(ne > 0.0,
                  "Electron density (" << ne << ") "
                  "less than or equal to zero in "
                  "ElectronThermalParaDiffusionCoef.");
      MFEM_VERIFY(Te > 0.0,
                  "Electron temperature (" << Te << ") "
                  "less than or equal to zero in "
                  "ElectronThermalParaDiffusionCoef.");

      double tau = tau_e(Te, z_i_, ne, 17.0);
      double dtau = dtau_e_dT(Te, z_i_, ne, 17.0);
      // std::cout << "Chi_e parallel: " << 3.16 * ne * Te * J_per_eV_ * tau / me_kg_
      // << ", n_e: " << ne << ", T_e: " << Te << std::endl;
      return 3.16 * ne * J_per_eV_ * (tau + Te * dtau)/ me_kg_;
   }
   */
};

class ElectronEnergyIonizationCoef : public StateVariableCoef
{
private:
   double mn_;
   Coefficient &phiIZCoef_;
   Coefficient &vnCoef_;
   Coefficient &TnCoef_;
   Coefficient &SizCoef_;
   StateVariableCoef *Siz_sv_;

public:
   ElectronEnergyIonizationCoef(double m_n_kg,
                                Coefficient &phiIZCoef,
                                Coefficient &vnCoef,
                                Coefficient &TnCoef,
                                Coefficient &SizCoef)
      : mn_(m_n_kg), phiIZCoef_(phiIZCoef), vnCoef_(vnCoef),
        TnCoef_(TnCoef), SizCoef_(SizCoef)
   {
      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   ElectronEnergyIonizationCoef(const ElectronEnergyIonizationCoef &other)
      : phiIZCoef_(other.phiIZCoef_), vnCoef_(other.vnCoef_),
        TnCoef_(other.TnCoef_), SizCoef_(other.SizCoef_)
   {
      derivType_ = other.derivType_;
      mn_        = other.mn_;

      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   virtual ElectronEnergyIonizationCoef * Clone() const
   {
      return new ElectronEnergyIonizationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE || diz);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double phiIZ = phiIZCoef_.Eval(T, ip) * J_per_eV_;
      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double Siz = SizCoef_.Eval(T, ip);

      return Siz * (phiIZ - 0.5 * me_kg_ * (3.0 * Tn / mn_ + vn * vn));
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      if (!diz) { return 0.0; }

      double phiIZ = phiIZCoef_.Eval(T, ip) * J_per_eV_;
      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double dSiz = Siz_sv_->Eval_dFunc(deriv, T, ip);

      return dSiz * (phiIZ - 0.5 * me_kg_ * (3.0 * Tn / mn_ + vn * vn));
   }
};

class ElectronEnergyRecombinationCoef : public StateVariableCoef
{
private:
   Coefficient &viCoef_;
   Coefficient &SrcCoef_;
   StateVariableCoef *Src_sv_;

public:
   ElectronEnergyRecombinationCoef(Coefficient &viCoef,
                                   Coefficient &SrcCoef)
      : viCoef_(viCoef), SrcCoef_(SrcCoef)
   {
      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   ElectronEnergyRecombinationCoef(const ElectronEnergyRecombinationCoef &other)
      : viCoef_(other.viCoef_), SrcCoef_(other.SrcCoef_)
   {
      derivType_ = other.derivType_;

      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   virtual ElectronEnergyRecombinationCoef * Clone() const
   {
      return new ElectronEnergyRecombinationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_PARA_VELOCITY ||
              drc);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);
      double Src = SrcCoef_.Eval(T, ip);

      return 0.5 * me_kg_ * Src * vi * vi;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      if (!drc || deriv == ION_PARA_VELOCITY)
      { return 0.0; }

      double vi = viCoef_.Eval(T, ip);
      double dvi = (deriv == ION_PARA_VELOCITY) ? 1.0 : 0.0;
      double Src = SrcCoef_.Eval(T, ip);
      double dSrc = (Src_sv_) ? Src_sv_->Eval_dFunc(deriv, T, ip) : 0.0;

      return 0.5 * me_kg_ * vi * (dSrc * vi + 2.0 * Src * dvi);
   }
};

class VectorXYCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * V_;
   Vector v3_;

public:
   VectorXYCoefficient(VectorCoefficient &V)
      : VectorCoefficient(2), V_(&V), v3_(3) {}

   void Eval(Vector &v,
             ElementTransformation &T,
             const IntegrationPoint &ip)
   { v.SetSize(2); V_->Eval(v3_, T, ip); v[0] = v3_[0]; v[1] = v3_[1]; }
};

class VectorZCoefficient : public Coefficient
{
private:
   VectorCoefficient * V_;
   Vector v3_;

public:
   VectorZCoefficient(VectorCoefficient &V)
      : V_(&V), v3_(3) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   { V_->Eval(v3_, T, ip); return v3_[2]; }
};

class UnitVectorXYCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * V_;
   Vector v3_;

public:
   UnitVectorXYCoefficient(VectorCoefficient &V)
      : VectorCoefficient(2), V_(&V), v3_(3) {}

   void Eval(Vector &v,
             ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      v.SetSize(2); V_->Eval(v3_, T, ip);
      double vmag = sqrt(v3_ * v3_);
      v[0] = v3_[0] / vmag; v[1] = v3_[1] / vmag;
   }
};

class Aniso2DDiffusionCoef : public StateVariableMatCoef
{
private:
   StateVariableCoef * Para_;
   StateVariableCoef * Perp_;
   VectorCoefficient * B3_;

   mutable Vector B_;

   void FormMat(double para, double perp,
                ElementTransformation &T,
                const IntegrationPoint &ip,
                DenseMatrix & M)
   {
      M.SetSize(2);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;

      M(0,0) = (B_[0] * B_[0] * para +
                (B_[1] * B_[1] + B_[2] * B_[2]) * perp ) / Bmag2;
      M(0,1) = B_[0] * B_[1] * (para - perp) / Bmag2;
      M(1,0) = M(0,1);
      M(1,1) = (B_[1] * B_[1] * para +
                (B_[0] * B_[0] + B_[2] * B_[2]) * perp ) / Bmag2;
   }

public:
   Aniso2DDiffusionCoef(StateVariableCoef *ParaCoef,
                        StateVariableCoef *PerpCoef,
                        VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2),
        Para_(ParaCoef), Perp_(PerpCoef),
        B3_(&B3Coef), B_(3) {}

   Aniso2DDiffusionCoef(bool para, StateVariableCoef &Coef,
                        VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2),
        Para_(para ? &Coef : NULL), Perp_(para ? NULL : &Coef),
        B3_(&B3Coef), B_(3) {}

   Aniso2DDiffusionCoef(const Aniso2DDiffusionCoef &other)
      : StateVariableMatCoef(2),
        Para_(other.Para_),
        Perp_(other.Perp_),
        B3_(other.B3_),
        B_(3) {}

   virtual Aniso2DDiffusionCoef * Clone() const
   {
      return new Aniso2DDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool ntv = deriv == INVALID_FIELD_TYPE;
      if (Para_)
      {
         ntv |= Para_->NonTrivialValue(deriv);
      }
      if (Perp_)
      {
         ntv |= Perp_->NonTrivialValue(deriv);
      }
      return ntv;
   }

   void Eval_Func(DenseMatrix & M,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dNn(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dNn(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dNn(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dNi(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dNi(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dNi(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dVi(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dVi(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dVi(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dTi(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dTi(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dTi(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dTe(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dTe(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dTe(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }
};

class NegGradPressureCoefficient : public StateVariableCoef
{
private:
   double zi_; // Stored as a double to avoid type casting in Eval methods
   double dt_;

   GridFunctionCoefficient ni0_;
   GridFunctionCoefficient Ti0_;
   GridFunctionCoefficient Te0_;

   GridFunctionCoefficient dni0_;
   GridFunctionCoefficient dTi0_;
   GridFunctionCoefficient dTe0_;

   GradientGridFunctionCoefficient grad_ni0_;
   GradientGridFunctionCoefficient grad_Ti0_;
   GradientGridFunctionCoefficient grad_Te0_;

   GradientGridFunctionCoefficient grad_dni0_;
   GradientGridFunctionCoefficient grad_dTi0_;
   GradientGridFunctionCoefficient grad_dTe0_;

   VectorCoefficient * B3_;

   mutable Vector gni0_;
   mutable Vector gTi0_;
   mutable Vector gTe0_;

   mutable Vector gdni0_;
   mutable Vector gdTi0_;
   mutable Vector gdTe0_;

   mutable Vector gni1_;
   mutable Vector gTi1_;
   mutable Vector gTe1_;

   mutable Vector B_;

public:
   NegGradPressureCoefficient(ParGridFunctionArray &yGF,
                              ParGridFunctionArray &kGF,
                              int zi, VectorCoefficient & B3Coef)
      : zi_((double)zi), dt_(0.0),
        ni0_(yGF[ION_DENSITY]),
        Ti0_(yGF[ION_TEMPERATURE]),
        Te0_(yGF[ELECTRON_TEMPERATURE]),
        dni0_(kGF[ION_DENSITY]),
        dTi0_(kGF[ION_TEMPERATURE]),
        dTe0_(kGF[ELECTRON_TEMPERATURE]),
        grad_ni0_(yGF[ION_DENSITY]),
        grad_Ti0_(yGF[ION_TEMPERATURE]),
        grad_Te0_(yGF[ELECTRON_TEMPERATURE]),
        grad_dni0_(kGF[ION_DENSITY]),
        grad_dTi0_(kGF[ION_TEMPERATURE]),
        grad_dTe0_(kGF[ELECTRON_TEMPERATURE]),
        B3_(&B3Coef), B_(3) {}

   NegGradPressureCoefficient(const NegGradPressureCoefficient & other)
      : ni0_(other.ni0_),
        Ti0_(other.Ti0_),
        Te0_(other.Te0_),
        dni0_(other.dni0_),
        dTi0_(other.dTi0_),
        dTe0_(other.dTe0_),
        grad_ni0_(ni0_.GetGridFunction()),
        grad_Ti0_(Ti0_.GetGridFunction()),
        grad_Te0_(Te0_.GetGridFunction()),
        grad_dni0_(dni0_.GetGridFunction()),
        grad_dTi0_(dTi0_.GetGridFunction()),
        grad_dTe0_(dTe0_.GetGridFunction()),
        B_(3)
   {
      derivType_ = other.derivType_;
      zi_ = other.zi_;
      dt_ = other.dt_;
      B3_ = other.B3_;
   }

   virtual NegGradPressureCoefficient * Clone() const
   {
      return new NegGradPressureCoefficient(*this);
   }

   void SetTimeStep(double dt) { dt_ = dt; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);
      double Ti0 = Ti0_.Eval(T, ip);
      double Te0 = Te0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);
      double dTi0 = dTi0_.Eval(T, ip);
      double dTe0 = dTe0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;
      double Ti1 = Ti0 + dt_ * dTi0;
      double Te1 = Te0 + dt_ * dTe0;

      grad_ni0_.Eval(gni0_, T, ip);
      grad_Ti0_.Eval(gTi0_, T, ip);
      grad_Te0_.Eval(gTe0_, T, ip);

      grad_dni0_.Eval(gdni0_, T, ip);
      grad_dTi0_.Eval(gdTi0_, T, ip);
      grad_dTe0_.Eval(gdTe0_, T, ip);

      gni1_.SetSize(gni0_.Size());
      gTi1_.SetSize(gTi0_.Size());
      gTe1_.SetSize(gTe0_.Size());

      add(gni0_, dt_, gdni0_, gni1_);
      add(gTi0_, dt_, gdTi0_, gTi1_);
      add(gTe0_, dt_, gdTe0_, gTe1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * ((zi_ * Te1 + Ti1) * (B_[0] * gni1_[0] + B_[1] * gni1_[1]) +
                           (zi_ * (B_[0] * gTe1_[0] + B_[1] * gTe1_[1]) +
                            (B_[0] * gTi1_[0] + B_[1] * gTi1_[1])) * ni1) / Bmag;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      grad_Ti0_.Eval(gTi0_, T, ip);
      grad_Te0_.Eval(gTe0_, T, ip);

      grad_dTi0_.Eval(gdTi0_, T, ip);
      grad_dTe0_.Eval(gdTe0_, T, ip);

      gTi1_.SetSize(gTi0_.Size());
      gTe1_.SetSize(gTe0_.Size());

      add(gTi0_, dt_, gdTi0_, gTi1_);
      add(gTe0_, dt_, gdTe0_, gTe1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * (dt_ * zi_ * (B_[0] * gTe1_[0] + B_[1] * gTe1_[1]) +
                           (B_[0] * gTi1_[0] + B_[1] * gTi1_[1])) / Bmag;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      grad_ni0_.Eval(gni0_, T, ip);

      grad_dni0_.Eval(gdni0_, T, ip);

      gni1_.SetSize(gni0_.Size());

      add(gni0_, dt_, gdni0_, gni1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * dt_ * (B_[0] * gni1_[0] + B_[1] * gni1_[1]) / Bmag;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      grad_ni0_.Eval(gni0_, T, ip);

      grad_dni0_.Eval(gdni0_, T, ip);

      gni1_.SetSize(gni0_.Size());

      add(gni0_, dt_, gdni0_, gni1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * dt_ * zi_ * (B_[0] * gni1_[0] + B_[1] * gni1_[1]) / Bmag;
   }
};

class NegBPressureCoefficient : public StateVariableVecCoef
{
private:
   double zi_; // Stored as a double to avoid type casting in Eval methods
   double dt_;

   GridFunctionCoefficient ni0_;
   GridFunctionCoefficient Ti0_;
   GridFunctionCoefficient Te0_;

   GridFunctionCoefficient dni0_;
   GridFunctionCoefficient dTi0_;
   GridFunctionCoefficient dTe0_;

   VectorCoefficient * B3_;
   /*
    mutable Vector gni0_;
    mutable Vector gTi0_;
    mutable Vector gTe0_;

    mutable Vector gdni0_;
    mutable Vector gdTi0_;
    mutable Vector gdTe0_;

    mutable Vector gni1_;
    mutable Vector gTi1_;
    mutable Vector gTe1_;
   */

   mutable Vector B_;

public:
   NegBPressureCoefficient(ParGridFunctionArray &yGF,
                           ParGridFunctionArray &kGF,
                           int zi, VectorCoefficient & B3Coef)
      : StateVariableVecCoef(2),
        zi_((double)zi), dt_(0.0),
        ni0_(yGF[ION_DENSITY]),
        Ti0_(yGF[ION_TEMPERATURE]),
        Te0_(yGF[ELECTRON_TEMPERATURE]),
        dni0_(kGF[ION_DENSITY]),
        dTi0_(kGF[ION_TEMPERATURE]),
        dTe0_(kGF[ELECTRON_TEMPERATURE]),
        B3_(&B3Coef), B_(3) {}

   NegBPressureCoefficient(const NegBPressureCoefficient & other)
      : StateVariableVecCoef(2),
        ni0_(other.ni0_),
        Ti0_(other.Ti0_),
        Te0_(other.Te0_),
        dni0_(other.dni0_),
        dTi0_(other.dTi0_),
        dTe0_(other.dTe0_),
        B_(3)
   {
      derivType_ = other.derivType_;
      zi_ = other.zi_;
      dt_ = other.dt_;
      B3_ = other.B3_;
   }

   virtual NegBPressureCoefficient * Clone() const
   {
      return new NegBPressureCoefficient(*this);
   }

   void SetTimeStep(double dt) { dt_ = dt; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   void Eval_Func(Vector &V, ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);
      double Ti0 = Ti0_.Eval(T, ip);
      double Te0 = Te0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);
      double dTi0 = dTi0_.Eval(T, ip);
      double dTe0 = dTe0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;
      double Ti1 = Ti0 + dt_ * dTi0;
      double Te1 = Te0 + dt_ * dTe0;

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = B_[0];
      V[1] = B_[1];

      V *= -J_per_eV_ * ni1 * (zi_ * Te1 + Ti1) / Bmag;
   }

   void Eval_dNi(Vector &V, ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double Ti0 = Ti0_.Eval(T, ip);
      double Te0 = Te0_.Eval(T, ip);

      double dTi0 = dTi0_.Eval(T, ip);
      double dTe0 = dTe0_.Eval(T, ip);

      double Ti1 = Ti0 + dt_ * dTi0;
      double Te1 = Te0 + dt_ * dTe0;

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = B_[0];
      V[1] = B_[1];

      V *= -J_per_eV_ * (zi_ * Te1 + Ti1) / Bmag;
   }

   void Eval_dTi(Vector &V, ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = B_[0];
      V[1] = B_[1];

      V *= -J_per_eV_ * ni1 / Bmag;
   }

   void Eval_dTe(Vector &V, ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = B_[0];
      V[1] = B_[1];

      V *= -J_per_eV_ * ni1 * zi_ / Bmag;
   }
};

class NegPressureCoefficient : public StateVariableCoef
{
private:
   double zi_; // Stored as a double to avoid type casting in Eval methods
   double dt_;

   GridFunctionCoefficient ni0_;
   GridFunctionCoefficient Ti0_;
   GridFunctionCoefficient Te0_;

   GridFunctionCoefficient dni0_;
   GridFunctionCoefficient dTi0_;
   GridFunctionCoefficient dTe0_;

public:
   NegPressureCoefficient(ParGridFunctionArray &yGF,
                          ParGridFunctionArray &kGF,
                          int zi)
      : zi_((double)zi), dt_(0.0),
        ni0_(yGF[ION_DENSITY]),
        Ti0_(yGF[ION_TEMPERATURE]),
        Te0_(yGF[ELECTRON_TEMPERATURE]),
        dni0_(kGF[ION_DENSITY]),
        dTi0_(kGF[ION_TEMPERATURE]),
        dTe0_(kGF[ELECTRON_TEMPERATURE])
   {}

   NegPressureCoefficient(const NegPressureCoefficient & other)
      : ni0_(other.ni0_),
        Ti0_(other.Ti0_),
        Te0_(other.Te0_),
        dni0_(other.dni0_),
        dTi0_(other.dTi0_),
        dTe0_(other.dTe0_)
   {
      derivType_ = other.derivType_;
      zi_ = other.zi_;
      dt_ = other.dt_;
   }

   virtual NegPressureCoefficient * Clone() const
   {
      return new NegPressureCoefficient(*this);
   }

   void SetTimeStep(double dt) { dt_ = dt; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);
      double Ti0 = Ti0_.Eval(T, ip);
      double Te0 = Te0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);
      double dTi0 = dTi0_.Eval(T, ip);
      double dTe0 = dTe0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;
      double Ti1 = Ti0 + dt_ * dTi0;
      double Te1 = Te0 + dt_ * dTe0;

      return -J_per_eV_ * ni1 * (zi_ * Te1 + Ti1);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti0 = Ti0_.Eval(T, ip);
      double Te0 = Te0_.Eval(T, ip);

      double dTi0 = dTi0_.Eval(T, ip);
      double dTe0 = dTe0_.Eval(T, ip);

      double Ti1 = Ti0 + dt_ * dTi0;
      double Te1 = Te0 + dt_ * dTe0;

      return -J_per_eV_ * (zi_ * Te1 + Ti1);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;

      return -J_per_eV_ * ni1;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;

      return -J_per_eV_ * ni1 * zi_;
   }
};

class TransportCoefFactory : public common::CoefFactory
{
public:
   TransportCoefFactory() {}
   TransportCoefFactory(const std::vector<std::string> & names,
                        ParGridFunctionArray & pgfa);

   using common::CoefFactory::GetScalarCoef;
   using common::CoefFactory::GetVectorCoef;
   using common::CoefFactory::GetMatrixCoef;

   Coefficient * GetScalarCoef(std::string &name, std::istream &input);
   VectorCoefficient * GetVectorCoef(std::string &name, std::istream &input);
};

} // namespace transport

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_TRANSPORT_COEFS
