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

#ifndef NONLINEARFORM_EXT_HPP
#define NONLINEARFORM_EXT_HPP

#include "../config/config.hpp"
#include "fespace.hpp"

namespace mfem
{

class NonlinearForm;
class NonlinearFormIntegrator;

/** @brief Class extending the NonlinearForm class to support the different
    AssemblyLevel%s. */
class NonlinearFormExtension : public Operator
{
protected:
   NonlinearForm *n; ///< Not owned
public:
   NonlinearFormExtension(NonlinearForm *form);
   virtual void Assemble() = 0;
   virtual Operator &GetGradient(const Vector&) const = 0;
   virtual double GetGridFunctionEnergy(const Vector &x) const = 0;
   virtual void AssembleGradientDiagonal(Vector &diag) const
   {
      MFEM_ABORT("Not implemented for this assembly level!");
   }
};

/// Data and methods for partially-assembled nonlinear forms
class PANonlinearFormExtension : public NonlinearFormExtension
{
private:
   class Gradient : public Operator
   {
   protected:
      const Operator *elemR;
      const FiniteElementSpace &fes;
      const Array<NonlinearFormIntegrator*> &dnfi;
      mutable Vector ge, xe, ye, ze;

   public:
      Gradient(const Vector &x, const PANonlinearFormExtension &ext);
      virtual void Mult(const Vector &x, Vector &y) const;

      /// Assumes that @a g is an ldof Vector.
      void ReInit(const Vector &g) { elemR->Mult(g, ge); }

      /// Assemble the diagonal of the gradient into the ldof Vector @a diag.
      virtual void AssembleDiagonal(Vector &diag) const;
   };

protected:
   mutable Vector xe, ye;
   mutable OperatorHandle Grad;
   const FiniteElementSpace &fes;  // Not owned
   const Array<NonlinearFormIntegrator*> &dnfi;
   const Operator *R; // Not owned

public:
   PANonlinearFormExtension(NonlinearForm*);
   void Assemble();
   void Mult(const Vector &x, Vector &y) const;
   Operator &GetGradient(const Vector &x) const;
   double GetGridFunctionEnergy(const Vector &x) const;
};

/// Data and methods for unassembled nonlinear forms
class MFNonlinearFormExtension : public NonlinearFormExtension
{
protected:
   const FiniteElementSpace &fes; // Not owned
   mutable Vector localX, localY;
   const Operator *elem_restrict_lex; // Not owned
public:
   MFNonlinearFormExtension(NonlinearForm*);
   void Assemble();
   void Mult(const Vector &x, Vector &y) const;
   virtual Operator &GetGradient(const Vector&) const
   {
      MFEM_ABORT("Not implemented for this assembly level!");
      return *new IdentityOperator(0);
   }
   virtual double GetGridFunctionEnergy(const Vector &x) const
   {
      MFEM_ABORT("Not implemented for this assembly level!");
      return 0.0;
   }
};

}
#endif // NONLINEARFORM_EXT_HPP
