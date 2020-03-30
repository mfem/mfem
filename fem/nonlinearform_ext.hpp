// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

class NonlinearFormExtension : public Operator
{
protected:
   NonlinearForm *n; ///< Not owned
public:
   NonlinearFormExtension(NonlinearForm *form);
   virtual void AssemblePA() = 0;
};

/// Data and methods for partially-assembled nonlinear forms
class PANonlinearFormExtension : public NonlinearFormExtension
{
protected:
   const FiniteElementSpace &fes; // Not owned
   mutable Vector localX, localY;
   const Operator *elem_restrict_lex; // Not owned
public:
   PANonlinearFormExtension(NonlinearForm*);
   void AssemblePA();
   void Mult(const Vector &x, Vector &y) const;
};
}
#endif // NONLINEARFORM_EXT_HPP
