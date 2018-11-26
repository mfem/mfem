// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM4_LINEARSYSTEM
#define MFEM4_LINEARSYSTEM

#include "fem/linearform.hpp"
#include "mfem4/fem/bilinearform.hpp"


namespace mfem4
{

using namespace mfem;

///
class LinearSystem
{
public:
   LinearSystem(const mfem4::BilinearForm &a, const LinearForm &b);

   LinearSystem(const mfem4::BilinearForm *a, const LinearForm *b);


   void SetEssentialDofs();

   void SetOperatorType();

   void EnableStaticCondensation(bool enable = true);


   void Assemble(bool partial = false);

   void PartialAssemble() { Assemble(true); }

   void Solve(const Solver &solver, GridFunction &x);

   void Solve(const Solver &precond, const Solver &solver, GridFunction &x);


   const Operator& GetMatrix() const;
   const Operator& GetOperator() const;
   const Vector& GetRHS() const;

   void RecoverGridFunction(const Vector &X, GridFunction &x);


protected:

};


} // namespace mfem4

#endif // MFEM4_LINEARSYSTEM
