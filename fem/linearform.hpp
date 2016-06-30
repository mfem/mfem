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

#ifndef MFEM_LINEARFORM
#define MFEM_LINEARFORM

#include "../config/config.hpp"
#include "lininteg.hpp"
#include "fespace.hpp"

namespace mfem
{

/// Class for linear form - Vector with associated FE space and LFIntegrators.
class LinearForm : public Vector
{
private:
   /// FE space on which LF lives.
   FiniteElementSpace * fes;

   /// Set of Domain Integrators to be applied.
   Array<LinearFormIntegrator*> dlfi;

   /// Set of Boundary Integrators to be applied.
   Array<LinearFormIntegrator*> blfi;

   /// Set of Boundary Face Integrators to be applied.
   Array<LinearFormIntegrator*> flfi;

public:
   /// Creates linear form associated with FE space *f.
   LinearForm (FiniteElementSpace * f) : Vector (f -> GetVSize())
   { fes = f; }

   LinearForm() { fes = NULL; }

   FiniteElementSpace * GetFES() { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator (LinearFormIntegrator * lfi);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   void Update() { SetSize(fes->GetVSize()); }

   void Update(FiniteElementSpace *f) { fes = f; SetSize(f->GetVSize()); }

   void Update(FiniteElementSpace *f, Vector &v, int v_offset);

   /// Destroys linear form.
   ~LinearForm();
};

}

#endif
