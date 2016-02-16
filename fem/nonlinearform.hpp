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

#ifndef MFEM_NONLINEARFORM
#define MFEM_NONLINEARFORM

#include "../config/config.hpp"
#include "nonlininteg.hpp"
#include "fespace.hpp"

namespace mfem
{

class NonlinearForm : public Operator
{
protected:
   /// FE space on which the form lives.
   FiniteElementSpace *fes;

   /// Set of Domain Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> dfi;

   mutable SparseMatrix *Grad;

   // A list of all essential vdofs
   Array<int> ess_vdofs;

public:
   NonlinearForm(FiniteElementSpace *f)
      : Operator(f->GetVSize()) { fes = f; Grad = NULL; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(NonlinearFormIntegrator *nlfi)
   { dfi.Append(nlfi); }

   virtual void SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                               Vector *rhs = NULL);

   void SetEssentialVDofs(const Array<int> &ess_vdofs_list)
   {
      ess_vdofs_list.Copy(ess_vdofs); // ess_vdofs_list --> ess_vdofs
   }

   virtual double GetEnergy(const Vector &x) const;

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual Operator &GetGradient(const Vector &x) const;

   virtual ~NonlinearForm();
};

}

#endif
