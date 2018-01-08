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
   FiniteElementSpace *fes; // not owned

   /// Set of Domain Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> dnfi; // owned

   /// Set of interior face Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> fnfi; // owned

   /// Set of fespace Integrators to be assembled (added).
   Array<NonlinearFESpaceIntegrator*> fesi; // owned

   /// Internal additional vector for tensorized assembly
   mutable Vector *X;
   mutable Vector *Y;

   /// True if gather/scatter needed
   bool needs_gs;

   /// Set of boundary face Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> bfnfi; // owned
   Array<Array<int>*>              bfnfi_marker; // not owned

   mutable SparseMatrix *Grad; // owned

   // A list of all essential vdofs
   Array<int> ess_vdofs;

public:
   NonlinearForm(FiniteElementSpace *f);

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(NonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /// Adds new FESpace Integrator.
   void AddIntegrator(NonlinearFESpaceIntegrator *nlfi)
   { fesi.Append(nlfi); }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nfi,
                             Array<int> &bdr_marker)
   { bfnfi.Append(nfi); bfnfi_marker.Append(&bdr_marker); }

   Array<NonlinearFESpaceIntegrator*> *GetFESI() { return &fesi; }

   virtual void SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                               Vector *rhs = NULL);

   void SetEssentialVDofs(const Array<int> &ess_vdofs_list)
   {
      ess_vdofs_list.Copy(ess_vdofs); // ess_vdofs_list --> ess_vdofs
   }

   virtual double GetEnergy(const Vector &x) const;

   /// General version of the Mult method.
   void MultGeneral(const Vector &x, Vector &y) const;

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual Operator &GetGradient(const Vector &x) const;

   virtual ~NonlinearForm();
};

}

#endif
