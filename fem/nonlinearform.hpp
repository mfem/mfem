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

   /// Set of boundary face Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> bfnfi; // owned
   Array<Array<int>*>              bfnfi_marker; // not owned

   mutable SparseMatrix *Grad; // owned

   // A list of all essential vdofs
   Array<int> ess_vdofs;

public:
   NonlinearForm(FiniteElementSpace *f)
      : Operator(f->GetVSize()) { fes = f; Grad = NULL; }

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(NonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nfi,
                             Array<int> &bdr_marker)
   { bfnfi.Append(nfi); bfnfi_marker.Append(&bdr_marker); }

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


class BlockNonlinearForm : public Operator
{
protected:
   /// FE spaces on which the form lives.
   Array<FiniteElementSpace*> fes;

   /// Set of Domain Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> dnfi;

   /// Set of interior face Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> fnfi;

   /// Set of Boundary Face Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> bfnfi;
   Array<Array<int>*>           bfnfi_marker;

   /** Auxiliary block-vectors for wrapping input and output vectors or holding
       GridFunction-like block-vector data (e.g. in parallel). */
   mutable BlockVector xs, ys;

   mutable Array2D<SparseMatrix*> Grads;
   mutable BlockOperator *BlockGrad;

   // A list of the offsets
   Array<int> block_offsets;
   Array<int> block_trueOffsets;

   // A list of all essential vdofs
   Array<Array<int> *> ess_vdofs;

   /// Specialized version of Mult() for BlockVector%s
   void MultBlocked(const BlockVector &bx, BlockVector &by) const;

   /// Specialized version of GetGradient() for BlockVector
   Operator &GetGradientBlocked(const BlockVector &bx) const;

public:
   BlockNonlinearForm();

   BlockNonlinearForm(Array<FiniteElementSpace *> &f);

   /// After a call to SetSpaces(), the essential b.c. must be set again.
   void SetSpaces(Array<FiniteElementSpace *> &f);

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *nlfi,
                             Array<int> &bdr_marker);

   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual Operator &GetGradient(const Vector &x) const;

   virtual ~BlockNonlinearForm();
};


}

#endif
