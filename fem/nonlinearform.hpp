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
#include "gridfunc.hpp"

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

   mutable SparseMatrix *Grad, *cGrad; // owned

   /// A list of all essential true dofs
   Array<int> ess_tdof_list;

   /// Counter for updates propagated from the FiniteElementSpace.
   long sequence;

   /// Auxiliary Vector%s
   mutable Vector aux1, aux2;

   /// Pointer to the prolongation matrix of fes, may be NULL.
   const Operator *P; // not owned
   /// The result of dynamic-casting P to SparseMatrix pointer.
   const SparseMatrix *cP; // not owned

   bool Serial() const { return (!P || cP); }
   const Vector &Prolongate(const Vector &x) const;

public:
   /// Construct a NonlinearForm on the given FiniteElementSpace, @a f.
   /** As an Operator, the NonlinearForm has input and output size equal to the
       number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   NonlinearForm(FiniteElementSpace *f)
      : Operator(f->GetTrueVSize()), fes(f), Grad(NULL), cGrad(NULL),
        sequence(f->GetSequence()), P(f->GetProlongationMatrix()),
        cP(dynamic_cast<const SparseMatrix*>(P))
   { }

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

   /// Specify essential boundary conditions.
   /** This method calls FiniteElementSpace::GetEssentialTrueDofs() and stores
       the result internally for use by other methods. If the @a rhs pointer is
       not NULL, its essential true dofs will be set to zero. This makes it
       "compatible" with the output vectors from the Mult() method which also
       have zero entries at the essential true dofs. */
   void SetEssentialBC(const Array<int> &bdr_attr_is_ess, Vector *rhs = NULL);

   /// (DEPRECATED) Specify essential boundary conditions.
   /** @deprecated Use either SetEssentialBC() or SetEssentialTrueDofs(). */
   void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

   /// Specify essential boundary conditions.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list)
   { ess_tdof_list.Copy(this->ess_tdof_list); }

   /// Return a (read-only) list of all essential true dofs.
   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

   /// Compute the enery corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a "GridFunction size" vector, i.e. its size must
       be fes->GetVSize(). */
   double GetGridFunctionEnergy(const Vector &x) const;

   /// Compute the enery corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a true-dof vector. */
   virtual double GetEnergy(const Vector &x) const
   { return GetGridFunctionEnergy(Prolongate(x)); }

   /// Evaluate the action of the NonlinearForm.
   /** The input essential dofs in @a x will, generally, be non-zero. However,
       the output essential dofs in @a y will always be set to zero.

       Both the input and the output vectors, @a x and @a y, must be true-dof
       vectors, i.e. their size must be fes->GetTrueVSize(). */
   virtual void Mult(const Vector &x, Vector &y) const;

   /** @brief Compute the gradient Operator of the NonlinearForm corresponding
       to the state @a x. */
   /** Any previously specified essential boundary conditions will be
       automatically imposed on the gradient operator.

       The returned object is valid until the next call to this method or the
       destruction of this object.

       In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a true-dof vector. */
   virtual Operator &GetGradient(const Vector &x) const;

   /// Update the NonlinearForm to propagate updates of the associated FE space.
   /** After calling this method, the essential boundary conditions need to be
       set again. */
   virtual void Update();

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const { return P; }
   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return fes->GetRestrictionMatrix(); }

   /** @brief Destroy the NoninearForm including the owned
       NonlinearFormIntegrator%s and gradient Operator. */
   virtual ~NonlinearForm();
};

}

#endif
