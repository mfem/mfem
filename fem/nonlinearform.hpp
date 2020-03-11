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

#ifndef MFEM_NONLINEARFORM
#define MFEM_NONLINEARFORM

#include "../config/config.hpp"
#include "nonlininteg.hpp"
#include "nonlinearform_ext.hpp"
#include "bilinearform.hpp"
#include "gridfunc.hpp"

namespace mfem
{

class NonlinearForm : public Operator
{
protected:
   /// The assembly level.
   AssemblyLevel assembly;

   /** Extension for supporting Partial Assembly (PA) or
       Matrix Free assembly (MF). */
   NonlinearFormExtension *ext;

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
      : Operator(f->GetTrueVSize()), assembly(AssemblyLevel::NONE),
        ext(NULL), fes(f), Grad(NULL), cGrad(NULL),
        sequence(f->GetSequence()), P(f->GetProlongationMatrix()),
        cP(dynamic_cast<const SparseMatrix*>(P))
   { }

   /// Set the desired assembly level. The default is AssemblyLevel::NONE.
   /** This method must be called before assembly. */
   void SetAssemblyLevel(AssemblyLevel assembly_level);

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(NonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Access all integrators added with AddDomainIntegrator().
   Array<NonlinearFormIntegrator*> *GetDNFI() { return &dnfi; }

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

   /// Setup the NonlinearForm
   virtual void Setup();

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const { return P; }
   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return fes->GetRestrictionMatrix(); }

   /** @brief Destroy the NoninearForm including the owned
       NonlinearFormIntegrator%s and gradient Operator. */
   virtual ~NonlinearForm();
};


/** @brief A class representing a general block nonlinear operator defined on
    the Cartesian product of multiple FiniteElementSpace%s. */
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

   // Essential vdofs: one list of vdofs for each space in 'fes'
   Array<Array<int> *> ess_vdofs;

   /// Specialized version of GetEnergy() for BlockVectors
   double GetEnergyBlocked(const BlockVector &bx) const;

   /// Specialized version of Mult() for BlockVector%s
   void MultBlocked(const BlockVector &bx, BlockVector &by) const;

   /// Specialized version of GetGradient() for BlockVector
   Operator &GetGradientBlocked(const BlockVector &bx) const;

public:
   /// Construct an empty BlockNonlinearForm. Initialize with SetSpaces().
   BlockNonlinearForm();

   /// Construct a BlockNonlinearForm on the given set of FiniteElementSpace%s.
   BlockNonlinearForm(Array<FiniteElementSpace *> &f);

   /// Return the @a k-th FE space of the BlockNonlinearForm.
   FiniteElementSpace *FESpace(int k) { return fes[k]; }
   /// Return the @a k-th FE space of the BlockNonlinearForm (const version).
   const FiniteElementSpace *FESpace(int k) const { return fes[k]; }

   /// (Re)initialize the BlockNonlinearForm.
   /** After a call to SetSpaces(), the essential b.c. must be set again. */
   void SetSpaces(Array<FiniteElementSpace *> &f);

   /// Return the regular dof offsets.
   const Array<int> &GetBlockOffsets() const { return block_offsets; }
   /// Return the true-dof offsets.
   const Array<int> &GetBlockTrueOffsets() const { return block_trueOffsets; }

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

   virtual double GetEnergy(const Vector &x) const;

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual Operator &GetGradient(const Vector &x) const;

   /// Destructor.
   virtual ~BlockNonlinearForm();
};


}

#endif
