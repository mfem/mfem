// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

   /// Extension for supporting different AssemblyLevel%s
   /** For nonlinear operators, the "matrix" assembly levels usually do not make
       sense, so only PARTIAL and NONE (matrix-free) are supported. */
   NonlinearFormExtension *ext; // owned

   /// FE space on which the form lives.
   FiniteElementSpace *fes; // not owned

   /** @brief Indicates the NonlinearFormIntegrator%s stored in #dnfi, #bnfi,
       #fnfi, and #bfnfi are not owned by this NonlinearForm. */
   bool extern_bfs;

   /// Set of Domain Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> dnfi; // owned
   Array<Array<int>*>              dnfi_marker; // not owned

   /// Set of Boundary Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> bnfi; // owned
   Array<Array<int>*> bnfi_marker; // not owned

   /// Set of interior face Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> fnfi; // owned

   /// Set of boundary face Integrators to be assembled (added).
   Array<NonlinearFormIntegrator*> bfnfi; // owned
   Array<Array<int>*>              bfnfi_marker; // not owned

   mutable SparseMatrix *Grad, *cGrad; // owned
   /// Gradient Operator when not assembled as a matrix.
   mutable OperatorHandle hGrad; // has internal ownership flag

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
      : Operator(f->GetTrueVSize()), assembly(AssemblyLevel::LEGACY),
        ext(NULL), fes(f), extern_bfs(0), Grad(NULL), cGrad(NULL),
        sequence(f->GetSequence()), P(f->GetProlongationMatrix()),
        cP(dynamic_cast<const SparseMatrix*>(P))
   { }

   /// Set the desired assembly level. The default is AssemblyLevel::LEGACY.
   /** For nonlinear operators, the "matrix" assembly levels usually do not make
       sense, so only LEGACY, NONE (matrix-free) and PARTIAL are supported.

       Currently, AssemblyLevel::LEGACY uses the standard nonlinear action
       methods like AssembleElementVector of the NonlinearFormIntegrator class
       which work only on CPU and do not utilize features such as fast
       tensor-product basis evaluations. In this mode, the gradient operator is
       constructed as a SparseMatrix (or, in parallel, format such as
       HypreParMatrix).

       When using AssemblyLevel::PARTIAL, the action is performed using methods
       like AddMultPA of the NonlinearFormIntegrator class which typically
       support both CPU and GPU backends and utilize features such as fast
       tensor-product basis evaluations. In this mode, the gradient operator
       also uses partial assembly with support for CPU and GPU backends.

       When using AssemblyLevel::NONE, the action is performed using methods
       like AddMultMF of the NonlinearFormIntegrator class which typically
       support both CPU and GPU backends and utilize features such as fast
       tensor-product basis evaluations. In this mode, the gradient operator
       is currently not supported.

       This method must be called before "assembly" with Setup(). */
   void SetAssemblyLevel(AssemblyLevel assembly_level);

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(NonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); dnfi_marker.Append(NULL); }

   /// Adds new Domain Integrator, restricted to specific attributes.
   void AddDomainIntegrator(NonlinearFormIntegrator *nlfi,
                            Array<int> &elem_marker)
   { dnfi.Append(nlfi); dnfi_marker.Append(&elem_marker); }

   /// Access all integrators added with AddDomainIntegrator().
   Array<NonlinearFormIntegrator*> *GetDNFI() { return &dnfi; }
   const Array<NonlinearFormIntegrator*> *GetDNFI() const { return &dnfi; }

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(NonlinearFormIntegrator *nlfi)
   { bnfi.Append(nlfi); bnfi_marker.Append(NULL); }

   /// Adds new Boundary Integrator, restricted to specific attributes.
   void AddBoundaryIntegrator(NonlinearFormIntegrator *nlfi,
                              Array<int> &elem_marker)
   { bnfi.Append(nlfi); bnfi_marker.Append(&elem_marker); }

   /// Access all integrators added with AddBoundaryIntegrator().
   Array<NonlinearFormIntegrator*> *GetBNFI() { return &bnfi; }
   const Array<NonlinearFormIntegrator*> *GetBNFI() const { return &bnfi; }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /** @brief Access all interior face integrators added with
       AddInteriorFaceIntegrator(). */
   const Array<NonlinearFormIntegrator*> &GetInteriorFaceIntegrators() const
   { return fnfi; }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nfi,
                             Array<int> &bdr_marker)
   { bfnfi.Append(nfi); bfnfi_marker.Append(&bdr_marker); }

   /** @brief Access all boundary face integrators added with
       AddBdrFaceIntegrator(). */
   const Array<NonlinearFormIntegrator*> &GetBdrFaceIntegrators() const
   { return bfnfi; }

   /// Specify essential boundary conditions.
   /** This method calls FiniteElementSpace::GetEssentialTrueDofs() and stores
       the result internally for use by other methods. If the @a rhs pointer is
       not NULL, its essential true dofs will be set to zero. This makes it
       "compatible" with the output vectors from the Mult() method which also
       have zero entries at the essential true dofs. */
   void SetEssentialBC(const Array<int> &bdr_attr_is_ess, Vector *rhs = NULL);

   /// Specify essential boundary conditions.
   /** Use either SetEssentialBC() or SetEssentialTrueDofs() if possible. */
   void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

   /// Specify essential boundary conditions.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list_)
   { ess_tdof_list_.Copy(this->ess_tdof_list); }

   /// Return a (read-only) list of all essential true dofs.
   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

   /// Compute the energy corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a "GridFunction size" vector, i.e. its size must
       be fes->GetVSize(). */
   real_t GetGridFunctionEnergy(const Vector &x) const;

   /// Compute the energy corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a true-dof vector. */
   virtual real_t GetEnergy(const Vector &x) const
   { return GetGridFunctionEnergy(Prolongate(x)); }

   /// Evaluate the action of the NonlinearForm.
   /** The input essential dofs in @a x will, generally, be non-zero. However,
       the output essential dofs in @a y will always be set to zero.

       Both the input and the output vectors, @a x and @a y, must be true-dof
       vectors, i.e. their size must be fes->GetTrueVSize(). */
   void Mult(const Vector &x, Vector &y) const override;

   /** @brief Compute the gradient Operator of the NonlinearForm corresponding
       to the state @a x. */
   /** Any previously specified essential boundary conditions will be
       automatically imposed on the gradient operator.

       The returned object is valid until the next call to this method or the
       destruction of this object.

       In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a true-dof vector. */
   Operator &GetGradient(const Vector &x) const override;

   /// Update the NonlinearForm to propagate updates of the associated FE space.
   /** After calling this method, the essential boundary conditions need to be
       set again. */
   virtual void Update();

   /** @brief Setup the NonlinearForm: based on the current AssemblyLevel and
       the current mesh, optionally, precompute and store data that will be
       reused in subsequent call to Mult(). */
   /** Typically, this method has to be called before Mult() when using
       AssemblyLevel::PARTIAL, after calling Update(), or after modifying the
       mesh coordinates. */
   virtual void Setup();

   /// Get the finite element space prolongation matrix
   const Operator *GetProlongation() const override { return P; }
   /// Get the finite element space restriction matrix
   const Operator *GetRestriction() const override
   { return fes->GetRestrictionMatrix(); }

   /// Indicate that integrators are not owned by the NonlinearForm
   void UseExternalIntegrators() { extern_bfs = 1; }

   /** @brief Destroy the NonlinearForm including the owned
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
   Array<Array<int>*>                   dnfi_marker;

   /// Set of Boundary Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> bnfi;
   Array<Array<int>*> bnfi_marker;

   /// Set of interior face Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> fnfi;

   /// Set of Boundary Face Integrators to be assembled (added).
   Array<BlockNonlinearFormIntegrator*> bfnfi;
   Array<Array<int>*>                   bfnfi_marker;

   /** Auxiliary block-vectors for wrapping input and output vectors or holding
       GridFunction-like block-vector data (e.g. in parallel). */
   mutable BlockVector xs, ys;

   mutable Array2D<SparseMatrix*> Grads, cGrads;
   mutable BlockOperator *BlockGrad;

   // A list of the offsets
   Array<int> block_offsets;
   Array<int> block_trueOffsets;

   // Array of Arrays of tdofs for each space in 'fes'
   Array<Array<int> *> ess_tdofs;

   /// Array of pointers to the prolongation matrix of fes, may be NULL
   Array<const Operator *> P;

   /// Array of results of dynamic-casting P to SparseMatrix pointer
   Array<const SparseMatrix *> cP;

   /// Indicator if the Operator is part of a parallel run
   bool is_serial = true;

   /// Indicator if the Operator needs prolongation on assembly
   bool needs_prolongation = false;

   mutable BlockVector aux1, aux2;

   const BlockVector &Prolongate(const BlockVector &bx) const;

   /// Specialized version of GetEnergy() for BlockVectors
   real_t GetEnergyBlocked(const BlockVector &bx) const;

   /// Specialized version of Mult() for BlockVector%s
   /// Block L-Vector to Block L-Vector
   void MultBlocked(const BlockVector &bx, BlockVector &by) const;

   /// Specialized version of GetGradient() for BlockVector
   void ComputeGradientBlocked(const BlockVector &bx) const;

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
   { dnfi.Append(nlfi); dnfi_marker.Append(NULL); }

   /// Adds new Domain Integrator, restricted to specific attributes.
   void AddDomainIntegrator(BlockNonlinearFormIntegrator *nlfi,
                            Array<int> &elem_marker)
   { dnfi.Append(nlfi); dnfi_marker.Append(&elem_marker); }

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { bnfi.Append(nlfi); bnfi_marker.Append(NULL); }

   /// Adds new Boundary Integrator, restricted to specific attributes.
   void AddBoundaryIntegrator(BlockNonlinearFormIntegrator *nlfi,
                              Array<int> &elem_marker)
   { bnfi.Append(nlfi); bnfi_marker.Append(&elem_marker); }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *nlfi,
                             Array<int> &bdr_marker)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(&bdr_marker); }

   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   virtual real_t GetEnergy(const Vector &x) const;

   /// Method is only called in serial, the parallel version calls MultBlocked
   /// directly.
   void Mult(const Vector &x, Vector &y) const override;

   /// Method is only called in serial, the parallel version calls
   /// GetGradientBlocked directly.
   Operator &GetGradient(const Vector &x) const override;

   /// Destructor.
   virtual ~BlockNonlinearForm();
};


}

#endif
