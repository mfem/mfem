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

#ifndef MFEM_PRMNONLINEARFORM
#define MFEM_PRMNONLINEARFORM

#include "../config/config.hpp"
#include "nonlininteg.hpp"
#include "nonlinearform_ext.hpp"
#include "bilinearform.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/** @brief A class representing a general parametric block nonlinear operator
    defined on the Cartesian product of multiple FiniteElementSpace%s. */
class PrmBlockNonlinearForm : public Operator
{
protected:
    /// FE spaces on which the form lives.
    Array<FiniteElementSpace*> fes;

    /// FE spaces for the parametric fields
    Array<FiniteElementSpace*> prmfes;

    int prmheight;
    int prmwidth;


    /// Set of Domain Integrators to be assembled (added).
    Array<PrmBlockNonlinearFormIntegrator*> dnfi;

    /// Set of interior face Integrators to be assembled (added).
    Array<PrmBlockNonlinearFormIntegrator*> fnfi;

    /// Set of Boundary Face Integrators to be assembled (added).
    Array<PrmBlockNonlinearFormIntegrator*> bfnfi;
    Array<Array<int>*>           bfnfi_marker;

    /** Auxiliary block-vectors for wrapping input and output vectors or holding
           GridFunction-like block-vector data (e.g. in parallel). */
    mutable BlockVector xs, ys;
    mutable BlockVector prmxs, prmys;


    /** Auxiliary block-vectors for holding
           GridFunction-like block-vector data (e.g. in parallel). */
    mutable BlockVector xsv;

    /** Auxiliary block-vectors for holding
           GridFunction-like block-vector data for the parameter fields
                                                (e.g. in parallel). */
    mutable BlockVector xdv;
    /** Auxiliary block-vectors for holding
           GridFunction-like block-vector data for the adjoint fields
                                                (e.g. in parallel). */
    mutable BlockVector adv;


    mutable Array2D<SparseMatrix*> Grads, cGrads;
    mutable BlockOperator *BlockGrad;

    // A list of the offsets
    Array<int> block_offsets;
    Array<int> block_trueOffsets;
    // A list with the offsets for the parametric fields
    Array<int> prmblock_offsets;
    Array<int> prmblock_trueOffsets;

    // Array of Arrays of tdofs for each space in 'fes'
    Array<Array<int> *> ess_tdofs;

    // Array of Arrays of tdofs for each space in 'prmfes'
    Array<Array<int> *> prmess_tdofs;

    /// Array of pointers to the prolongation matrix of fes, may be NULL
    Array<const Operator *> P;

    /// Array of pointers to the prolongation matrix of prmfes, may be NULL
    Array<const Operator *> Pprm;

    /// Array of results of dynamic-casting P to SparseMatrix pointer
    Array<const SparseMatrix *> cP;

    /// Array of results of dynamic-casting Pprm to SparseMatrix pointer
    Array<const SparseMatrix *> cPprm;


    /// Indicator if the Operator is part of a parallel run
    bool is_serial = true;

    /// Indicator if the Operator needs prolongation on assembly
    bool needs_prolongation = false;

    /// Indicator if the Operator needs prolongation on assembly
    bool prmneeds_prolongation = false;


    mutable BlockVector aux1, aux2;

    mutable BlockVector prmaux1, prmaux2;

    const BlockVector &Prolongate(const BlockVector &bx) const;

    const BlockVector &PrmProlongate(const BlockVector &bx) const;

    /// Specialized version of GetEnergy() for BlockVectors
    //double GetEnergyBlocked(const BlockVector &bx) const;
    double GetEnergyBlocked(const BlockVector &bx, const BlockVector &dx) const;


    /// Specialized version of Mult() for BlockVector%s
    /// Block L-Vector to Block L-Vector
    void MultBlocked(const BlockVector &bx, const BlockVector &dx, BlockVector &by) const;

    /// Specialized version of Mult() for BlockVector%s
    /// Block L-Vector to Block L-Vector
    /// bx - state vector, ax - adjoint vector, dx - parametric fields
    /// dy = ax' d(residual(bx))/d(dx)
    void MultPrmBlocked(const BlockVector &bx, const BlockVector & ax, const BlockVector &dx, BlockVector &dy) const;


    /// Specialized version of GetGradient() for BlockVector
    //void ComputeGradientBlocked(const BlockVector &bx) const;
    void ComputeGradientBlocked(const BlockVector &bx, const BlockVector &dx) const;

public:
   /// Construct an empty BlockNonlinearForm. Initialize with SetSpaces().
   PrmBlockNonlinearForm();

   /// Construct a BlockNonlinearForm on the given set of FiniteElementSpace%s.
   PrmBlockNonlinearForm(Array<FiniteElementSpace *> &f, Array<FiniteElementSpace *> &pf );

   /// Return the @a k-th FE space of the PrmBlockNonlinearForm.
   FiniteElementSpace *FESpace(int k) { return fes[k]; }

   /// Return the @a k-th parametric FE space of the PrmBlockNonlinearForm.
   FiniteElementSpace *PrmFESpace(int k) { return prmfes[k]; }


   /// Return the @a k-th FE space of the BlockNonlinearForm (const version).
   const FiniteElementSpace *FESpace(int k) const { return fes[k]; }

   /// Return the @a k-th parametric FE space of the BlockNonlinearForm (const version).
   const FiniteElementSpace *PrmFESpace(int k) const { return prmfes[k]; }

   Array<PrmBlockNonlinearFormIntegrator*>& GetDNFI(){ return dnfi;}


   /// (Re)initialize the PrmBlockNonlinearForm.
   /** After a call to SetSpaces(), the essential b.c. must be set again. */
   void SetSpaces(Array<FiniteElementSpace *> &f, Array<FiniteElementSpace *> &prmf);

   /// Return the regular dof offsets.
   const Array<int> &GetBlockOffsets() const { return block_offsets; }
   /// Return the true-dof offsets.
   const Array<int> &GetBlockTrueOffsets() const { return block_trueOffsets; }

   /// Return the regular dof offsets for the parameters.
   const Array<int> &PrmGetBlockOffsets() const { return prmblock_offsets; }
   /// Return the true-dof offsets for the parameters.
   const Array<int> &PrmGetBlockTrueOffsets() const { return prmblock_trueOffsets; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(PrmBlockNonlinearFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(PrmBlockNonlinearFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(PrmBlockNonlinearFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(PrmBlockNonlinearFormIntegrator *nlfi,
                             Array<int> &bdr_marker);

   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   virtual void SetPrmEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);


   virtual double GetEnergy(const Vector &x) const;

   /// Method is only called in serial, the parallel version calls MultBlocked
   /// directly.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Method is only called in serial, the parallel version calls MultBlocked
   /// directly.
   virtual void PrmMult(const Vector &x, Vector &t) const;

   /// Method is only called in serial, the parallel version calls
   /// GetGradientBlocked directly.
   virtual Operator &GetGradient(const Vector &x) const;

   /// Set the state fields
   virtual void SetStateFields(const Vector &xv) const;

   /// Set the adjoint fields
   virtual void SetAdjointFields(const Vector &av) const;

   /// Set the parameters/design fields
   virtual void SetPrmFields(const Vector &dv) const;



   /// Destructor.
   virtual ~PrmBlockNonlinearForm();

};

}

#endif


