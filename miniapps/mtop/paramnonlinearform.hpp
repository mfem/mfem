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

#ifndef MFEM_PRMNONLINEARFORM
#define MFEM_PRMNONLINEARFORM

#include "mfem.hpp"

namespace mfem
{

/** The abstract base class ParametricBNLFormIntegrator is a generalization of
    the BlockNonlinearFormIntegrator class suitable for block state and
    parameter vectors. */
class ParametricBNLFormIntegrator
{
public:
   /// Compute the local energy
   virtual real_t GetElementEnergy(const Array<const FiniteElement *>&el,
                                   const Array<const FiniteElement *>&pel,
                                   ElementTransformation &Tr,
                                   const Array<const Vector *>&elfun,
                                   const Array<const Vector *>&pelfun);

   /// Perform the local action of the BlockNonlinearFormIntegrator
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      const Array<const FiniteElement *>&pel,
                                      ElementTransformation &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<const Vector *>&pelfun,
                                      const Array<Vector *> &elvec);

   /// Perform the local action of the BlockNonlinearFormIntegrator on element
   /// faces
   virtual void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                                   const Array<const FiniteElement *> &el2,
                                   const Array<const FiniteElement *> &pel1,
                                   const Array<const FiniteElement *> &pel2,
                                   FaceElementTransformations &Tr,
                                   const Array<const Vector *> &elfun,
                                   const Array<const Vector *>&pelfun,
                                   const Array<Vector *> &elvect);

   /// Perform the local action on the parameters of the BNLFormIntegrator
   virtual void AssemblePrmElementVector(const Array<const FiniteElement *> &el,
                                         const Array<const FiniteElement *>&pel,
                                         ElementTransformation &Tr,
                                         const Array<const Vector *> &elfun,
                                         const Array<const Vector *> &alfun,
                                         const Array<const Vector *>&pelfun,
                                         const Array<Vector *> &pelvec);

   /// Perform the local action on the parameters of the BNLFormIntegrator on
   /// faces
   virtual void AssemblePrmFaceVector(const Array<const FiniteElement *> &el1,
                                      const Array<const FiniteElement *> &el2,
                                      const Array<const FiniteElement *> &pel1,
                                      const Array<const FiniteElement *> &pel2,
                                      FaceElementTransformations &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<const Vector *> &alfun,
                                      const Array<const Vector *>&pelfun,
                                      const Array<Vector *> &pelvect);

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                    const Array<const FiniteElement *>&pel,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array<const Vector *>&pelfun,
                                    const Array2D<DenseMatrix *> &elmats);

   /// Assemble the local gradient matrix on faces of the elements
   virtual void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                                 const Array<const FiniteElement *>&el2,
                                 const Array<const FiniteElement *> &pel1,
                                 const Array<const FiniteElement *> &pel2,
                                 FaceElementTransformations &Tr,
                                 const Array<const Vector *> &elfun,
                                 const Array<const Vector *>&pelfun,
                                 const Array2D<DenseMatrix *> &elmats);


   virtual ~ParametricBNLFormIntegrator() { }
};


/** @brief A class representing a general parametric block nonlinear operator
    defined on the Cartesian product of multiple FiniteElementSpace%s. */
class ParametricBNLForm : public Operator
{
protected:
   /// FE spaces on which the form lives.
   Array<FiniteElementSpace*> fes;

   /// FE spaces for the parametric fields
   Array<FiniteElementSpace*> paramfes;

   int paramheight;
   int paramwidth;

   /// Set of Domain Integrators to be assembled (added).
   Array<ParametricBNLFormIntegrator*> dnfi;

   /// Set of interior face Integrators to be assembled (added).
   Array<ParametricBNLFormIntegrator*> fnfi;

   /// Set of Boundary Face Integrators to be assembled (added).
   Array<ParametricBNLFormIntegrator*> bfnfi;
   Array<Array<int>*>           bfnfi_marker;

   /** Auxiliary block-vectors for wrapping input and output vectors or holding
       GridFunction-like block-vector data (e.g. in parallel). */
   mutable BlockVector xs, ys;
   mutable BlockVector prmxs, prmys;

   /** Auxiliary block-vectors for holding GridFunction-like block-vector data
       (e.g. in parallel). */
   mutable BlockVector xsv;

   /** Auxiliary block-vectors for holding GridFunction-like block-vector data
       for the parameter fields (e.g. in parallel). */
   mutable BlockVector xdv;
   /** Auxiliary block-vectors for holding GridFunction-like block-vector data
       for the adjoint fields (e.g. in parallel). */
   mutable BlockVector adv;

   mutable Array2D<SparseMatrix*> Grads, cGrads;
   mutable BlockOperator *BlockGrad;

   // A list of the offsets
   Array<int> block_offsets;
   Array<int> block_trueOffsets;
   // A list with the offsets for the parametric fields
   Array<int> paramblock_offsets;
   Array<int> paramblock_trueOffsets;

   // Array of Arrays of tdofs for each space in 'fes'
   Array<Array<int> *> ess_tdofs;

   // Array of Arrays of tdofs for each space in 'paramfes'
   Array<Array<int> *> paramess_tdofs;

   /// Array of pointers to the prolongation matrix of fes, may be NULL
   Array<const Operator *> P;

   /// Array of pointers to the prolongation matrix of paramfes, may be NULL
   Array<const Operator *> Pparam;

   /// Array of results of dynamic-casting P to SparseMatrix pointer
   Array<const SparseMatrix *> cP;

   /// Array of results of dynamic-casting Pparam to SparseMatrix pointer
   Array<const SparseMatrix *> cPparam;

   /// Indicator if the Operator is part of a parallel run
   bool is_serial = true;

   /// Indicator if the Operator needs prolongation on assembly
   bool needs_prolongation = false;

   /// Indicator if the Operator needs prolongation on assembly
   bool prmneeds_prolongation = false;

   mutable BlockVector aux1, aux2;

   mutable BlockVector prmaux1, prmaux2;

   const BlockVector &Prolongate(const BlockVector &bx) const;

   const BlockVector &ParamProlongate(const BlockVector &bx) const;

   real_t GetEnergyBlocked(const BlockVector &bx, const BlockVector &dx) const;


   /// Specialized version of Mult() for BlockVector%s
   /// Block L-Vector to Block L-Vector
   void MultBlocked(const BlockVector &bx, const BlockVector &dx,
                    BlockVector &by) const;

   /// Specialized version of Mult() for BlockVector%s
   /// Block L-Vector to Block L-Vector
   /// bx - state vector, ax - adjoint vector, dx - parametric fields
   /// dy = ax' d(residual(bx))/d(dx)
   void MultParamBlocked(const BlockVector &bx, const BlockVector & ax,
                         const BlockVector &dx, BlockVector &dy) const;


   /// Specialized version of GetGradient() for BlockVector
   void ComputeGradientBlocked(const BlockVector &bx, const BlockVector &dx) const;

public:
   /// Construct an empty BlockNonlinearForm. Initialize with SetSpaces().
   ParametricBNLForm();

   /// Construct a BlockNonlinearForm on the given set of FiniteElementSpace%s.
   ParametricBNLForm(Array<FiniteElementSpace *> &statef,
                     Array<FiniteElementSpace *> &paramf);

   /// Return the @a k-th FE space of the ParametricBNLForm.
   FiniteElementSpace *FESpace(int k) { return fes[k]; }

   /// Return the @a k-th parametric FE space of the ParametricBNLForm.
   FiniteElementSpace *ParamFESpace(int k) { return paramfes[k]; }


   /// Return the @a k-th FE space of the BlockNonlinearForm (const version).
   const FiniteElementSpace *FESpace(int k) const { return fes[k]; }

   /// Return the @a k-th parametric FE space of the BlockNonlinearForm (const
   /// version).
   const FiniteElementSpace *ParamFESpace(int k) const { return paramfes[k]; }

   /// Return the integrators
   Array<ParametricBNLFormIntegrator*>& GetDNFI() { return dnfi;}


   /// (Re)initialize the ParametricBNLForm.
   /** After a call to SetSpaces(), the essential b.c. must be set again. */
   void SetSpaces(Array<FiniteElementSpace *> &statef,
                  Array<FiniteElementSpace *> &paramf);

   /// Return the regular dof offsets.
   const Array<int> &GetBlockOffsets() const { return block_offsets; }

   /// Return the true-dof offsets.
   const Array<int> &GetBlockTrueOffsets() const { return block_trueOffsets; }

   /// Return the regular dof offsets for the parameters.
   const Array<int> &ParamGetBlockOffsets() const { return paramblock_offsets; }

   /// Return the true-dof offsets for the parameters.
   const Array<int> &ParamGetBlockTrueOffsets() const { return paramblock_trueOffsets; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(ParametricBNLFormIntegrator *nlfi)
   { dnfi.Append(nlfi); }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(ParametricBNLFormIntegrator *nlfi)
   { fnfi.Append(nlfi); }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(ParametricBNLFormIntegrator *nlfi)
   { bfnfi.Append(nlfi); bfnfi_marker.Append(NULL); }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(ParametricBNLFormIntegrator *nlfi,
                             Array<int> &bdr_marker);

   /// Set the essential boundary conditions.
   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   /// Set the essential boundary conditions on the parametric fields.
   virtual void SetParamEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                                    Array<Vector *> &rhs);


   /// Computes the energy for a state vector x.
   virtual real_t GetEnergy(const Vector &x) const;

   /// Method is only called in serial, the parallel version calls MultBlocked
   /// directly.
   void Mult(const Vector &x, Vector &y) const override;

   /// Method is only called in serial, the parallel version calls MultBlocked
   /// directly.
   virtual void ParamMult(const Vector &x, Vector &y) const;

   /// Method is only called in serial, the parallel version calls
   /// GetGradientBlocked directly.
   BlockOperator &GetGradient(const Vector &x) const override;

   /// Set the state fields
   virtual void SetStateFields(const Vector &xv) const;

   /// Set the adjoint fields
   virtual void SetAdjointFields(const Vector &av) const;

   /// Set the parameters/design fields
   virtual void SetParamFields(const Vector &dv) const;

   /// Destructor.
   virtual ~ParametricBNLForm();

};

}

#endif
