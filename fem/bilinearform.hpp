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

#ifndef MFEM_BILINEARFORM
#define MFEM_BILINEARFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "fespace.hpp"
#include "gridfunc.hpp"
#include "linearform.hpp"
#include "bilininteg.hpp"
#include "staticcond.hpp"
#include "hybridization.hpp"

namespace mfem
{

/** Class for bilinear form - "Matrix" with associated FE space and
    BLFIntegrators. */
class BilinearForm : public Matrix
{
protected:
   /// Sparse matrix to be associated with the form. Owned.
   SparseMatrix *mat;

   /// Matrix used to eliminate b.c. Owned.
   SparseMatrix *mat_e;

   /// FE space on which the form lives. Not owned.
   FiniteElementSpace *fes;

   /// Indicates the Mesh::sequence corresponding to the current state of the
   /// BilinearForm.
   long sequence;

   /** @brief Indicates the BilinearFormIntegrator%s stored in #dbfi, #bbfi,
       #fbfi, and #bfbfi are owned by another BilinearForm. */
   int extern_bfs;

   /// Set of Domain Integrators to be applied.
   Array<BilinearFormIntegrator*> dbfi;

   /// Set of Boundary Integrators to be applied.
   Array<BilinearFormIntegrator*> bbfi;
   Array<Array<int>*>             bbfi_marker; ///< Entries are not owned.

   /// Set of interior face Integrators to be applied.
   Array<BilinearFormIntegrator*> fbfi;

   /// Set of boundary face Integrators to be applied.
   Array<BilinearFormIntegrator*> bfbfi;
   Array<Array<int>*>             bfbfi_marker; ///< Entries are not owned.

   DenseMatrix elemmat;
   Array<int>  vdofs;

   DenseTensor *element_matrices; ///< Owned.

   StaticCondensation *static_cond; ///< Owned.
   Hybridization *hybridization; ///< Owned.

   /**
    * This member allows one to specify what should be done
    * to the diagonal matrix entries and corresponding RHS
    * values upon elimination of the constrained DoFs.
    */
   DiagonalPolicy diag_policy;

   int precompute_sparsity;
   // Allocate appropriate SparseMatrix and assign it to mat
   void AllocMat();

   void ConformingAssemble();

   // may be used in the construction of derived classes
   BilinearForm() : Matrix (0)
   {
      fes = NULL; sequence = -1;
      mat = mat_e = NULL; extern_bfs = 0; element_matrices = NULL;
      static_cond = NULL; hybridization = NULL;
      precompute_sparsity = 0;
      diag_policy = DIAG_KEEP;
   }

private:
   /// Copy construction is not supported; body is undefined.
   BilinearForm(const BilinearForm &);

   /// Copy assignment is not supported; body is undefined.
   BilinearForm &operator=(const BilinearForm &);

public:
   /// Creates bilinear form associated with FE space @a *f.
   /** The pointer @a f is not owned by the newly constructed object. */
   BilinearForm(FiniteElementSpace *f);

   /** @brief Create a BilinearForm on the FiniteElementSpace @a f, using the
       same integrators as the BilinearForm @a bf.

       The pointer @a f is not owned by the newly constructed object.

       The integrators in @a bf are copied as pointers and they are not owned by
       the newly constructed BilinearForm.

       The optional parameter @a ps is used to initialize the internal flag
       #precompute_sparsity, see UsePrecomputedSparsity() for details. */
   BilinearForm(FiniteElementSpace *f, BilinearForm *bf, int ps = 0);

   /// Get the size of the BilinearForm as a square matrix.
   int Size() const { return height; }

   /** Enable the use of static condensation. For details see the description
       for class StaticCondensation in fem/staticcond.hpp This method should be
       called before assembly. If the number of unknowns after static
       condensation is not reduced, it is not enabled. */
   void EnableStaticCondensation();

   /** Check if static condensation was actually enabled by a previous call to
       EnableStaticCondensation(). */
   bool StaticCondensationIsEnabled() const { return static_cond; }

   /// Return the trace FE space associated with static condensation.
   FiniteElementSpace *SCFESpace() const
   { return static_cond ? static_cond->GetTraceFESpace() : NULL; }

   /** Enable hybridization; for details see the description for class
       Hybridization in fem/hybridization.hpp. This method should be called
       before assembly. */
   void EnableHybridization(FiniteElementSpace *constr_space,
                            BilinearFormIntegrator *constr_integ,
                            const Array<int> &ess_tdof_list);

   /** For scalar FE spaces, precompute the sparsity pattern of the matrix
       (assuming dense element matrices) based on the types of integrators
       present in the bilinear form. */
   void UsePrecomputedSparsity(int ps = 1) { precompute_sparsity = ps; }

   /** @brief Use the given CSR sparsity pattern to allocate the internal
       SparseMatrix.

       - The @a I and @a J arrays must define a square graph with size equal to
         GetVSize() of the associated FiniteElementSpace.
       - This method should be called after enabling static condensation or
         hybridization, if used.
       - In the case of static condensation, @a I and @a J are not used.
       - The ownership of the arrays @a I and @a J remains with the caller. */
   void UseSparsity(int *I, int *J, bool isSorted);

   /// Use the sparsity of @a A to allocate the internal SparseMatrix.
   void UseSparsity(SparseMatrix &A);

   /** Pre-allocate the internal SparseMatrix before assembly. If the flag
       'precompute sparsity' is set, the matrix is allocated in CSR format (i.e.
       finalized) and the entries are initialized with zeros. */
   void AllocateMatrix() { if (mat == NULL) { AllocMat(); } }

   /// Access all integrators added with AddDomainIntegrator().
   Array<BilinearFormIntegrator*> *GetDBFI() { return &dbfi; }

   /// Access all integrators added with AddBoundaryIntegrator().
   Array<BilinearFormIntegrator*> *GetBBFI() { return &bbfi; }
   /** @brief Access all boundary markers added with AddBoundaryIntegrator().
       If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<Array<int>*> *GetBBFI_Marker() { return &bbfi_marker; }

   /// Access all integrators added with AddInteriorFaceIntegrator().
   Array<BilinearFormIntegrator*> *GetFBFI() { return &fbfi; }

   /// Access all integrators added with AddBdrFaceIntegrator().
   Array<BilinearFormIntegrator*> *GetBFBFI() { return &bfbfi; }
   /** @brief Access all boundary markers added with AddBdrFaceIntegrator().
       If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<Array<int>*> *GetBFBFI_Marker() { return &bfbfi_marker; }

   const double &operator()(int i, int j) { return (*mat)(i,j); }

   /// Returns reference to a_{ij}.
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.
   virtual const double &Elem(int i, int j) const;

   /// Matrix vector multiplication.
   virtual void Mult(const Vector &x, Vector &y) const { mat->Mult(x, y); }

   void FullMult(const Vector &x, Vector &y) const
   { mat->Mult(x, y); mat_e->AddMult(x, y); }

   virtual void AddMult(const Vector &x, Vector &y, const double a = 1.0) const
   { mat -> AddMult (x, y, a); }

   void FullAddMult(const Vector &x, Vector &y) const
   { mat->AddMult(x, y); mat_e->AddMult(x, y); }

   virtual void AddMultTranspose(const Vector & x, Vector & y,
                                 const double a = 1.0) const
   { mat->AddMultTranspose(x, y, a); }

   void FullAddMultTranspose(const Vector & x, Vector & y) const
   { mat->AddMultTranspose(x, y); mat_e->AddMultTranspose(x, y); }

   virtual void MultTranspose(const Vector & x, Vector & y) const
   { y = 0.0; AddMultTranspose (x, y); }

   double InnerProduct(const Vector &x, const Vector &y) const
   { return mat->InnerProduct (x, y); }

   /// Returns a pointer to (approximation) of the matrix inverse.
   virtual MatrixInverse *Inverse() const;

   /// Finalizes the matrix initialization.
   virtual void Finalize(int skip_zeros = 1);

   /// Returns a reference to the sparse matrix
   const SparseMatrix &SpMat() const
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return *mat;
   }
   SparseMatrix &SpMat()
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return *mat;
   }
   SparseMatrix *LoseMat() { SparseMatrix *tmp = mat; mat = NULL; return tmp; }

   /// Returns a reference to the sparse matrix of eliminated b.c.
   const SparseMatrix &SpMatElim() const
   {
      MFEM_VERIFY(mat_e, "mat_e is NULL and can't be dereferenced");
      return *mat_e;
   }
   SparseMatrix &SpMatElim()
   {
      MFEM_VERIFY(mat_e, "mat_e is NULL and can't be dereferenced");
      return *mat_e;
   }

   /// Adds new Domain Integrator. Assumes ownership of @a bfi.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator. Assumes ownership of @a bfi.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi);

   /** @brief Adds new Boundary Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi. The array @a bdr_marker is stored internally
       as a pointer to the given Array<int> object. */
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi,
                              Array<int> &bdr_marker);

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new boundary Face Integrator. Assumes ownership of @a bfi.
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi. The array @a bdr_marker is stored internally
       as a pointer to the given Array<int> object. */
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi,
                             Array<int> &bdr_marker);

   void operator=(const double a)
   {
      if (mat != NULL) { *mat = a; }
      if (mat_e != NULL) { *mat_e = a; }
   }

   /// Assembles the form i.e. sums over all domain/bdr integrators.
   void Assemble(int skip_zeros = 1);

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const
   { return fes->GetConformingProlongation(); }
   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return fes->GetConformingRestriction(); }

   /// Form a linear system, A X = B.
   /** Form the linear system A X = B, corresponding to the current bilinear
       form and b(.), by applying any necessary transformations such as:
       eliminating boundary conditions; applying conforming constraints for
       non-conforming AMR; static condensation; hybridization.

       The GridFunction-size vector @a x must contain the essential b.c. The
       BilinearForm and the LinearForm-size vector @a b must be assembled.

       The vector @a X is initialized with a suitable initial guess: when using
       hybridization, the vector @a X is set to zero; otherwise, the essential
       entries of @a X are set to the corresponding b.c. and all other entries
       are set to zero (@a copy_interior == 0) or copied from @a x
       (@a copy_interior != 0).

       This method can be called multiple times (with the same @a ess_tdof_list
       array) to initialize different right-hand sides and boundary condition
       values.

       After solving the linear system, the finite element solution @a x can be
       recovered by calling RecoverFEMSolution() (with the same vectors @a X,
       @a b, and @a x).

       NOTE: If there are no transformations, @a X simply reuses the data of
             @a x. */
   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         SparseMatrix &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /// Form the linear system matrix A, see FormLinearSystem() for details.
   void FormSystemMatrix(const Array<int> &ess_tdof_list, SparseMatrix &A);

   /// Recover the solution of a linear system formed with FormLinearSystem().
   /** Call this method after solving a linear system constructed using the
       FormLinearSystem() method to recover the solution as a GridFunction-size
       vector in @a x. Use the same arguments as in the FormLinearSystem() call.
   */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   /// Compute and store internally all element matrices.
   void ComputeElementMatrices();

   /// Free the memory used by the element matrices.
   void FreeElementMatrices()
   { delete element_matrices; element_matrices = NULL; }

   void ComputeElementMatrix(int i, DenseMatrix &elmat);
   void AssembleElementMatrix(int i, const DenseMatrix &elmat,
                              Array<int> &vdofs, int skip_zeros = 1);
   void AssembleBdrElementMatrix(int i, const DenseMatrix &elmat,
                                 Array<int> &vdofs, int skip_zeros = 1);

   /// Eliminate essential boundary DOFs from the system.
   /** The array @a bdr_attr_is_ess marks boundary attributes that constitute
       the essential part of the boundary. By default, the diagonal at the
       essential DOFs is set to 1.0. This behavior is controlled by the argument
       @a dpolicy. */
   void EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                             const Vector &sol, Vector &rhs,
                             DiagonalPolicy dpolicy = DIAG_ONE);

   /// Eliminate essential boundary DOFs from the system matrix.
   void EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                             DiagonalPolicy dpolicy = DIAG_ONE);
   /// Perform elimination and set the diagonal entry to the given value
   void EliminateEssentialBCDiag(const Array<int> &bdr_attr_is_ess,
                                 double value);

   /// Eliminate the given @a vdofs. NOTE: here, @a vdofs is a list of DOFs.
   void EliminateVDofs(const Array<int> &vdofs, const Vector &sol, Vector &rhs,
                       DiagonalPolicy dpolicy = DIAG_ONE);

   /// Eliminate the given @a vdofs, storing the eliminated part internally.
   /** This method works in conjunction with EliminateVDofsInRHS() and allows
       elimination of boundary conditions in multiple right-hand sides. In this
       method, @a vdofs is a list of DOFs. */
   void EliminateVDofs(const Array<int> &vdofs,
                       DiagonalPolicy dpolicy = DIAG_ONE);

   /** @brief Similar to
       EliminateVDofs(const Array<int> &, const Vector &, Vector &, DiagonalPolicy)
       but here @a ess_dofs is a marker (boolean) array on all vector-dofs
       (@a ess_dofs[i] < 0 is true). */
   void EliminateEssentialBCFromDofs(const Array<int> &ess_dofs, const Vector &sol,
                                     Vector &rhs, DiagonalPolicy dpolicy = DIAG_ONE);

   /** @brief Similar to EliminateVDofs(const Array<int> &, DiagonalPolicy) but
       here @a ess_dofs is a marker (boolean) array on all vector-dofs
       (@a ess_dofs[i] < 0 is true). */
   void EliminateEssentialBCFromDofs(const Array<int> &ess_dofs,
                                     DiagonalPolicy dpolicy = DIAG_ONE);
   /// Perform elimination and set the diagonal entry to the given value
   void EliminateEssentialBCFromDofsDiag(const Array<int> &ess_dofs,
                                         double value);

   /** @brief Use the stored eliminated part of the matrix (see
       EliminateVDofs(const Array<int> &, DiagonalPolicy)) to modify the r.h.s.
       @a b; @a vdofs is a list of DOFs (non-directional, i.e. >= 0). */
   void EliminateVDofsInRHS(const Array<int> &vdofs, const Vector &x,
                            Vector &b);

   double FullInnerProduct(const Vector &x, const Vector &y) const
   { return mat->InnerProduct(x, y) + mat_e->InnerProduct(x, y); }

   virtual void Update(FiniteElementSpace *nfes = NULL);

   /// (DEPRECATED) Return the FE space associated with the BilinearForm.
   /** @deprecated Use FESpace() instead. */
   FiniteElementSpace *GetFES() { return fes; }

   /// Return the FE space associated with the BilinearForm.
   FiniteElementSpace *FESpace() { return fes; }
   /// Read-only access to the associated FiniteElementSpace.
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Sets diagonal policy used upon construction of the linear system
   void SetDiagonalPolicy(DiagonalPolicy policy);

   /// Destroys bilinear form.
   virtual ~BilinearForm();
};

/**
   Class for assembling of bilinear forms `a(u,v)` defined on different
   trial and test spaces. The assembled matrix `A` is such that

       a(u,v) = V^t A U

   where `U` and `V` are the vectors representing the functions `u` and `v`,
   respectively.  The first argument, `u`, of `a(,)` is in the trial space
   and the second argument, `v`, is in the test space. Thus,

       # of rows of A = dimension of the test space and
       # of cols of A = dimension of the trial space.

   Both trial and test spaces should be defined on the same mesh.
*/
class MixedBilinearForm : public Matrix
{
protected:
   SparseMatrix *mat; ///< Owned.

   FiniteElementSpace *trial_fes, ///< Not owned
                      *test_fes;  ///< Not owned

   /** @brief Indicates the BilinearFormIntegrator%s stored in #dom, #bdr, and
       #skt are owned by another MixedBilinearForm. */
   int extern_bfs;

   /// Domain integrators.
   Array<BilinearFormIntegrator*> dom;
   /// Boundary integrators.
   Array<BilinearFormIntegrator*> bdr;
   /// Trace face (skeleton) integrators.
   Array<BilinearFormIntegrator*> skt;

private:
   /// Copy construction is not supported; body is undefined.
   MixedBilinearForm(const MixedBilinearForm &);

   /// Copy assignment is not supported; body is undefined.
   MixedBilinearForm &operator=(const MixedBilinearForm &);

public:
   /** @brief Construct a MixedBilinearForm on the given trial, @a tr_fes, and
       test, @a te_fes, FiniteElementSpace%s. */
   /** The pointers @a tr_fes and @a te_fes are not owned by the newly
       constructed object. */
   MixedBilinearForm(FiniteElementSpace *tr_fes,
                     FiniteElementSpace *te_fes);

   /** @brief Create a MixedBilinearForm on the given trial, @a tr_fes, and
       test, @a te_fes, FiniteElementSpace%s, using the same integrators as the
       MixedBilinearForm @a mbf.

       The pointers @a tr_fes and @a te_fes are not owned by the newly
       constructed object.

       The integrators in @a mbf are copied as pointers and they are not owned
       by the newly constructed MixedBilinearForm. */
   MixedBilinearForm(FiniteElementSpace *tr_fes,
                     FiniteElementSpace *te_fes,
                     MixedBilinearForm *mbf);

   virtual double &Elem(int i, int j);

   virtual const double &Elem(int i, int j) const;

   virtual void Mult(const Vector & x, Vector & y) const;

   virtual void AddMult(const Vector & x, Vector & y,
                        const double a = 1.0) const;

   virtual void AddMultTranspose(const Vector & x, Vector & y,
                                 const double a = 1.0) const;

   virtual void MultTranspose(const Vector & x, Vector & y) const
   { y = 0.0; AddMultTranspose (x, y); }

   virtual MatrixInverse *Inverse() const;

   virtual void Finalize(int skip_zeros = 1);

   /** Extract the associated matrix as SparseMatrix blocks. The number of
       block rows and columns is given by the vector dimensions (vdim) of the
       test and trial spaces, respectively. */
   void GetBlocks(Array2D<SparseMatrix *> &blocks) const;

   const SparseMatrix &SpMat() const { return *mat; }
   SparseMatrix &SpMat() { return *mat; }
   SparseMatrix *LoseMat() { SparseMatrix *tmp = mat; mat = NULL; return tmp; }

   /// Adds a domain integrator. Assumes ownership of @a bfi.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi);

   /// Adds a boundary integrator. Assumes ownership of @a bfi.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi);

   /** @brief Add a trace face integrator. Assumes ownership of @a bfi.

       This type of integrator assembles terms over all faces of the mesh using
       the face FE from the trial space and the two adjacent volume FEs from the
       test space. */
   void AddTraceFaceIntegrator(BilinearFormIntegrator *bfi);

   /// Access all integrators added with AddDomainIntegrator().
   Array<BilinearFormIntegrator*> *GetDBFI() { return &dom; }

   /// Access all integrators added with AddBoundaryIntegrator().
   Array<BilinearFormIntegrator*> *GetBBFI() { return &bdr; }

   /// Access all integrators added with AddTraceFaceIntegrator().
   Array<BilinearFormIntegrator*> *GetTFBFI() { return &skt; }

   void operator=(const double a) { *mat = a; }

   void Assemble(int skip_zeros = 1);

   /** For partially conforming trial and/or test FE spaces, complete the
       assembly process by performing A := P2^t A P1 where A is the internal
       sparse matrix; P1 and P2 are the conforming prolongation matrices of the
       trial and test FE spaces, respectively. After this call the
       MixedBilinearForm becomes an operator on the conforming FE spaces. */
   void ConformingAssemble();

   void EliminateTrialDofs(Array<int> &bdr_attr_is_ess,
                           const Vector &sol, Vector &rhs);

   void EliminateEssentialBCFromTrialDofs(Array<int> &marked_vdofs,
                                          const Vector &sol, Vector &rhs);

   virtual void EliminateTestDofs(Array<int> &bdr_attr_is_ess);

   void Update();

   virtual ~MixedBilinearForm();
};


/**
   Class for constructing the matrix representation of a linear operator,
   `v = L u`, from one FiniteElementSpace (domain) to another FiniteElementSpace
   (range). The constructed matrix `A` is such that

       V = A U

   where `U` and `V` are the vectors of degrees of freedom representing the
   functions `u` and `v`, respectively. The dimensions of `A` are

       number of rows of A = dimension of the range space and
       number of cols of A = dimension of the domain space.

   This class is very similar to MixedBilinearForm. One difference is that
   the linear operator `L` is defined using a special kind of
   BilinearFormIntegrator (we reuse its functionality instead of defining a
   new class). The other difference with the MixedBilinearForm class is that
   the "assembly" process overwrites the global matrix entries using the
   local element matrices instead of adding them.

   Note that if we define the bilinear form `b(u,v) := (Lu,v)` using an inner
   product in the range space, then its matrix representation, `B`, is

       B = M A, (since V^t B U = b(u,v) = (Lu,v) = V^t M A U)

   where `M` denotes the mass matrix for the inner product in the range space:
   `V1^t M V2 = (v1,v2)`. Similarly, if `c(u,w) := (Lu,Lw)` then

       C = A^t M A.
*/
class DiscreteLinearOperator : public MixedBilinearForm
{
private:
   /// Copy construction is not supported; body is undefined.
   DiscreteLinearOperator(const DiscreteLinearOperator &);

   /// Copy assignment is not supported; body is undefined.
   DiscreteLinearOperator &operator=(const DiscreteLinearOperator &);

public:
   /** @brief Construct a DiscreteLinearOperator on the given
       FiniteElementSpace%s @a domain_fes and @a range_fes. */
   /** The pointers @a domain_fes and @a range_fes are not owned by the newly
       constructed object. */
   DiscreteLinearOperator(FiniteElementSpace *domain_fes,
                          FiniteElementSpace *range_fes)
      : MixedBilinearForm(domain_fes, range_fes) { }

   /// Adds a domain interpolator. Assumes ownership of @a di.
   void AddDomainInterpolator(DiscreteInterpolator *di)
   { AddDomainIntegrator(di); }

   /// Adds a trace face interpolator. Assumes ownership of @a di.
   void AddTraceFaceInterpolator(DiscreteInterpolator *di)
   { AddTraceFaceIntegrator(di); }

   /// Access all interpolators added with AddDomainInterpolator().
   Array<BilinearFormIntegrator*> *GetDI() { return &dom; }

   /** @brief Construct the internal matrix representation of the discrete
       linear operator. */
   virtual void Assemble(int skip_zeros = 1);
};

}

#endif
