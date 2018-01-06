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
   // TODO remove mat
   /// Sparse matrix to be associated with the form.
   SparseMatrix *mat;

   /// Matrix used to eliminate b.c.
   SparseMatrix *mat_e;

   /// Generic operator associated with the form.
   Operator *oper;

   /// Operator type.
   enum Type oper_type;

   /// FE space on which the form lives.
   FiniteElementSpace *fes;

   /// Indicates the Mesh::sequence corresponding to the current state of the
   /// BilinearForm.
   long sequence;

   int extern_bfs;

   /// Set of Domain Integrators to be applied.
   Array<BilinearFormIntegrator*> dbfi;

   /// Set of Boundary Integrators to be applied.
   Array<BilinearFormIntegrator*> bbfi;

   /// Set of interior face Integrators to be applied.
   Array<BilinearFormIntegrator*> fbfi;

   /// Set of boundary face Integrators to be applied.
   Array<BilinearFormIntegrator*> bfbfi;
   Array<Array<int>*>             bfbfi_marker;

   /// Set of fespace integrators (does not matter what type)
   Array<LinearFESpaceIntegrator*> fesi;

   DenseMatrix elemmat;
   Array<int>  vdofs;

   DenseTensor *element_matrices;

   StaticCondensation *static_cond;
   Hybridization *hybridization;

   int precompute_sparsity;
   // Allocate appropriate SparseMatrix and assign it to mat
   void AllocMat();

   void ConformingAssemble();

   // may be used in the construction of derived classes
   BilinearForm() : Matrix (0)
   {
      fes = NULL; sequence = -1;
      mat = mat_e = NULL;
      oper = NULL; oper_type = MFEM_SPARSEMAT;
      extern_bfs = 0; element_matrices = NULL;
      static_cond = NULL; hybridization = NULL;
      precompute_sparsity = 0;
   }

public:
   /// Creates bilinear form associated with FE space *f.
   BilinearForm(FiniteElementSpace *f);

   BilinearForm(FiniteElementSpace *f, BilinearForm *bf, int ps = 0);

   /// Get the size of the BilinearForm as a square matrix.
   int Size() const { return height; }

   /** Enable the use of static condensation. For details see the description
       for class StaticCondensation in fem/staticcond.hpp This method should be
       called before assembly. If the number of unknowns after static
       condensation is not reduced, it is not enabled. */
   void EnableStaticCondensation();

   /** Check if static condensation was actually enabled by a previous call to
       EnableStaticCondensation. */
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

   Array<BilinearFormIntegrator*> *GetDBFI() { return &dbfi; }

   Array<BilinearFormIntegrator*> *GetBBFI() { return &bbfi; }

   Array<BilinearFormIntegrator*> *GetFBFI() { return &fbfi; }

   Array<BilinearFormIntegrator*> *GetBFBFI() { return &bfbfi; }

   Array<LinearFESpaceIntegrator*> *GetFESI() { return &fesi; }

   const double &operator()(int i, int j) { return (*mat)(i,j); }

   /// Returns reference to a_{ij}.
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.
   virtual const double &Elem(int i, int j) const;

   /// Matrix vector multiplication.
   virtual void Mult(const Vector &x, Vector &y) const { oper->Mult(x, y); }

   void FullMult(const Vector &x, Vector &y) const
   { mat->Mult(x, y); mat_e->AddMult(x, y); }

   virtual void AddMult(const Vector &x, Vector &y, const double a = 1.0) const
   { mat -> AddMult (x, y, a); }

   void FullAddMult(const Vector &x, Vector &y) const
   { mat->AddMult(x, y); mat_e->AddMult(x, y); }

   virtual void AddMultTranspose(const Vector & x, Vector & y,
                                 const double a = 1.0) const
   { mat->AddMultTranspose(x, y, a); }

   void FullAddMultTranspose (const Vector & x, Vector & y) const
   { mat->AddMultTranspose(x, y); mat_e->AddMultTranspose(x, y); }

   virtual void MultTranspose (const Vector & x, Vector & y) const
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

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new interior Face Integrator.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new boundary Face Integrator.
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi);

   /// Adds a LinearFESpaceIntegrator.
   void AddIntegrator(LinearFESpaceIntegrator *integ);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi,
                             Array<int> &bdr_marker);

   void operator=(const double a)
   {
      if (mat != NULL) { *mat = a; }
      if (mat_e != NULL) { *mat_e = a; }
   }

   /// Assembles the form i.e. sums over all domain/bdr integrators.
   void Assemble(int skip_zeros = 1);

   /** @brief Assembles the form with type Atype (one of enum AssemblyType) into
       an operator of type Otype (one of enum Operator::Type). */
   void AssembleForm(enum AssemblyType type = FullAssembly, int skip_zeros = 1);
   Operator *FinalizeForm();

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const
   { return fes->GetConformingProlongation(); }
   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return fes->GetConformingRestriction(); }

   /** Form the linear system A X = B, corresponding to the current bilinear
       form and b(.), by applying any necessary transformations such as:
       eliminating boundary conditions; applying conforming constraints for
       non-conforming AMR; static condensation; hybridization.

       The GridFunction-size vector x must contain the essential b.c. The
       BilinearForm and the LinearForm-size vector b must be assembled.

       The vector X is initialized with a suitable initial guess: when using
       hybridization, the vector X is set to zero; otherwise, the essential
       entries of X are set to the corresponding b.c. and all other entries are
       set to zero (copy_interior == 0) or copied from x (copy_interior != 0).

       This method can be called multiple times (with the same ess_tdof_list
       array) to initialize different right-hand sides and boundary condition
       values.

       After solving the linear system, the finite element solution x can be
       recovered by calling RecoverFEMSolution (with the same vectors X, b, and
       x).

       NOTE: If there are no transformations, X simply reuses the data of x. */
   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         Operator * &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         SparseMatrix &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /// Form the linear system matrix A, see FormLinearSystem for details.
   void FormSystemOperator(const Array<int> &ess_tdof_list, Operator * &Aoper);

   void FormSystemMatrix(const Array<int> &ess_tdof_list, SparseMatrix &A);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a GridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
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

   /** Eliminate essential boundary DOFs from the system. The array
       'bdr_attr_is_ess' marks boundary attributes that constitute the essential
       part of the boundary. If d == 0, the diagonal at the essential DOFs is
       set to 1.0, otherwise it is left the same. */
   void EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                             Vector &sol, Vector &rhs, int d = 0);

   void EliminateEssentialBC(const Array<int> &bdr_attr_is_ess, int d = 0);
   /// Perform elimination and set the diagonal entry to the given value
   void EliminateEssentialBCDiag(const Array<int> &bdr_attr_is_ess,
                                 double value);

   /// Eliminate the given vdofs. NOTE: here, vdofs is a list of DOFs.
   void EliminateVDofs(const Array<int> &vdofs, Vector &sol, Vector &rhs,
                       int d = 0);

   /** Eliminate the given vdofs storing the eliminated part internally; this
       method works in conjunction with EliminateVDofsInRHS and allows
       elimination of boundary conditions in multiple right-hand sides. In this
       method, vdofs is a list of DOFs. */
   void EliminateVDofs(const Array<int> &vdofs, int d = 0);

   /** Similar to EliminateVDofs but here ess_dofs is a marker
       (boolean) array on all vdofs (ess_dofs[i] < 0 is true). */
   void EliminateEssentialBCFromDofs(const Array<int> &ess_dofs, Vector &sol,
                                     Vector &rhs, int d = 0);

   /** Similar to EliminateVDofs but here ess_dofs is a marker
       (boolean) array on all vdofs (ess_dofs[i] < 0 is true). */
   void EliminateEssentialBCFromDofs(const Array<int> &ess_dofs, int d = 0);
   /// Perform elimination and set the diagonal entry to the given value
   void EliminateEssentialBCFromDofsDiag(const Array<int> &ess_dofs,
                                         double value);

   /** Use the stored eliminated part of the matrix (see EliminateVDofs) to
       modify r.h.s.; vdofs is a list of DOFs (non-directional, i.e. >= 0). */
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
   /// Sparse matrix to be associated with the form.
   SparseMatrix *mat;

   /// Generic operator associated with the form.
   Operator *oper;

   /// Operator type.
   enum Type oper_type;

   FiniteElementSpace *trial_fes, *test_fes;

   Array<BilinearFormIntegrator*> dom;
   Array<BilinearFormIntegrator*> bdr;
   Array<BilinearFormIntegrator*> skt; // trace face integrators
   Array<LinearFESpaceIntegrator*> fesi;

public:

   MixedBilinearForm (FiniteElementSpace *tr_fes,
                      FiniteElementSpace *te_fes);

   virtual double& Elem (int i, int j);

   virtual const double& Elem (int i, int j) const;

   virtual void Mult (const Vector & x, Vector & y) const;

   virtual void AddMult (const Vector & x, Vector & y,
                         const double a = 1.0) const;

   virtual void AddMultTranspose (const Vector & x, Vector & y,
                                  const double a = 1.0) const;

   virtual void MultTranspose (const Vector & x, Vector & y) const
   { y = 0.0; AddMultTranspose (x, y); }

   virtual MatrixInverse * Inverse() const;

   virtual void Finalize (int skip_zeros = 1);

   /** Extract the associated matrix as SparseMatrix blocks. The number of
       block rows and columns is given by the vector dimensions (vdim) of the
       test and trial spaces, respectively. */
   void GetBlocks(Array2D<SparseMatrix *> &blocks) const;

   const SparseMatrix &SpMat() const { return *mat; }
   SparseMatrix &SpMat() { return *mat; }
   SparseMatrix *LoseMat() { SparseMatrix *tmp = mat; mat = NULL; return tmp; }

   void AddDomainIntegrator (BilinearFormIntegrator * bfi);

   void AddBoundaryIntegrator (BilinearFormIntegrator * bfi);

   /** Add a trace face integrator. This type of integrator assembles terms
       over all faces of the mesh using the face FE from the trial space and the
       two adjacent volume FEs from the test space. */
   void AddTraceFaceIntegrator (BilinearFormIntegrator * bfi);

   /// Add an FESpaceIntegrator
   void AddIntegrator (LinearFESpaceIntegrator *integ);

   Array<BilinearFormIntegrator*> *GetDBFI() { return &dom; }

   Array<BilinearFormIntegrator*> *GetBBFI() { return &bdr; }

   Array<BilinearFormIntegrator*> *GetTFBFI() { return &skt; }

   Array<LinearFESpaceIntegrator*> *GetFESI() { return &fesi; }

   void operator= (const double a) { *mat = a; }

   void Assemble (int skip_zeros = 1);
   /** @brief Assembles the form with type Atype (one of enum AssemblyType) into
       an operator of type Otype (one of enum Operator::Type). */
   void AssembleForm(enum AssemblyType type = FullAssembly, int skip_zeros = 1);
   Operator *FinalizeForm();

   /** For partially conforming trial and/or test FE spaces, complete the
       assembly process by performing A := P2^t A P1 where A is the internal
       sparse matrix; P1 and P2 are the conforming prolongation matrices of the
       trial and test FE spaces, respectively. After this call the
       MixedBilinearForm becomes an operator on the conforming FE spaces. */
   void ConformingAssemble();

   void EliminateTrialDofs(Array<int> &bdr_attr_is_ess,
                           Vector &sol, Vector &rhs);

   void EliminateEssentialBCFromTrialDofs(Array<int> &marked_vdofs,
                                          Vector &sol, Vector &rhs);

   virtual void EliminateTestDofs(Array<int> &bdr_attr_is_ess);

   void Update();

   FiniteElementSpace *TrialFESpace() const { return trial_fes; }

   FiniteElementSpace *TestFESpace() const { return test_fes; }

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
public:
   DiscreteLinearOperator(FiniteElementSpace *domain_fes,
                          FiniteElementSpace *range_fes)
      : MixedBilinearForm(domain_fes, range_fes) { }

   void AddDomainInterpolator(DiscreteInterpolator *di)
   { AddDomainIntegrator(di); }

   void AddTraceFaceInterpolator(DiscreteInterpolator *di)
   { AddTraceFaceIntegrator(di); }

   Array<BilinearFormIntegrator*> *GetDI() { return &dom; }

   virtual void Assemble(int skip_zeros = 1);
};

}

#endif
