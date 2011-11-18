// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BILINEARFORM
#define MFEM_BILINEARFORM


/** Class for bilinear form - "Matrix" with asociated FE space and
    BLFIntegrators. */
class BilinearForm : public Matrix
{
protected:
   /// Sparse matrix to be associated with the form.
   SparseMatrix *mat;

   // Matrix used to eliminate b.c.
   SparseMatrix *mat_e;

   /// FE space on which the form lives.
   FiniteElementSpace *fes;

   int extern_bfs;

   /// Set of Domain Integrators to be applied.
   Array<BilinearFormIntegrator*> dbfi;

   /// Set of Boundary Integrators to be applied.
   Array<BilinearFormIntegrator*> bbfi;

   /// Set of interior face Integrators to be applied.
   Array<BilinearFormIntegrator*> fbfi;

   /// Set of boundary face Integrators to be applied.
   Array<BilinearFormIntegrator*> bfbfi;

   DenseMatrix elemmat;
   Array<int>  vdofs;

   DenseTensor *element_matrices;

   // may be used in the construction of derived classes
   BilinearForm() : Matrix (0)
   { fes = NULL; mat = mat_e = NULL; extern_bfs = 0; element_matrices = NULL; }

public:
   /// Creates bilinear form associated with FE space *f.
   BilinearForm (FiniteElementSpace * f);

   BilinearForm (FiniteElementSpace * f, BilinearForm * bf);

   Array<BilinearFormIntegrator*> *GetDBFI() { return &dbfi; }

   Array<BilinearFormIntegrator*> *GetBBFI() { return &bbfi; }

   Array<BilinearFormIntegrator*> *GetFBFI() { return &fbfi; }

   Array<BilinearFormIntegrator*> *GetBFBFI() { return &bfbfi; }

   const double &operator()(int i, int j) { return (*mat)(i,j); }

   /// Returns reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual const double &Elem(int i, int j) const;

   /// Matrix vector multiplication.
   virtual void Mult(const Vector &x, Vector &y) const;

   void FullMult(const Vector &x, Vector &y) const
   { mat->Mult(x, y); mat_e->AddMult(x, y); }

   virtual void AddMult(const Vector &x, Vector &y, const double a = 1.0) const
   { mat -> AddMult (x, y, a); }

   void FullAddMult(const Vector &x, Vector &y) const
   { mat->AddMult(x, y); mat_e->AddMult(x, y); }

   double InnerProduct(const Vector &x, const Vector &y) const
   { return mat->InnerProduct (x, y); }

   /// Returns a pointer to (approximation) of the matrix inverse.
   virtual MatrixInverse *Inverse() const;

   /// Finalizes the matrix initialization.
   virtual void Finalize(int skip_zeros = 1);

   /// Returns a reference to the sparse martix
   const SparseMatrix &SpMat() const { return *mat; }
   SparseMatrix &SpMat() { return *mat; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new interior Face Integrator.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new boundary Face Integrator.
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi);

   void operator=(const double a)
   { if (mat != NULL) *mat = a; if (mat_e != NULL) *mat_e = a; }

   /// Assembles the form i.e. sums over all domain/bdr integrators.
   void Assemble(int skip_zeros = 1);

   /// Compute and store internally all element matrices.
   void ComputeElementMatrices();

   /// Free the memory used by the element matrices.
   void FreeElementMatrices()
   { delete element_matrices; element_matrices = NULL; }

   void ComputeElementMatrix(int i, DenseMatrix &elmat);
   void AssembleElementMatrix(int i, const DenseMatrix &elmat,
                              Array<int> &vdofs, int skip_zeros = 1);

   /** If d == 0 the diagonal at the ess. b.c. is set to 1.0,
       otherwise leave it the same.      */
   void EliminateEssentialBC(Array<int> &bdr_attr_is_ess,
                             Vector &sol, Vector &rhs, int d = 0);

   /// Here, vdofs is a list of DOFs.
   void EliminateVDofs(Array<int> &vdofs, Vector &sol, Vector &rhs, int d = 0);

   /** Eliminate the given vdofs storing the eliminated part internally;
       vdofs is a list of DOFs. */
   void EliminateVDofs(Array<int> &vdofs, int d = 0);

   /** Use the stored eliminated part of the matrix to modify r.h.s.;
       vdofs is a list of DOFs (non-directional, i.e. >= 0). */
   void EliminateVDofsInRHS(Array<int> &vdofs, const Vector &x, Vector &b);

   double FullInnerProduct(const Vector &x, const Vector &y) const
   { return mat->InnerProduct(x, y) + mat_e->InnerProduct(x, y); }

   void EliminateEssentialBC(Array<int> &bdr_attr_is_ess, int d = 0);

   /** Similar to EliminateVDofs but here ess_dofs is a marker
       (boolean) array on all vdofs (ess_dofs[i] < 0 is true). */
   void EliminateEssentialBCFromDofs(Array<int> &ess_dofs, Vector &sol,
                                     Vector &rhs, int d = 0);

   /** Similar to EliminateVDofs but here ess_dofs is a marker
       (boolean) array on all vdofs (ess_dofs[i] < 0 is true). */
   void EliminateEssentialBCFromDofs(Array<int> &ess_dofs, int d = 0);

   void Update(FiniteElementSpace *nfes = NULL);

   FiniteElementSpace *GetFES() { return fes; }

   /// Destroys bilinear form.
   virtual ~BilinearForm();
};

/**
   Class for assembling of bilinear forms a(u,v) defined on different
   trial and test spaces. The assembled matrix A is such that

   a(u,v) = V^t A U

   where U and V are the vectors representing the function u and v,
   respectively.  The first argument, u, of a(,) is in the trial space
   and the second argument, v, is in the test space. Thus,

   # of rows of A = dimension of the test space and
   # of cols of A = dimension of the trial space.

   Both trial and test spaces should be defined on the same mesh.
*/
class MixedBilinearForm : public Matrix
{
protected:
   SparseMatrix *mat;

   FiniteElementSpace *trial_fes, *test_fes;

   Array<BilinearFormIntegrator*> dom;
   Array<BilinearFormIntegrator*> bdr;

   int width;

public:
   MixedBilinearForm (FiniteElementSpace *tr_fes,
                      FiniteElementSpace *te_fes);

   int Width() const { return width; };

   virtual double& Elem (int i, int j);

   virtual const double& Elem (int i, int j) const;

   virtual void Mult (const Vector & x, Vector & y) const;

   virtual void AddMult (const Vector & x, Vector & y,
                         const double a = 1.0) const;

   virtual void AddMultTranspose (const Vector & x, Vector & y,
                                  const double a = 1.0) const;

   virtual void MultTranspose (const Vector & x, Vector & y) const
   { y = 0.0; AddMultTranspose (x, y); };

   virtual MatrixInverse * Inverse() const;

   virtual void Finalize (int skip_zeros = 1);

   /** Extract the associated matrix as SparseMatrix blocks. The number of
       block rows and columns is given by the vector dimensions (vdim) of the
       test and trial spaces, respectively. */
   void GetBlocks(Array2D<SparseMatrix *> &blocks) const;

   const SparseMatrix &SpMat() const { return *mat; };
   SparseMatrix &SpMat() { return *mat; };

   void AddDomainIntegrator (BilinearFormIntegrator * bfi);

   void AddBoundaryIntegrator (BilinearFormIntegrator * bfi);

   void operator= (const double a) { *mat = a; }

   virtual void Assemble (int skip_zeros = 1);

   void EliminateTrialDofs (Array<int> &bdr_attr_is_ess,
                            Vector &sol, Vector &rhs);

   virtual void EliminateTestDofs (Array<int> &bdr_attr_is_ess);

   void Update();

   virtual ~MixedBilinearForm();
};


/**
   Class for constructing the matrix representation of a linear operator,
   v = L u, from one FiniteElementSpace (domain) to another FiniteElementSpace
   (range). The constructed matrix A is such that

   V = A U

   where U and V are the vectors of degrees of freedom representing the
   functions u and v, respectively. The dimensions of A are

   number of rows of A = dimension of the range space and
   number of cols of A = dimension of the domain space.

   This class is very similar to MixedBilinearForm. One difference is that
   the linear operator L is defined using a special kind of
   BilinearFormIntegrator (we reuse its functionality instead of defining a
   new class). The other difference with the MixedBilinearForm class is that
   the "assembly" process overwrites the global matrix entries using the
   local element matrices instead of adding them.

   Note that if we define the bilinear form b(u,v) := (Lu,v) using an inner
   product in the range space, then its matrix representation, B, is

   B = M A, (since V^t B U = b(u,v) = (Lu,v) = V^t M A U)

   where M denotes the mass matrix for the inner product in the range space:
   V1^t M V2 = (v1,v2). Similarly, if c(u,w) := (Lu,Lw) then

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

   virtual void Assemble(int skip_zeros = 1);
};

#endif
