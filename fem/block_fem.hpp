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

#ifndef MFEM_BLOCK_FEM
#define MFEM_BLOCK_FEM

#include "gridfunc.hpp"
#include "linearform.hpp"
#include "bilinearform.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#include "plinearform.hpp"
#include "pbilinearform.hpp"
#endif

namespace mfem
{

class BlockObject
{
protected:
   int nblocks;

   BlockObject(int num_blocks) : nblocks(num_blocks) {}

   inline void CheckIndex(int index) const
   {
      MFEM_ASSERT(index >= 0 && index < nblocks,
                  "Out of bounds access: " << index << ", size = " << nblocks);
   }

public:

   inline int GetNBlocks() const { return nblocks; }

};

/// Tensor Product Block Object
class TPBlockObject
{
protected:
   int nrows;
   int ncols;
   int nblocks;

   // Square tensor product
   TPBlockObject(int size)
      : nrows(size), ncols(size), nblocks(size * size) {}
   // Rectangular tensor product
   TPBlockObject(int num_rows, int num_cols)
      : nrows(num_rows), ncols(num_cols), nblocks(nrows * ncols) {}

   inline int CheckIndex(int r, int c) const
   {
      MFEM_ASSERT(r >= 0 && r < nrows && c >= 0 && c < ncols,
                  "Out of bounds access: (" << r << "," << c
                  << "), dimensions = " << nrows << " x " << ncols);
      return r * ncols + c;
   }

public:
   inline int GetNBlocks() const { return nblocks; }
   inline int GetNRows() const { return nrows; }
   inline int GetNColumns() const { return ncols; }

};

class BlockFiniteElementSpace : public BlockObject
{
private:
   Array<int> vsize_offsets;
   FiniteElementSpace **fes;
   bool owns_fes;

public:
   /** Construct FiniteElementSpaces with different finite element
       collections but the same vdim and ordering.  @note The internally
       constructed FiniteElementSpace objects will be owned, and free'd, by
       the BlockFiniteElementSpace object. */
   BlockFiniteElementSpace(int num_blocks, Mesh *mesh,
                           FiniteElementCollection ** fec,
                           int vdim = 1, int ordering = Ordering::byNODES);

   /** Construct from an arbitrary set of finite element spaces.
       @note The FiniteElementSpace objects must be maintained by the caller. */
   BlockFiniteElementSpace(int num_blocks, FiniteElementSpace ** fespace);

   virtual ~BlockFiniteElementSpace();

   /** The following methods might be useful but let's hold off and see
       which are actually needed. */

   /// Returns the mesh
   // inline Mesh *GetMesh() const { return mesh; }

   /// Returns number of degrees of freedom.
   // inline int GetNDofs() const { return ndofs; }

   /// Return the number of vector dofs, i.e. GetNDofs() x GetVDim().
   inline int GetVSize() const { return vsize_offsets[nblocks]; }

   /// Return the number of vector true (conforming) dofs.
   // virtual int GetTrueVSize() const { return GetConformingVSize(); }

   /// Returns the number of conforming ("true") degrees of freedom
   /// (if the space is on a nonconforming mesh with hanging nodes).
   // int GetNConformingDofs() const;

   // int GetConformingVSize() const { return vdim * GetNConformingDofs(); }

   void Update();

   inline int GetVSizeOffset(int index) const
   { return vsize_offsets[index]; }

   FiniteElementSpace & GetBlock(int index)
   { CheckIndex(index); return *fes[index]; }
   const FiniteElementSpace & GetBlock(int index) const
   { CheckIndex(index); return *fes[index]; }
};

/// Class for grid functions defined in blocks
class BlockGridFunction : public BlockObject, public Vector
{
private:
   BlockFiniteElementSpace *fes;
   GridFunction **gf;

public:

   /* @brief Construct a BlockGridFunction associated with the
      FiniteElementSpace @a *f. */
   BlockGridFunction(BlockFiniteElementSpace *f);

   void Update();

   /// Assign constant values to the BlockGridFunction data.
   BlockGridFunction &operator=(double value);

   /*
    virtual void ProjectCoefficient(Coefficient &real_coeff,
                                    Coefficient &imag_coeff);
    virtual void ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                    VectorCoefficient &imag_vcoeff);

    virtual void ProjectBdrCoefficient(Coefficient &real_coeff,
                                       Coefficient &imag_coeff,
                                       Array<int> &attr);
    virtual void ProjectBdrCoefficientNormal(VectorCoefficient &real_coeff,
                                             VectorCoefficient &imag_coeff,
                                             Array<int> &attr);
    virtual void ProjectBdrCoefficientTangent(VectorCoefficient &real_coeff,
                                              VectorCoefficient &imag_coeff,
                                              Array<int> &attr);
   */
   BlockFiniteElementSpace *FESpace() { return fes; }
   const BlockFiniteElementSpace *FESpace() const { return fes; }
   /*
    GridFunction & real() { return *gfr_; }
    GridFunction & imag() { return *gfi_; }
    const GridFunction & real() const { return *gfr_; }
    const GridFunction & imag() const { return *gfi_; }
   */
   GridFunction & GetBlock(int index)
   { CheckIndex(index); return *gf[index]; }
   const GridFunction & GetBlock(int index) const
   { CheckIndex(index); return *gf[index]; }

   /// Destroys grid function.
   virtual ~BlockGridFunction();
};

class BlockLinearForm : public BlockObject, public Vector
{
private:
   BlockFiniteElementSpace *fes;
   LinearForm **lf;

public:

   BlockLinearForm(BlockFiniteElementSpace *f);

   virtual ~BlockLinearForm();

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(int index, LinearFormIntegrator *lfi);

   BlockFiniteElementSpace *FESpace() const { return fes; }
   /*
    LinearForm & real() { return *lfr_; }
    LinearForm & imag() { return *lfi_; }
    const LinearForm & real() const { return *lfr_; }
    const LinearForm & imag() const { return *lfi_; }
   */
   LinearForm & GetBlock(int index)
   { CheckIndex(index); return *lf[index]; }
   const LinearForm & GetBlock(int index) const
   { CheckIndex(index); return *lf[index]; }

   void Update();
   void Update(BlockFiniteElementSpace *f);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   double operator()(const BlockGridFunction &gf) const;

};

// Class for block-structured bilinear forms
class BlockBilinearForm : public TPBlockObject
{
private:
   BlockFiniteElementSpace *trial_fes; // Row space
   BlockFiniteElementSpace *test_fes; // Col space
   Matrix **blf;
   bool *mixed;
   bool sym;

   void initBilinearForm(int r, int c);

public:
   BlockBilinearForm(BlockFiniteElementSpace *fes, bool symmetric = false);
   BlockBilinearForm(BlockFiniteElementSpace *trial_fes,
                     BlockFiniteElementSpace *test_fes);
   /*
    BilinearForm & real() { return *blfr_; }
    BilinearForm & imag() { return *blfi_; }
    const BilinearForm & real() const { return *blfr_; }
    const BilinearForm & imag() const { return *blfi_; }
   */
   bool isBlockMixed(int r, int c) const
   { int index = CheckIndex(r, c); return mixed[index]; }

   /** The following methods can return NULL if the requested block is empty */
   BilinearForm * GetSquareBlock(int r, int c)
   { int index = CheckIndex(r, c); return (BilinearForm*)blf[index]; }
   const BilinearForm * GetSquareBlock(int r, int c) const
   { int index = CheckIndex(r, c); return (BilinearForm*)blf[index]; }

   MixedBilinearForm * GetMixedBlock(int r, int c)
   { int index = CheckIndex(r, c); return (MixedBilinearForm*)blf[index]; }
   const MixedBilinearForm * GetMixedBlock(int r, int c) const
   { int index = CheckIndex(r, c); return (MixedBilinearForm*)blf[index]; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(int r, int c, BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(int r, int c, BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator, restricted to specific boundary attributes.
   void AddBoundaryIntegrator(int r, int c, BilinearFormIntegrator *bfi,
                              Array<int> &bdr_marker);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   // BlockSparseMatrix *AssembleCompSpMat();

   /// Return the parallel FE space associated with the ParBilinearForm.
   BlockFiniteElementSpace *TrialFESpace() const { return trial_fes; }
   BlockFiniteElementSpace *RowFESpace() const { return trial_fes; }
   BlockFiniteElementSpace *TestFESpace() const { return test_fes; }
   BlockFiniteElementSpace *ColumnFESpace() const { return test_fes; }

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   void Update(BlockFiniteElementSpace *nfes = NULL);

   virtual ~BlockBilinearForm();
};

#ifdef MFEM_USE_MPI

class ParBlockFiniteElementSpace : public BlockObject
{
private:
   Array<int> vsize_offsets;
   Array<int> truev_offsets;
   ParFiniteElementSpace **pfes;
   bool owns_pfes;

public:
   /** Construct FiniteElementSpaces with different finite element
       collections but the same vdim and ordering.  @note The internally
       constructed FiniteElementSpace objects will be owned, and free'd, by
       the BlockFiniteElementSpace object. */
   ParBlockFiniteElementSpace(int num_blocks, ParMesh *mesh,
                              FiniteElementCollection ** fec,
                              int vdim = 1, int ordering = Ordering::byNODES);

   /** Construct from an arbitrary set of finite element spaces.
       @note The FiniteElementSpace objects must be maintained by the caller. */
   ParBlockFiniteElementSpace(int num_blocks, ParFiniteElementSpace ** fespace);

   virtual ~ParBlockFiniteElementSpace();

   /** The following methods might be useful but let's hold off and see
       which are actually needed. */

   /// Returns the mesh
   // inline Mesh *GetMesh() const { return mesh; }

   /// Returns number of degrees of freedom.
   // inline int GetNDofs() const { return ndofs; }

   /// Return the number of vector dofs, i.e. GetNDofs() x GetVDim().
   inline int GetVSize() const { return vsize_offsets[nblocks]; }

   /// Return the number of vector true (conforming) dofs.
   // virtual int GetTrueVSize() const { return GetConformingVSize(); }

   /// Returns the number of conforming ("true") degrees of freedom
   /// (if the space is on a nonconforming mesh with hanging nodes).
   // int GetNConformingDofs() const;

   // int GetConformingVSize() const { return vdim * GetNConformingDofs(); }

   void Update();

   inline int GetVSizeOffset(int index) const
   { return vsize_offsets[index]; }

   inline int GetTrueVSizeOffset(int index) const
   { return truev_offsets[index]; }

   ParFiniteElementSpace & GetBlock(int index)
   { CheckIndex(index); return *pfes[index]; }
   const ParFiniteElementSpace & GetBlock(int index) const
   { CheckIndex(index); return *pfes[index]; }
};

/// Class for complex-valued grid function - Vector with associated FE space.
class ParBlockGridFunction : public BlockObject, public Vector
{
private:
   ParBlockFiniteElementSpace *pfes;
   ParGridFunction **pgf;

protected:
   void Destroy();

public:

   /* @brief Construct a ParBlockGridFunction associated with the
      ParFiniteElementSpace @a *f. */
   ParBlockGridFunction(ParBlockFiniteElementSpace *pf);

   void Update();

   /// Assign constant values to the ParBlockGridFunction data.
   ParBlockGridFunction &operator=(double value);

   /*
    virtual void ProjectCoefficient(Coefficient &real_coeff,
                                    Coefficient &imag_coeff);
    virtual void ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                    VectorCoefficient &imag_vcoeff);

    virtual void ProjectBdrCoefficient(Coefficient &real_coeff,
                                       Coefficient &imag_coeff,
                                       Array<int> &attr);
    virtual void ProjectBdrCoefficientNormal(VectorCoefficient &real_coeff,
                                             VectorCoefficient &imag_coeff,
                                             Array<int> &attr);
    virtual void ProjectBdrCoefficientTangent(VectorCoefficient &real_coeff,
                                              VectorCoefficient &imag_coeff,
                                              Array<int> &attr);
   */
   void Distribute(const Vector *tv);
   void Distribute(const Vector &tv) { Distribute(&tv); }

   /// Returns the vector restricted to the true dofs.
   void ParallelProject(Vector &tv) const;

   ParBlockFiniteElementSpace *ParFESpace() { return pfes; }
   const ParBlockFiniteElementSpace *ParFESpace() const { return pfes; }
   /*
    ParGridFunction & real() { return *pgfr_; }
    ParGridFunction & imag() { return *pgfi_; }
    const ParGridFunction & real() const { return *pgfr_; }
    const ParGridFunction & imag() const { return *pgfi_; }
   */
   ParGridFunction & GetBlock(int index)
   { CheckIndex(index); return *pgf[index]; }
   const ParGridFunction & GetBlock(int index) const
   { CheckIndex(index); return *pgf[index]; }

   /// Destroys grid function.
   virtual ~ParBlockGridFunction() { Destroy(); }

};

class ParBlockLinearForm : public BlockObject, public Vector
{
private:
   ParBlockFiniteElementSpace *pfes;
   ParLinearForm ** plf;

public:

   ParBlockLinearForm(ParBlockFiniteElementSpace *pf);

   virtual ~ParBlockLinearForm();

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(int index, LinearFormIntegrator *lfi);

   ParBlockFiniteElementSpace *ParFESpace() const { return pfes; }
   /*
    ParLinearForm & real() { return *plfr_; }
    ParLinearForm & imag() { return *plfi_; }
    const ParLinearForm & real() const { return *plfr_; }
    const ParLinearForm & imag() const { return *plfi_; }
   */
   ParLinearForm & GetBlock(int index)
   { CheckIndex(index); return *plf[index]; }
   const ParLinearForm & GetBlock(int index) const
   { CheckIndex(index); return *plf[index]; }

   void Update(ParBlockFiniteElementSpace *pf = NULL);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Assemble the vector on the true dofs, i.e. P^t v.
   void ParallelAssemble(Vector &tv);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   BlockVector *ParallelAssemble();

   double operator()(const ParBlockGridFunction &gf) const;

};

// Class for parallel block-structured bilinear forms
class ParBlockBilinearForm : public TPBlockObject
{
private:
   ParBlockFiniteElementSpace *trial_pfes; // Row space
   ParBlockFiniteElementSpace *test_pfes; // Col space
   Matrix **pblf;
   bool *mixed;
   bool sym;

   void initParBilinearForm(int r, int c);

public:
   ParBlockBilinearForm(ParBlockFiniteElementSpace *pf, bool symmetric = false);
   ParBlockBilinearForm(ParBlockFiniteElementSpace *trial_pf,
                        ParBlockFiniteElementSpace *test_pf);
   /*
    ParBilinearForm & real() { return *pblf[index]; }
    ParBilinearForm & imag() { return *pblfi_; }
    const ParBilinearForm & real() const { return *pblfr_; }
    const ParBilinearForm & imag() const { return *pblfi_; }
   */
   bool isBlockMixed(int r, int c) const
   { int index = CheckIndex(r, c); return mixed[index]; }
  
   /** The following methods can return NULL if the requested block is empty */
   ParBilinearForm * GetSquareBlock(int r, int c)
   { int index = CheckIndex(r, c); return (ParBilinearForm*)pblf[index]; }
   const ParBilinearForm * GetSquareBlock(int r, int c) const
   { int index = CheckIndex(r, c); return (ParBilinearForm*)pblf[index]; }

   ParMixedBilinearForm * GetMixedBlock(int r, int c)
   { int index = CheckIndex(r, c); return (ParMixedBilinearForm*)pblf[index]; }
   const ParMixedBilinearForm * GetMixedBlock(int r, int c) const
   { int index = CheckIndex(r, c); return (ParMixedBilinearForm*)pblf[index]; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(int r, int c, BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(int r, int c, BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator, restricted to specific boundary attributes.
   void AddBoundaryIntegrator(int r, int c, BilinearFormIntegrator *bfi,
                              Array<int> &bdr_marker);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   BlockOperator *ParallelAssemble();

   /// Return the parallel FE space associated with the ParBilinearForm.
   ParBlockFiniteElementSpace *TrialParFESpace() const { return trial_pfes; }
   ParBlockFiniteElementSpace *RowParFESpace() const { return trial_pfes; }
   ParBlockFiniteElementSpace *TestParFESpace() const { return test_pfes; }
   ParBlockFiniteElementSpace *ColumnParFESpace() const { return test_pfes; }

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   void Update(ParBlockFiniteElementSpace *nfes = NULL);

   virtual ~ParBlockBilinearForm();
};

#endif // MFEM_USE_MPI

}

#endif // MFEM_BLOCK_FEM
