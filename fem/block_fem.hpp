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

class BlockFiniteElementSpace
{
private:
   int nblocks;
   FiniteElementSpace ** fes;
   bool owns_fes;
  
protected:
   inline void CheckIndex(int index) const
   {
      MFEM_ASSERT(index >= 0 && index < nblocks,
                  "Out of bounds access: " << index << ", size = " << nblocks);
   }

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

  ~BlockFiniteElementSpace();

  /** The following methods might be useful but let's hold off and see
      which are actually needed. */
  
   /// Returns the mesh
  // inline Mesh *GetMesh() const { return mesh; }

   /// Returns number of degrees of freedom.
  // inline int GetNDofs() const { return ndofs; }

   /// Return the number of vector dofs, i.e. GetNDofs() x GetVDim().
  // inline int GetVSize() const { return vdim * ndofs; }

   /// Return the number of vector true (conforming) dofs.
  // virtual int GetTrueVSize() const { return GetConformingVSize(); }

   /// Returns the number of conforming ("true") degrees of freedom
   /// (if the space is on a nonconforming mesh with hanging nodes).
  // int GetNConformingDofs() const;

  // int GetConformingVSize() const { return vdim * GetNConformingDofs(); }

  virtual void Update();
  
  FiniteElementSpace * FESpace(int index)
  { CheckIndex(index); return fes[index]; }
  const FiniteElementSpace * FESpace(int index) const
  { CheckIndex(index); return fes[index]; }
};
  
/// Class for grid functions defined in blocks
class BlockGridFunction : public Vector
{
private:

   GridFunction * gfr_;
   GridFunction * gfi_;

protected:
   void Destroy() { delete gfr_; delete gfi_; }

public:

   /* @brief Construct a BlockGridFunction associated with the
      FiniteElementSpace @a *f. */
   BlockGridFunction(FiniteElementSpace *f);

   void Update();

   /// Assign constant values to the BlockGridFunction data.
   BlockGridFunction &operator=(const std::complex<double> & value)
   { *gfr_ = value.real(); *gfi_ = value.imag(); return *this; }

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

   FiniteElementSpace *FESpace() { return gfr_->FESpace(); }
   const FiniteElementSpace *FESpace() const { return gfr_->FESpace(); }

   GridFunction & real() { return *gfr_; }
   GridFunction & imag() { return *gfi_; }
   const GridFunction & real() const { return *gfr_; }
   const GridFunction & imag() const { return *gfi_; }

   /// Destroys grid function.
   virtual ~BlockGridFunction() { Destroy(); }

};

class BlockLinearForm : public Vector
{
private:
   BlockOperator::Convention conv_;

protected:
   LinearForm * lfr_;
   LinearForm * lfi_;

   // HYPRE_Int * tdof_offsets_;

public:

   BlockLinearForm(FiniteElementSpace *fes,
                     BlockOperator::Convention
                     convention = BlockOperator::HERMITIAN);

   virtual ~BlockLinearForm();

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                            LinearFormIntegrator *lfi_imag);

   FiniteElementSpace *FESpace() const { return lfr_->FESpace(); }

   LinearForm & real() { return *lfr_; }
   LinearForm & imag() { return *lfi_; }
   const LinearForm & real() const { return *lfr_; }
   const LinearForm & imag() const { return *lfi_; }

   void Update();
   void Update(FiniteElementSpace *f);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   std::complex<double> operator()(const BlockGridFunction &gf) const;

};

// Class for block-structured bilinear forms
class BlockBilinearForm
{
private:
   //protected:
   BilinearForm *blfr_;
   BilinearForm *blfi_;

public:
   BlockBilinearForm(FiniteElementSpace *fes);

   BilinearForm & real() { return *blfr_; }
   BilinearForm & imag() { return *blfi_; }
   const BilinearForm & real() const { return *blfr_; }
   const BilinearForm & imag() const { return *blfi_; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                            BilinearFormIntegrator *bfi_imag);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag);

   /// Adds new Boundary Integrator, restricted to specific boundary attributes.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag,
                              Array<int> &bdr_marker);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   BlockSparseMatrix *AssembleCompSpMat();

   /// Return the parallel FE space associated with the ParBilinearForm.
   FiniteElementSpace *FESpace() const { return blfr_->FESpace(); }

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   virtual void Update(FiniteElementSpace *nfes = NULL);

   virtual ~BlockBilinearForm();
};

#ifdef MFEM_USE_MPI

/// Class for complex-valued grid function - Vector with associated FE space.
class ParBlockGridFunction : public Vector
{
private:

   ParGridFunction * pgfr_;
   ParGridFunction * pgfi_;

protected:
   void Destroy() { delete pgfr_; delete pgfi_; }

public:

   /* @brief Construct a ParBlockGridFunction associated with the
      ParFiniteElementSpace @a *f. */
   ParBlockGridFunction(ParFiniteElementSpace *pf);

   void Update();

   /// Assign constant values to the ParBlockGridFunction data.
   ParBlockGridFunction &operator=(const std::complex<double> & value)
   { *pgfr_ = value.real(); *pgfi_ = value.imag(); return *this; }

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

   void Distribute(const Vector *tv);
   void Distribute(const Vector &tv) { Distribute(&tv); }

   /// Returns the vector restricted to the true dofs.
   void ParallelProject(Vector &tv) const;

   FiniteElementSpace *FESpace() { return pgfr_->FESpace(); }
   const FiniteElementSpace *FESpace() const { return pgfr_->FESpace(); }

   ParGridFunction & real() { return *pgfr_; }
   ParGridFunction & imag() { return *pgfi_; }
   const ParGridFunction & real() const { return *pgfr_; }
   const ParGridFunction & imag() const { return *pgfi_; }

   /// Destroys grid function.
   virtual ~ParBlockGridFunction() { Destroy(); }

};

class ParBlockLinearForm : public Vector
{
private:
   BlockOperator::Convention conv_;

protected:
   ParLinearForm * plfr_;
   ParLinearForm * plfi_;

   HYPRE_Int * tdof_offsets_;

public:

   ParBlockLinearForm(ParFiniteElementSpace *pf,
                        BlockOperator::Convention
                        convention = BlockOperator::HERMITIAN);

   virtual ~ParBlockLinearForm();

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                            LinearFormIntegrator *lfi_imag);

   ParFiniteElementSpace *ParFESpace() const { return plfr_->ParFESpace(); }

   ParLinearForm & real() { return *plfr_; }
   ParLinearForm & imag() { return *plfi_; }
   const ParLinearForm & real() const { return *plfr_; }
   const ParLinearForm & imag() const { return *plfi_; }

   void Update(ParFiniteElementSpace *pf = NULL);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Assemble the vector on the true dofs, i.e. P^t v.
   void ParallelAssemble(Vector &tv);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   HypreParVector *ParallelAssemble();

   std::complex<double> operator()(const ParBlockGridFunction &gf) const;

};

// Class for parallel block-structured bilinear forms
class ParBlockBilinearForm
{
private:
   BlockOperator::Convention conv_;

   //protected:
   ParBilinearForm *pblfr_;
   ParBilinearForm *pblfi_;

public:
   ParBlockBilinearForm(ParFiniteElementSpace *pf,
                       BlockOperator::Convention
                       convention = BlockOperator::HERMITIAN);

   BlockOperator::Convention GetConvention() const { return conv_; }
   void SetConvention(const BlockOperator::Convention &
                      convention) { conv_  = convention; }

   ParBilinearForm & real() { return *pblfr_; }
   ParBilinearForm & imag() { return *pblfi_; }
   const ParBilinearForm & real() const { return *pblfr_; }
   const ParBilinearForm & imag() const { return *pblfi_; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                            BilinearFormIntegrator *bfi_imag);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag);

   /// Adds new Boundary Integrator, restricted to specific boundary attributes.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag,
                              Array<int> &bdr_marker);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   BlockHypreParMatrix *ParallelAssemble();

   /// Return the parallel FE space associated with the ParBilinearForm.
   ParFiniteElementSpace *ParFESpace() const { return pblfr_->ParFESpace(); }

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   virtual void Update(FiniteElementSpace *nfes = NULL);

   virtual ~ParBlockBilinearForm();
};

#endif // MFEM_USE_MPI

}

#endif // MFEM_BLOCK_FEM
