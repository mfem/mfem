// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Implementation of class HDGBilinearForm
//
// Contributed by: T. Horvath: Oakland University
//                 S. Rhebergen, A. Sivas: University of Waterloo

#ifndef MFEM_HDGBILINEARFORM
#define MFEM_HDGBILINEARFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "fespace.hpp"
#include "gridfunc.hpp"
#include "linearform.hpp"
#include "bilininteg.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif


namespace mfem
{

class HDGBilinearForm
{
protected:
   /// FE spaces on which the form lives.
   Array<FiniteElementSpace*> volume_fes, skeletal_fes;

   int NVolumeFES, NSkeletalFES;

   bool parallel;

   /// Sparse matrix to be assembled
   SparseMatrix *mat;

   /// Right hand side vector to be assembled.
   Vector *rhs_SC;

   /// Table that contains the faces for all elements
   Table *el_to_face;

   /// List that separates the interior edges from the shared edges
   Array<int> ess_dofs, vdofs1, vdofs2, vdofs3;

   /// Vectors to store A and B, the corresponding offsets and the number
   /// of elements on which A and B will be stored
   Array<int> A_offsets, B_offsets;
   double *A_data, *B_data;
   int elements_A, elements_B;

   /// HDG Integrators
   Array<BilinearFormIntegrator*> hdg_dbfi;
   Array<BilinearFormIntegrator*> hdg_fbfi;

   /// Dense matrices to be used for computing the integrals
   DenseMatrix elemmat1, elemmat2, elemmat3, elemmat4;

   // may be used in the construction of derived classes
   HDGBilinearForm()
   {
      for (int i =0; i<NVolumeFES; i++)
      {
         delete volume_fes[i];
      }
      for (int i =0; i<NSkeletalFES; i++)
      {
         delete skeletal_fes[i];
      }
      NVolumeFES = 0;
      NSkeletalFES = 0;
      volume_fes = NULL; skeletal_fes = NULL;
      parallel = false;
      mat = NULL;
      rhs_SC = NULL;
      el_to_face = NULL;
      A_data = NULL; B_data = NULL;
      elements_A = elements_B = 0;
   }

public:
   /// Creates bilinear form associated with FE spaces *_fes1 and _fes2.
   HDGBilinearForm(Array<FiniteElementSpace*> &_volume_fes,
                   Array<FiniteElementSpace*> &_skeletal_fes,
                   bool _parallel = false);

   // Advection-reaction test case without FES arrays
   HDGBilinearForm(FiniteElementSpace *_fes1,
                   FiniteElementSpace *_fes2,
                   bool _parallel = false);

   // Diffusion test case without FES arrays
   HDGBilinearForm(FiniteElementSpace *_fes1,
                   FiniteElementSpace *_fes2,
                   FiniteElementSpace *_fes3,
                   bool _parallel = false);

   // Arrays of the HDG domain integrators
   Array<BilinearFormIntegrator*> *GetHDG_DBFI() { return &hdg_dbfi; }

   // Arrays of the HDG face integrators
   Array<BilinearFormIntegrator*> *GetHDG_FBFI() { return &hdg_fbfi; }

   /// Finalizes the matrix
   virtual void Finalize(int skip_zeros = 1);

   /// Returns a reference to the sparse matrix
   const SparseMatrix *SpMatSC() const
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return mat;
   }

   void GetInteriorVDofs(int i, Array<int> &vdofs) const;

   void GetInteriorSubVector(const Array<GridFunction*> &rhs_vector,
                             int i, int ndof, Vector &SubVector) const;

   void SetInteriorSubVector(Array<GridFunction*> &sol_gridfunctions,
                             int i, int ndof, Vector &SubVector);

   void GetFaceVDofs(int face, Array<int> &vdofs) const;

   SparseMatrix *SpMatSC()
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return mat;
   }

   /// Returns a reference to the right hand side vector
   const Vector *VectorSC() const
   {
      return rhs_SC;
   }

   Vector *VectorSC()
   {
      return rhs_SC;
   }

   /// Adds new HDG Integrators.
   void AddHDGDomainIntegrator(BilinearFormIntegrator *bfi);

   void AddHDGFaceIntegrator(BilinearFormIntegrator *bfi);

   /// Allocates the vectors for the part of A and B that will be stored
   void Allocate(const Array<int> &bdr_attr_is_ess,
                 const double memA = 0.0, const double memB = 0.0);

   /// Assembles the Schur complement
   void AssembleSC(Array<GridFunction*> &rhs_F,
                   const Array<int> &bdr_attr_is_ess,
                   const GridFunction &sol,
                   const int elimBC,
                   const double memA = 0.0, const double memB = 0.0,
                   int skip_zeros = 1);

   /// Assembles the Schur complement - advection-reaction test case
   void AssembleSC(GridFunction *F,
                   const double memA = 0.0, const double memB = 0.0,
                   int skip_zeros = 1);

   /// Assembles the Schur complement - diffusion test case
   void AssembleSC(GridFunction *F1,
                   GridFunction *F2,
                   Array<int> &bdr_attr_is_ess,
                   GridFunction &sol,
                   const double memA = 0.0, const double memB = 0.0,
                   int skip_zeros = 1);

   void Eliminate_BC(const Array<int> &vdofs_e1, const int ndof_u,
                     const GridFunction &sol, Vector *rhs_F, Vector *rhs_G,
                     DenseMatrix *B_local, DenseMatrix *C_local, DenseMatrix *D_local);

   /// Computes domain based integrators
   void compute_domain_integrals(const int elem, DenseMatrix *A_local);

   /// Computes face based integrators
   void compute_face_integrals(const int elem,
                               const int edge,
                               //                               const int ishared,
                               const bool is_reconstruction,
                               DenseMatrix *A_local,
                               DenseMatrix *B_local,
                               DenseMatrix *C_local,
                               DenseMatrix *D_local);

   /// Reconstructs u from the facet unknowns
   void Reconstruct(Array<GridFunction*> &F,
                    const GridFunction *ubar,
                    Array<GridFunction*> &u);

   /// Reconstructs u from the facet unknowns - advection-reaction test case
   void Reconstruct(GridFunction *F,
                    const GridFunction *ubar,
                    GridFunction *u);

   /// Reconstructs u and q from the facet unknowns - diffusion test case
   void Reconstruct(GridFunction *R,
                    GridFunction *F,
                    const GridFunction *ubar,
                    GridFunction *q,
                    GridFunction *u);

   /// Updates the spaces
   virtual void Update(FiniteElementSpace *nfes1 = NULL,
                       FiniteElementSpace *nfes2 = NULL);

   /// Destroys bilinear form.
   virtual ~HDGBilinearForm();

#ifdef MFEM_USE_MPI
   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   HypreParMatrix *ParallelAssembleSC() { return ParallelAssembleSC(mat); }
   /// Return the matrix m assembled on the true dofs, i.e. P^t A P
   HypreParMatrix *ParallelAssembleSC(SparseMatrix *m);

   /// Returns the rhs vector assembled on the true dofs
   HypreParVector *ParallelVectorSC();
#endif
};



}

#endif
