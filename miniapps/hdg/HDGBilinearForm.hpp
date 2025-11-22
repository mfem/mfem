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
//
// Implementation of class HDGBilinearForm
//
// Contributed by: T. Horvath: Oakland University
//                 S. Rhebergen, A. Sivas: University of Waterloo

#ifndef MFEM_HDGBILINEARFORM
#define MFEM_HDGBILINEARFORM

#include "../../config/config.hpp"
#include "../../linalg/linalg.hpp"
#include "../../fem/fespace.hpp"
#include "../../fem/gridfunc.hpp"
#include "../../fem/linearform.hpp"
#include "../../fem/bilininteg.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

using namespace std;
using namespace mfem;

class HDGBilinearForm
{
protected:
   /// FE spaces on which the form lives.
   Array<FiniteElementSpace*> volume_fes, skeletal_fes;

   int NVolumeFES, NSkeletalFES;

   bool parallel;

   /// Sparse matrix to be assembled
   Array<SparseMatrix*> mat;

   /// Right hand side vector to be assembled.
   Array<Vector*> rhs_SC;

   /// Table that contains the faces for all elements
   Table *el_to_face;

   /// List that separates the interior edges from the shared edges
   Array<int> ess_dofs, Edge_to_SharedEdge;

   /// HDG Integrators
   Array<BilinearFormIntegrator*> hdg_dbfi;
   Array<BilinearFormIntegrator*> hdg_fbfi;

   /// Dense matrices to be used for computing the integrals
   DenseMatrix elemmat1, elemmat2, elemmat3, elemmat4;

   /// Vectors to store A and B, the corresponding offsets and the number
   /// of elements on which A and B will be stored
   Array<int> A_offsets, B_offsets;
   real_t *A_data, *B_data;
   int elements_A, elements_B;


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
         delete rhs_SC[i];
      }
      for (int i =0; i<NSkeletalFES*NSkeletalFES; i++)
      {
         delete mat[i];
      }
      NVolumeFES = 0;
      NSkeletalFES = 0;
      volume_fes = NULL;
      skeletal_fes = NULL;
      parallel = false;
      el_to_face = NULL;
      A_data = NULL;
      B_data = NULL;
      elements_A = elements_B = 0;
   }

public:
   /// Creates bilinear form associated with FE spaces *_fes1 and _fes2.
   HDGBilinearForm(Array<FiniteElementSpace*> &_fes1,
                   Array<FiniteElementSpace*> &_fes2,
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
   Array<BilinearFormIntegrator*> *GetHDG_DBFI()
   {
      return &hdg_dbfi;
   }

   // Arrays of the HDG face integrators
   Array<BilinearFormIntegrator*> *GetHDG_FBFI()
   {
      return &hdg_fbfi;
   }

   /// Finalizes the matrix
   virtual void Finalize(int skip_zeros = 1);

   // Gets all dofs for a given element (goes over all volume FES)
   void GetInteriorVDofs(int i, Array<int> &vdofs) const;

   // Gets the values of volume GFs at all dofs for a given element (goes over all volume FES)
   void GetInteriorSubVector(const Array<GridFunction*> &rhs_gridfunctions,
                             int i, int ndof, Vector &SubVector) const;

   // Gets the values of a skeletal GFs at all dofs for a given element (goes over all skeletal FES)
   void GetFaceSubVector(const Array<GridFunction*> &face_gridfunctions,
                         int i, int ndof, Vector &SubVector) const;

   // Sets the values of volume GFs at all dofs for a given element (goes over all volume FES)
   void SetInteriorSubVector(Array<GridFunction*> &sol_gridfunctions,
                             int i, int ndof, Vector &SubVector);

   // Gets all dofs for a given element (goes over all skeletal FES)
   // includes an array that counts the number of dofs per FES
   void GetFaceVDofs(int i, Array<int> &vdofs, Array<int> &dof_length) const;

   // Gets all dofs for a given element (goes over all volume FES)
   void GetFaceVDofs(int i, Array<int> &vdofs) const;

   /// Returns the sparse Schur complement matrix
   // For block systems: the matrices are counted row-wise
   const SparseMatrix *SpMatSC(int m = 0) const
   {
      MFEM_VERIFY(mat[m], "mat is NULL and can't be dereferenced");
      return mat[m];
   }

   /// Returns the
   // For block systems: the matrices are counted row-wise
   SparseMatrix *SpMatSC(int m = 0)
   {
      MFEM_VERIFY(mat[m], "mat is NULL and can't be dereferenced");
      return mat[m];
   }

   /// Returns a constant reference to the right hand side vector
   const Vector *VectorSC(int m = 0) const
   {
      return rhs_SC[m];
   }

   /// Returns a reference to the right hand side vector
   Vector *VectorSC(int m = 0)
   {
      return rhs_SC[m];
   }


   /// Adds new HDG Integrators (domain terms).
   void AddHDGDomainIntegrator(BilinearFormIntegrator *bfi);

   /// Adds new HDG Integrators (facet terms).
   void AddHDGFaceIntegrator(BilinearFormIntegrator *bfi);

   /// Allocates the vectors for the part of A and B that will be stored
   void Allocate(const Array<int> &bdr_attr_is_ess,
                 const real_t memA = 0.0, const real_t memB = 0.0);

   /// Assembles the Schur complement - general approach, with FES arrays.
   // rhs_F is the volume right hand side gf array
   // rhs_G is the skeleton right hand side gf array
   // sol is the array of gfs containing the known boundary conditions
   void AssembleSC(Array<GridFunction*> rhs_F,
                   Array<GridFunction*> rhs_G,
                   const Array<int> &bdr_attr_is_ess,
                   Array<GridFunction*> sol,
                   int skip_zeros = 1);

   /// Assembles the Schur complement - for the hdg_advection.cpp and hdg_advectionp.cpp test cases
   void AssembleSC(GridFunction *F,
                   const real_t memA = 0.0, const real_t memB = 0.0,
                   int skip_zeros = 1);

   /// Assembles the Schur complement - for the hdg_poisson.cpp and hdg_poissonp.cpp test cases
   void AssembleSC(GridFunction *F1,
                   GridFunction *F2,
                   Array<int> &bdr_attr_is_ess,
                   GridFunction &sol,
                   const real_t memA = 0.0, const real_t memB = 0.0,
                   int skip_zeros = 1);

   /** Eliminate the boundary condition
    * Currently only used for the diffusion case, but written in a way that it can be used for mutiple skeleton veriables
    * Eliminates the row and also the column (to keep the SPD matrix for diffusion)
    */
   void Eliminate_BC(const Array<int> &vdofs_e1, const Array<int> &vdofs_e1_length,
                     const int ndof_u,
                     Array<GridFunction*> sol, Vector *rhs_Volume, Vector *rhs_Skeleton,
                     DenseMatrix *B_local, DenseMatrix *C_local, DenseMatrix *D_local);

   /* To calculate the inverse of the local matrix A
    * Can be optimized is some parts are easy to implement (such as the vector mass integrator A11 for the diffusion case)
    */
   DenseMatrix CalculateInverse(DenseMatrix A_local);

   /// Computes domain based integrators
   void compute_domain_integrals(const int elem, DenseMatrix *A_local);

   /// Computes face based integrators
   void compute_face_integrals(const int elem,
                               const int edge,
                               const int isshared,
                               const bool reconstruct_only,
                               DenseMatrix *A_local,
                               DenseMatrix *B_local,
                               DenseMatrix *C_local,
                               DenseMatrix *D_local);

   /// Reconstructs u from the facet unknowns - general approach, with FES arrays.
   // Volume_GF is the volume right hand side gf array
   // Skeleton_GF is the skeleton solution gf array
   // u is the volume solution gf array
   void Reconstruct(Array<GridFunction*> Volume_GF,
                    Array<GridFunction*> Skeleton_GF,
                    Array<GridFunction*> u);

   /// Reconstructs u from the facet unknowns - for the hdg_advection.cpp and hdg_advectionp.cpp test cases
   void Reconstruct(GridFunction *F,
                    GridFunction *ubar,
                    GridFunction *u);

   /// Reconstructs u and q from the facet unknowns - for the hdg_poisson.cpp and hdg_poissonp.cpp test cases
   void Reconstruct(GridFunction *R,
                    GridFunction *F,
                    GridFunction *ubar,
                    GridFunction *q,
                    GridFunction *u);

   /**
    Depending on 'assemble' it either assembles the Schur complement, or
    reconstruct the volume unknowns from the facet solution.
    For the asssebly part:
     - Vol_GF is the array of the volume equations right hand side grid functions (might be all zero)
     - Skel_GF is the array of the skeletal equations right hand side grid functions (might be all zero)
     - bdr_sol_sol_GF is the projection of the exact solution to the facet unkowns (used only on the boundary for elimination)
     - bdr_attr_is_ess is the array is essential bouddaries for all facet eqautions (might be empty)
    For the reconstructions part:
     - Vol_GF is the array of the volume equations right hand side grid functions (might be all zero)
     - Skel_GF is the array of the skeletal solutions
     - bdr_sol_sol_GF is the array of the volume unknowns (to be recontructed)

    */
   void AssembleReconstruct(Array<GridFunction*> Vol_GF,
                            Array<GridFunction*> Skel_GF,
                            const Array<int> &bdr_attr_is_ess,
                            Array<GridFunction*> bdr_sol_sol_GF,
                            bool assemble = true,
                            const real_t memA = 0.0, const real_t memB = 0.0,
                            int skip_zeros = 1);


   /// Updates the spaces
   virtual void Update(FiniteElementSpace *nfes1 = NULL,
                       FiniteElementSpace *nfes2 = NULL);

   // Add the vector v_add to the right hand side of the Shur complement system. Works even for block Shur complement systems.
   void AddToRHS(Array<int> &skeletal_vdofs, Array<int> &skeletal_vdof_length,
                 Vector v_add);

   // Add the matrix dm_add to the Schur complement system. Works even for block Shur complement systems.
   void AddToMat(Array<int> &skeletal_vdofs_edge_i,
                 Array<int> &skeletal_vdof_length_edge_i,
                 Array<int> &skeletal_vdofs_edge_j, Array<int> &skeletal_vdof_length_edge_j,
                 DenseMatrix dm_add, int skip_zeros);

   /// Destroys bilinear form.
   virtual ~HDGBilinearForm();

#ifdef MFEM_USE_MPI
   // Returns the matrix assembled on the true dofs, i.e. P^t A P.
   // For block systems: the matrices are counted row-wise
   HypreParMatrix *ParallelAssembleSC(int i = 0)
   {
      return ParallelAssembleSC(i,mat[i]);
   }
   // Return the matrix m assembled on the true dofs, i.e. P^t A P
   HypreParMatrix *ParallelAssembleSC(int i, SparseMatrix *m);

   // Return the right hand side vector
   HypreParVector *ParallelVectorSC(int i = 0);

   // Return the right hand side vector
   void ParallelVectorSC(int i, Vector &tv);

   // 2025 Sept begins
   HypreParMatrix *ParallelAssemble(int i, SparseMatrix *m);

   void ParallelAssemble(OperatorHandle &A, int i = 0) { ParallelAssemble(i, A, mat[i]); }

   /** Returns the matrix @a A_local assembled on the true dofs, i.e.
      @a A = P^t A_local P in the format (type id) specified by @a A. */
   // needs input i to know which part of the Schur complement are we working on
   void ParallelAssemble(int i, OperatorHandle &A, SparseMatrix *m);
   // 2025 Sept ends

#endif
};

#endif
