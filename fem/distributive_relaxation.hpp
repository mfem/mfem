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

#ifndef MFEM_DISTRIBUTIVE_RELAXATION
#define MFEM_DISTRIBUTIVE_RELAXATION

#include "../mesh/mesh.hpp"
#include "../linalg/sparsesmoothers.hpp"
#include "../general/sets.hpp"

namespace mfem
{

class DRSmootherG;
class LORInfo;

/// Data type for distributive relaxation smoother of sparse matrix
class DRSmoother : public Solver
{

protected:
   bool l1;
   double scale;
   bool composite;

   const DRSmootherG *G;
   const SparseMatrix *A;

   Vector diagonal_scaling;

   const Operator *oper;

   mutable Vector tmp, tmp2;

   // Smoother helpers
   void DRSmootherJacobi(const Vector &b, Vector &x) const;
   static void L1Jacobi(const Vector &x0, Vector &x1, const SparseMatrix &A);
   void GtAGDiagScale(const Vector &b, Vector &x) const;

   // Constructor helpers
   void FormG(const DisjointSets *clustering);

public:

   /// Create distributive relaxation smoother.
   DRSmoother(DisjointSets *clustering, const SparseMatrix *A, bool composite=true,
              double sc=2.0/3.0, bool l1=false, const Operator *op=NULL);

   /// Destroy distributive relaxation smoother.
   ~DRSmoother();

   /// Matrix vector multiplication with distributive relaxation smoother.
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void SetOperator(const Operator &oper);

   const DRSmootherG *GetG() const;
   const SparseMatrix *GetGtAG() const;

   // Get diagonal blocks
   static std::vector<DenseMatrix> *DiagonalBlocks(const SparseMatrix *oper,
                                                   const DisjointSets *clustering);

   // For testing
   static void DiagonalDominance(const SparseMatrix *A, double &dd1,  double &dd2);

};

class DRSmootherG : public Operator
{
protected:
   const DisjointSets *clustering; // owned
   const Array<double> *coeffs; // owned
   const SparseMatrix *G; // owned
   bool matrix_free;

public:
   ~DRSmootherG();

   DRSmootherG(const SparseMatrix *g, const DisjointSets *clusters=NULL)
   {
      G = g; clustering = clusters; coeffs = NULL; matrix_free = false;
      width = G->Width(); height = G->Height();
   }
   DRSmootherG(const DisjointSets *clusters, const Array<double> *coeff_data)
   {
      G = NULL; clustering = clusters; coeffs = coeff_data; matrix_free = true;
      width = clustering->Size(); height = clustering->Size();
   }

   void GtAG(SparseMatrix *&GtAG_mat, Vector &GtAG_diagonal, const SparseMatrix &A,
             const std::vector<DenseMatrix> *diag_blocks) const;

   void AddMultTranspose(const Vector &x, Vector &y, double scale=1.0) const;
   void AddMult(const Vector &x, Vector &y, double scale=1.0) const;
   void MultTranspose(const Vector &x, Vector &y) const { y = 0.0; AddMultTranspose(x, y, 1.0); }
   void Mult(const Vector &x, Vector &y) const { y = 0.0; AddMult(x, y, 1.0); }

   const DisjointSets *GetClustering() const { return clustering; }
   const SparseMatrix *GetMatrix() const { return G; }
   bool MatrixFree() const { return matrix_free; }

};

class LORInfo
{
protected:
   int order;
   int dim;
   int num_dofs;
   Array<int> *dofs;

public:
   LORInfo(int order, int dim, int num_dofs, Array<int> *dofs)
   {
      this->order = order;
      this->dim = dim;
      this->num_dofs = num_dofs;
      this->dofs = dofs;
   }

   LORInfo(const Mesh &lor_mesh, Mesh &ho_mesh, int order);

   ~LORInfo() { delete dofs; }

   int Order() const { return order; }
   int Dim() const { return dim; }
   int NumDofs() const { return num_dofs; }
   const Array<int> *Dofs() const { return dofs; }

   DisjointSets *Cluster() const;
};

void PrintClusteringStats(std::ostream &out, const DisjointSets *clustering);
void PrintClusteringForVis(std::ostream &out, const DisjointSets *clustering,
                           const Mesh *mesh);


}

#endif // MFEM_DISTRIBUTIVE_RELAXATION
