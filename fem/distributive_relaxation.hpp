// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

/// Data type for distributive relaxation smoother of sparse matrix
class DRSmoother : public Solver
{

protected:
   bool l1;
   double scale;
   bool composite;

   const SparseMatrix *A;

   Vector diagonal_scaling;
   std::vector<Array<int>> clusterPack;

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

   /// Matrix vector multiplication with distributive relaxation smoother.
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void SetOperator(const Operator &oper);

   const SparseMatrix *GetGtAG() const;

   // Get diagonal blocks
   static std::vector<DenseMatrix> *DiagonalBlocks(const SparseMatrix *oper,
                                                   const DisjointSets *clustering);

   // For testing
   static void DiagonalDominance(const SparseMatrix *A, double &dd1,  double &dd2);

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
