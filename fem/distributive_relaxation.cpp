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

// Implementation of data types for distributive relaxation smoother


#include "../linalg/kernels.hpp"
#include "../general/forall.hpp"
#include "distributive_relaxation.hpp"
#include "fem.hpp"
#include <iostream>
#include <unordered_map>
#include <cmath>

#define DRSMOOTHER_3D_EDGES

namespace mfem
{

DRSmoother::DRSmoother(DisjointSets *clustering, const SparseMatrix *a,
                       bool use_composite, double sc, bool use_l1, const Operator *op)
  : l1(use_l1), scale(sc), composite(use_composite), A(a)
{
   MFEM_VERIFY(use_l1 == false, "l1 scaling not currently supported.");

   if (op != NULL) { SetOperator((Operator&) *op);}
   else { SetOperator((Operator&) *A);}

   FormG(clustering);
   tmp.SetSize(A->Width());
}

void DRSmoother::SetOperator(const Operator &op)
{
   oper = &op;
   width = this->oper->Width();
   height = this->oper->Height();
}

/// Matrix vector multiplication with distributive relaxation smoother.
void DRSmoother::Mult(const Vector &b, Vector &x) const
{
   const Vector *rhs = &b;
   if (iterative_mode)
   {
      MFEM_ASSERT(oper != NULL, "Must set operation before using DRSmoother");
      tmp2 = x;
      oper->Mult(x, tmp);
      add(b, -1.0, tmp, tmp);
      rhs = &tmp;
   }
   DRSmootherJacobi(*rhs, x);
   if (iterative_mode) { add(tmp2, x, x); }

   if (composite)
   {
      tmp2 = x;
      A->Jacobi2(b,tmp2,x);
   }
}

namespace kernels
{

#ifdef MFEM_USE_CUDA
#define MFEM_FORCE_INLINE __forceinline__
#  ifdef __CUDA_ARCH__
#    define MFEM_BLOCK_SIZE(k) gridDim.k
#  else
#    define MFEM_BLOCK_SIZE(k) 1
#  endif
#else
#define MFEM_FORCE_INLINE
#define MfEM_BLOCK_SIZE(k) 1
#endif

template <int DIM>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE static void computeSmootherAction(
   const double *__restrict__ d, const double *__restrict__ g,
   const double *__restrict__ v, double *__restrict__ ret)
{
  double d0givi = g[0]*v[0];

  MFEM_UNROLL(DIM-1)
  for (int i = 1; i < DIM; ++i) d0givi += g[i]*v[i]; // make the givi portion
  d0givi *= d[0]; // and finally the d0

  MFEM_UNROLL(DIM)
  for (int i = 0; i < DIM; ++i) {
    ret[i] = d0givi*g[i];
    if (i) ret[i] += d[i]*v[i];
  }
  return;
}

// specialize for DIM = 1 since g = 1 so we save the multiply
template <>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void computeSmootherAction<1>
(const double *__restrict__ d, const double *__restrict__ g,
 const double *__restrict__ v, double *__restrict__ ret)
{
   ret[0] = d[0]*v[0];
   return;
}

} // namespace kernels

template <int DIM>
static void forAllDispatch(const int size, const double scale,
                           const int *__restrict__ c, const double *__restrict__ dg,
                           const double *__restrict__ b, double *__restrict__ x)
{
  MFEM_FORALL_3D_GRID(group, size, MFEM_CUDA_BLOCKS, 1, 1, 1,
   {
     group = group*MFEM_THREAD_SIZE(x)+MFEM_THREAD_ID(x);
     if (group < size) {
      double diags[DIM],gcoeffs[DIM],vin[DIM],vout[DIM];
      int    cmap[DIM];

      // load the diagonals of g^TAg ("D")
      MFEM_UNROLL(DIM)
      for (int i = 0; i < DIM; ++i) diags[i]   = MFEM_LDG(dg+2*DIM*group+i);

      // load the coefficient array ("G")
      MFEM_UNROLL(DIM)
      for (int i = 0; i < DIM; ++i) gcoeffs[i] = MFEM_LDG(dg+2*DIM*group+DIM+i);

      // load the map from packed representation to DOF layout in the vector
      MFEM_UNROLL(DIM)
      for (int i = 0; i < DIM; ++i) cmap[i]    = MFEM_LDG(c+DIM*group+i);

      // load the input vector
      MFEM_UNROLL(DIM)
      for (int i = 0; i < DIM; ++i) vin[i]     = MFEM_LDG(b+cmap[i]);

      kernels::computeSmootherAction<DIM>(diags,gcoeffs,vin,vout);

      // stream results back down
      MFEM_UNROLL(DIM)
      for (int i = 0; i < DIM; ++i) x[cmap[i]] = scale*vout[i];
     }
   });
   MFEM_DEVICE_SYNC;
   return;
}

template <>
void forAllDispatch<1>(const int size, const double scale,
                       const int *__restrict__ c, const double *__restrict__ dg,
                       const double *__restrict__ b, double *__restrict__ x)
{
   MFEM_FORALL_3D_GRID(group, size, MFEM_CUDA_BLOCKS, 1, 1, 1,
   {
     group = group*MFEM_THREAD_SIZE(x)+MFEM_THREAD_ID(x);
     if (group < size) {
      double diags,vin,vout;
      int    cmap;

      // load the diagonals of g^TAg ("D")
      diags = MFEM_LDG(dg+group);

      // load the map from packed representation to DOF layout in the vector
      cmap  = MFEM_LDG(c+group);

      // load the input vector
      vin   = MFEM_LDG(b+cmap);

      // G is unused
      kernels::computeSmootherAction<1>(&diags,NULL,&vin,&vout);

      // stream results back down
      x[cmap] = scale*vout;
     }
   });
   MFEM_DEVICE_SYNC;
   return;
}

static inline bool isCloseAtTol(double a, double b,
                                double rtol = std::numeric_limits<double>::epsilon(),
                                double atol = std::numeric_limits<double>::epsilon())
{
   using std::fabs; // koenig lookup
   using std::max;
   return fabs(a-b) <= max(rtol*max(fabs(a),fabs(b)),atol);
}

void DRSmoother::DRSmootherJacobi(const Vector &b, Vector &x) const
{
   mfem::out<<"========================================================="<<std::endl;
   auto devDG = diagonal_scaling.Read();
   auto devB  = b.Read();
   auto devX  = x.Write();

   for (int i = 0, totalSize = 0; i < clusterPack.size(); ++i)
   {
      const int cpackSize = clusterPack[i].Size();

      if (cpackSize)
      {
         auto devC = clusterPack[i].Read();
         switch (i+1)
         {
            case 1:
	      forAllDispatch<1>(cpackSize,scale,devC,devDG,devB,devX);
	      totalSize += cpackSize; // no interleaving
	      break;
            case 2:
	      forAllDispatch<2>(cpackSize/2,scale,devC,devDG+totalSize,devB,devX);
	      totalSize += 2*cpackSize;
	      break;
	    case 3:
	      forAllDispatch<3>(cpackSize/3,scale,devC,devDG+totalSize,devB,devX);
	      totalSize += 2*cpackSize;
	      break;
	    case 4:
	      forAllDispatch<4>(cpackSize/4,scale,devC,devDG+totalSize,devB,devX);
	      totalSize += 2*cpackSize;
	      break;
            default:
	      MFEM_ABORT("Clustering of size "<<i+1<<" not yet implemented");
	      break;
         }
      }
   }
   b.PrintHash(mfem::out);
   return;
}

LORInfo::LORInfo(const Mesh &lor_mesh, Mesh &ho_mesh, int p)
{
   // This function is based on the constructor Mesh(Mesh*, int, int)
   // for constructing a low-order refined mesh from a high-order mesh.
   // This method assumes that it is passed the high-order mesh used
   // to construct the LOR mesh

   const int ref_type = BasisType::GaussLobatto;

   dim = ho_mesh.Dimension();
   order = p;

   MFEM_VERIFY(order >= 1, "the order must be >= 1");
   MFEM_VERIFY(dim == 1 || dim == 2 ||
               dim == 3,"only implemented for Segment, Quadrilateral and Hexahedron elements in 1D/2D/3D");
   MFEM_VERIFY(ho_mesh.GetNumGeometries(dim) <= 1,
               "meshes with mixed elements are not supported");

   // Construct a scalar H1 FE space and use its dofs as
   // the indices of the new, refined vertices.
   H1_FECollection rfec(order, dim, ref_type);
   FiniteElementSpace rfes(&ho_mesh, &rfec);

   // Add refined elements and set vertex coordinates
   Array<int> rdofs;
   H1_FECollection vertex_fec(1, dim);

   dofs = new Array<int>();

   for (int el = 0; el < ho_mesh.GetNE(); el++)
   {
      Geometry::Type geom = ho_mesh.GetElementBaseGeometry(el);
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, order);

      rfes.GetElementDofs(el, rdofs);
      MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(),
                  "Element degrees of freedom must have same size as refined geometry");

      const int lower = dofs->Size();
      dofs->SetSize(dofs->Size()+rdofs.Size());

      if (dim == 1)
      {
         (*dofs)[lower] = rdofs[0];
         (*dofs)[lower+order] = rdofs[1];
         for (int j = 1; j < order; ++j) { (*dofs)[lower+j] = rdofs[j+1]; }
      }
      else if (dim == 2)
      {
         MFEM_ASSERT(rdofs.Size() == (order+1)*(order+1),"Wrong number of DOFs!");

         (*dofs)[lower] = rdofs[0];
         (*dofs)[lower+order] = rdofs[1];
         (*dofs)[dofs->Size()-1] = rdofs[2];
         (*dofs)[dofs->Size()-1-order] = rdofs[3];

         int idx = 4;
         for (int j = 1; j < order; ++j) {(*dofs)[lower+j] = rdofs[idx]; ++idx;}
         for (int j = 1; j < order; ++j) {(*dofs)[lower+order+j*(order+1)] = rdofs[idx]; ++idx;}
         for (int j = 1; j < order; ++j) {(*dofs)[dofs->Size()-1-j] = rdofs[idx]; ++idx;}
         for (int j = 1; j < order; ++j) {(*dofs)[dofs->Size()-1-order-j*(order+1)] = rdofs[idx]; ++idx;}
         for (int j = 1; j < order; ++j)
         {
            for (int k = 1; k < order; ++k)
            {
               (*dofs)[lower+k+j*(order+1)] = rdofs[idx]; ++idx;
            }
         }
      }
      else if (dim == 3)
      {
         const int lx = 1;
         const int ly = order+1;
         const int lz = ly * (order+1);
         const int s = order+1;

         MFEM_ASSERT(rdofs.Size() == s*s*s,"Wrong number of DOFs!");

         // Vertices
         (*dofs)[lower] = rdofs[0];
         (*dofs)[lower+(s-1)*lx] = rdofs[1];
         (*dofs)[lower+(s-1)*lx+(s-1)*ly] = rdofs[2];
         (*dofs)[lower+(s-1)*ly] = rdofs[3];
         (*dofs)[lower+(s-1)*lz] = rdofs[4];
         (*dofs)[lower+(s-1)*lx+(s-1)*lz] = rdofs[5];
         (*dofs)[lower+(s-1)*lx+(s-1)*ly+(s-1)*lz] = rdofs[6];
         (*dofs)[lower+(s-1)*ly+(s-1)*lz] = rdofs[7];

         // Edges
         int idx = 8;
         for (int j = 0; j < 12; ++j)
         {
            for (int k = 1; k < s-1; ++k)
            {
               int iz,iy,ix;

               if (j < 4) { iz = 0; }
               else if (j < 8) { iz = s-1; }
               else { iz = k; }

               if (j == 0 || j == 4 || j == 8 || j == 9) { iy = 0; }
               else if (j == 2 || j == 6 || j == 10 || j == 11) { iy = s-1; }
               else { iy = k; }

               if (j % 2 == 0 && j < 8) { ix = k; }
               else if (j == 1 || j == 5 || j == 9 || j == 10) { ix = s-1; }
               else { ix = 0; }

               (*dofs)[lower+ix*lx+iy*ly+iz*lz] = rdofs[idx]; ++idx;
            }
         }

         // Faces
         for (int i = 0; i < 6; ++i)
         {
            for (int j = 1; j < s-1; ++j)
            {
               for (int k = 1; k < s-1; ++k)
               {
                  int iz,iy,ix;

                  if (i == 0) { iz = 0; }
                  else if (i == 5) { iz = s-1; }
                  else { iz = j; }

                  if (i == 0) { iy = s-1 - j; }
                  else if (i == 5) { iy = j; }
                  else if (i == 1) { iy = 0; }
                  else if (i == 3) { iy = s-1; }
                  else if (i == 2) { iy = k; }
                  else if (i == 4) { iy = s-1 - k; }
                  else { MFEM_ABORT("i must lie between 0 and 5"); iy = 0;}

                  if (i == 0 || i == 1 || i == 5) { ix = k; }
                  else if (i == 3) { ix = s-1 - k; }
                  else if (i == 2) { ix = s-1; }
                  else { ix = 0; }

                  (*dofs)[lower+ix*lx+iy*ly+iz*lz] = rdofs[idx]; ++idx;
               }
            }
         }

         // Interior
         for (int iz = 1; iz < s-1; ++iz)
         {
            for (int iy = 1; iy < s-1; ++iy)
            {
               for (int ix = 1; ix < s-1; ++ix)
               {
                  (*dofs)[lower+ix*lx+iy*ly+iz*lz] = rdofs[idx]; ++idx;
               }
            }
         }
         MFEM_ASSERT(lower + idx == dofs->Size(),
                     "Wrong number of elements added to array");
      }
      else
      {
         MFEM_ABORT("Only implemented for one, two, or three dimensions");
      }
   }
   num_dofs = lor_mesh.GetNV();
}

namespace kernels
{

// convert i,j to row-major
#define MFEM_MAT_ROW_MAJOR(_mat,_lda,_i,_j) (_mat)[((_lda)*(_i))+(_j)]
// convert i,j to column-major
#define MFEM_MAT_COL_MAJOR(_mat,_lda,_i,_j) MFEM_MAT_ROW_MAJOR(_mat,_lda,_j,_i)
//#define MFEM_MAT_COL_MAJOR(_mat,_lda,_i,_j) (_mat)[((_lda)*(_j))+(_i)]
//#define MFEM_MAT_ROW_MAJOR(_mat,_lda,_i,_j) (_mat)[((_lda)*(_i))+(_j)]


enum class MATRIX_LAYOUT {ROW_MAJOR,COLUMN_MAJOR};
enum class MATRIX_FACTOR_TYPE {CHOLESKY};

#if 0
template <typename T_, int size_>
class DataArray
{
  typedef T_ value_type;

  value_type _data[size_];

};

template <typename T_, int dim_, MATRIX_LAYOUT layout_ = MATRIX_LAYOUT::COLUMN_MAJOR>
struct Tensor
{
public:
  typedef T_               value_type;
  typedef MATRIX_LAYOUT    layout_type;
  static const layout_type _layout = layout_;


  static const int _dim = dim_;

private:
  value_type _data[_dim*_dim];

  MFEM_HOST_DEVICE MFEM_FORCE_INLINE value_type rvalueHelper(const int i, const int j)
  {
    if (_layout == layout_type::COLUMN_MAJOR) {
      return  MFEM_MAT_COL_MAJOR(_data,_dim,i,j);
    } else if (_layout == layout_type::ROW_MAJOR) {
      return MFEM_MAT_ROW_MAJOR(_data,_dim,i,j);
    }
  }

  MFEM_HOST_DEVICE MFEM_FORCE_INLINE void lvalueHelper(value_type &&other, const int i, const int j)
  {
    if (_layout == layout_type::COLUMN_MAJOR) {
      MFEM_MAT_COL_MAJOR(_data,_dim,i,j) = other;
    } else if (_layout == layout_type::ROW_MAJOR) {
      MFEM_MAT_ROW_MAJOR(_data,_dim,i,j) = other;
    }
    return;
  }

  MFEM_HOST_DEVICE MFEM_FORCE_INLINE value_type scratchVecDot(const double *__restrict__ a, const double *__restrict__ b)
  {
    double ret = a[0]*b[0];

    MFEM_UNROLL(_dim)
    for (int i = 1; i < _dim; ++i) ret += a[i]*b[i];

    return ret;
  }

public:
  MFEM_HOST_DEVICE Tensor() {}
  MFEM_HOST_DEVICE ~Tensor() {}

  MFEM_HOST_DEVICE MFEM_FORCE_INLINE value_type operator()(const int _i, const int _j)
  {
    return rvalueHelper(_i,_j);
  }

  MFEM_HOST_DEVICE MFEM_FORCE_INLINE const value_type& operator()(const int _i, const int _j) const
  {
    if (_layout == layout_type::COLUMN_MAJOR) {
      return MFEM_MAT_COL_MAJOR(_data,_dim,_i,_j);
    } else if (_layout == layout_type::ROW_MAJOR) {
      return MFEM_MAT_ROW_MAJOR(_data,_dim,_i,_j);
    }
  }

  MFEM_HOST_DEVICE MFEM_FORCE_INLINE void load(const int *__restrict__ I,
					       const int *__restrict__ J,
					       const double *__restrict__ data,
					       const int *__restrict__ clusters)
  {
   MFEM_UNROLL(_dim)
   for (int i = 0; i < _dim; ++i)
   {
      const int dof_i = MFEM_LDG(clusters+i);
      const int dofLo = MFEM_LDG(I+dof_i);
      const int dofHi = MFEM_LDG(I+dof_i+1);
      // shouldn't unroll here, we don't know trip count and nvcc is kinda terrible at
      // guessing it
      for (int j = dofLo; j < dofHi; ++j)
      {
         const int dof_j = MFEM_LDG(J+j);

         MFEM_UNROLL(_dim)
         for (int k = 0; k < _dim; ++k)
         {
            if (dof_j == MFEM_LDG(clusters+k))
            {
	      //_assignHelper(MFEM_LDG(data+j),i,k);
               break;
            }
         }
      }
   }
   return;
  }

  // SFINAE for factorization
  template <MATRIX_FACTOR_TYPE F, std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,int> = 0>
  MFEM_HOST_DEVICE MFEM_FORCE_INLINE void factor(void)
  {
    // compute in-place lower cholesky using Cholesky-Crout algorithm for matrix A = LL^T
    MFEM_UNROLL(_dim)
    for (int j = 0; j < _dim; ++j) {
      double x = rvalueHelper(j,j);

      // trip count is known at compile time
      for (int k = 0; k < j; ++k) x -= rvalueHelper(j,k)*rvalueHelper(j,k);

      x = sqrt(x);
      lvalueHelper(j,j) = x;

      const double r = 1.0/x;

      MFEM_UNROLL(LDA)
      for (int i = j+1; i < _dim; ++i) {
	x = rvalueHelper(i,j);

	// trip count also known at compile time
	for (int k = 0; k < j; ++k) x -= rvalueHelper(i,k)*rvalueHelper(j,k);
	lvalueHelper(i,j) = x*r;
      }
    }
    return;
  }

  // Solves Ax = b assuming that A is in cholesky factorized, i.e. A = L L^T. This routine
  // is also the equivalent of computing x = A^-1 b
  template <MATRIX_FACTOR_TYPE F, std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,int> = 0>
  MFEM_HOST_DEVICE MFEM_FORCE_INLINE void matMultInv(const double *__restrict__ b, double *__restrict__ x)
  {
    // scratch space to hold the intermediate soln
    double y[_dim];

    MFEM_UNROLL(_dim)
    for (int i = 0; i < _dim; ++i) {y[i] = 0.0;}

    // forward substitution
    MFEM_UNROLL(_dim)
    for (int i = 0; i < _dim; ++i) {
      double tmp = b[i];

      for (int j = 0; j < i-1; ++j) tmp -= rvalueHelper(i,j)*y[j];
      y[i] = tmp/rvalueHelper(i,i);
    }

    // backward substitution
    MFEM_UNROLL(_dim)
    for (int i = _dim-1; i > -1; --i) {
      double tmp = y[i];

      // indices are flipped here for transpose
      for (int j = i+1; j < _dim; ++j) tmp -= rvalueHelper(j,i)*x[j];
      x[i] = tmp/rvalueHelper(i,i);
    }
    return;
  }

  template <MATRIX_FACTOR_TYPE F>
  MFEM_HOST_DEVICE MFEM_FORCE_INLINE void lowestEigenPair(double *__restrict__ eigVec, double &eigVal)
  {
    double eigVecTmp[_dim],scratch[_dim];
    double eigTmp;

    constexpr double LDA_d   = static_cast<double>(_dim);
    constexpr double initVal = LDA_d/constexprSqrt(LDA_d);

    MFEM_UNROLL(_dim)
    for (int i = 0; i < _dim; ++i) eigVecTmp[i] = initVal;

    // factor ourselves
    factor<F>();

    matMultInv<F>(eigVec,scratch);
    eigTmp = scratchVecDot(eigVec,scratch);

    int iter = 0;
    do {
      // scratch already contains valid vector in first iteration
      if (iter) matMultInv<F>(mat,eigVec,scratch);

      vecNormalize<LDA>(scratch,eigVecTmp);
      ++iter;
    } while (1);
    return;
  }
};
#endif

// Load a dense submatrix from global memory into local memory using efficient RO data
// cache loads
template <int LDA>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void loadSubmatLDG(const int *__restrict__ I,
                                                      const int *__restrict__ J, const double *__restrict__ data,
                                                      const int *__restrict__ clusters, double *__restrict__ subMat)
{
   MFEM_UNROLL(LDA)
   for (int i = 0; i < LDA; ++i)
   {
      const int dof_i = MFEM_LDG(clusters+i);
      const int dofLo = MFEM_LDG(I+dof_i);
      const int dofHi = MFEM_LDG(I+dof_i+1);
      // shouldn't unroll here, we don't know trip count and nvcc is kinda terrible at
      // guessing it
      for (int j = dofLo; j < dofHi; ++j)
      {
         const int dof_j = MFEM_LDG(J+j);

         MFEM_UNROLL(LDA)
         for (int k = 0; k < LDA; ++k)
         {
            if (dof_j == MFEM_LDG(clusters+k))
            {
	       MFEM_MAT_COL_MAJOR(subMat,LDA,i,k) = MFEM_LDG(data+j);
               break;
            }
         }
      }
   }
   return;
}

// compute diag(g^T x A x g) for generic sizes
template <int LDA>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void GTAGDiag(const double G[LDA],
                                                 const double A[LDA*LDA], double result[LDA])
{
  result[0] = 0;
  MFEM_UNROLL(LDA-1)
  for (int i = 1; i < LDA; ++i) result[i] = MFEM_MAT_COL_MAJOR(A,LDA,i,i);

  MFEM_UNROLL(LDA)
  for (int i = 0; i < LDA; ++i) {
    const double gi = G[i];

    MFEM_UNROLL(LDA)
    for (int j = 0; j < LDA; ++j) {
      result[0] += MFEM_MAT_COL_MAJOR(A,LDA,j,i)*G[j]*gi; // let the gods decide double
							  // precision round-off error
    }
  }
  return;
}

// declare generic template so size 4+ compiles. Sadly constexpr if is c++17...
template <int d> void CalcEigenvalues(const double *data, double *lambda, double *vec) {}

// Solves Ax = b assuming that A is in cholesky factorized, i.e. A = L L^T. This routine
// is also the equivalent of computing x = A^-1 b
template <int LDA, MATRIX_FACTOR_TYPE F, typename std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,bool>::type = true>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void	matMultInv(const double *__restrict__ mat, const double *__restrict__ b, double *__restrict__ x)
{
  // scratch space to hold the intermediate soln
  double y[LDA];

  // forward substitution
  MFEM_UNROLL(LDA)
  for (int i = 0; i < LDA; ++i) {
    double tmp = b[i];

    for (int j = 0; j < i; ++j) tmp -= MFEM_MAT_COL_MAJOR(mat,LDA,i,j)*y[j];
    y[i] = tmp/MFEM_MAT_COL_MAJOR(mat,LDA,i,i);
  }

  // backward substitution
  MFEM_UNROLL(LDA)
  for (int i = LDA-1; i > -1; --i) {
    double tmp = y[i];

    // indices are flipped here for transpose
    for (int j = i+1; j < LDA; ++j) tmp -= MFEM_MAT_COL_MAJOR(mat,LDA,j,i)*x[j];
    x[i] = tmp/MFEM_MAT_COL_MAJOR(mat,LDA,i,i);
  }
  return;
}

// Does matmultinv inplace on the input vector
template <int LDA, MATRIX_FACTOR_TYPE F, typename std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,bool>::type = true>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void	matMultInvInPlace(const double *__restrict__ mat, double *__restrict__ b)
{
  // scratch space to hold the intermediate soln
  double y[LDA];

  // forward substitution
  MFEM_UNROLL(LDA)
  for (int i = 0; i < LDA; ++i) {
    double tmp = b[i];

    for (int j = 0; j < i; ++j) tmp -= MFEM_MAT_COL_MAJOR(mat,LDA,i,j)*y[j];
    y[i] = tmp/MFEM_MAT_COL_MAJOR(mat,LDA,i,i);
  }

  // backward substitution
  MFEM_UNROLL(LDA)
  for (int i = LDA-1; i > -1; --i) {
    double tmp = y[i];

    // indices are flipped here for transpose
    for (int j = i+1; j < LDA; ++j) tmp -= MFEM_MAT_COL_MAJOR(mat,LDA,j,i)*b[j];
    b[i] = tmp/MFEM_MAT_COL_MAJOR(mat,LDA,i,i);
  }
  return;
}

template <int LDA, MATRIX_FACTOR_TYPE F, typename std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,bool>::type = true>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void matFactorInPlace(double *__restrict__ mat)
{
  // compute in-place lower cholesky using Cholesky-Crout algorithm for matrix A = LL^T
  MFEM_UNROLL(LDA)
  for (int j = 0; j < LDA; ++j) {
    double x = MFEM_MAT_COL_MAJOR(mat,LDA,j,j);

    // trip count is known at compile time
    for (int k = 0; k < j; ++k) x -= MFEM_MAT_COL_MAJOR(mat,LDA,j,k)*MFEM_MAT_COL_MAJOR(mat,LDA,j,k);

    x = sqrt(x);
    MFEM_MAT_COL_MAJOR(mat,LDA,j,j) = x;

    const double r = 1.0/x;

    MFEM_UNROLL(LDA)
    for (int i = j+1; i < LDA; ++i) {
      x = MFEM_MAT_COL_MAJOR(mat,LDA,i,j);

      // trip count also known at compile time
      for (int k = 0; k < j; ++k) x -= MFEM_MAT_COL_MAJOR(mat,LDA,i,k)*MFEM_MAT_COL_MAJOR(mat,LDA,j,k);
      MFEM_MAT_COL_MAJOR(mat,LDA,i,j) = x*r;
    }
  }
  return;
}

template <int LDA, MATRIX_FACTOR_TYPE F, typename std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,bool>::type = true>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void matFactor(const double *__restrict__ mat, double *__restrict__ factored)
{
  MFEM_UNROLL(LDA)
  for (int i = 0; i < LDA; ++i) {
    for (int k = 0; k < i; ++k) {
      double tmp = MFEM_MAT_COL_MAJOR(mat,LDA,i,k);

      for (int j = 0; j < k; ++j) {
	tmp -= MFEM_MAT_COL_MAJOR(factored,LDA,i,j)*MFEM_MAT_COL_MAJOR(factored,LDA,k,j);
      }
      MFEM_MAT_COL_MAJOR(factored,LDA,i,k) = tmp/MFEM_MAT_COL_MAJOR(factored,LDA,k,k);
    }

    double tmp = MFEM_MAT_COL_MAJOR(mat,LDA,i,i);
    for (int j = 0; j < i; ++j) {
      tmp -= MFEM_MAT_COL_MAJOR(factored,LDA,i,j)*MFEM_MAT_COL_MAJOR(factored,LDA,i,j);
    }
    MFEM_MAT_COL_MAJOR(factored,LDA,i,i) = sqrt(tmp);
  }
  return;
}

template <int LDA>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE double vecDot(const double *__restrict__ v1, const double *__restrict__ v2)
{
  double ret = v1[0]*v2[0];

  MFEM_UNROLL(LDA-1)
  for (int i = 1; i < LDA; ++i)	ret += v1[i]*v2[i];

  return ret;
}

template <int LDA>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void vecNormalize(double *v)
{
  double mod = v[0]*v[0];

  MFEM_UNROLL(LDA-1)
  for (int i = 1; i < LDA; ++i) mod += v[i]*v[i];

  mod = 1.0/sqrt(mod);

  MFEM_UNROLL(LDA)
  for (int i = 0; i < LDA; ++i) { v[i] *= mod; }
  return;
}

template <int LDA, MATRIX_FACTOR_TYPE F, typename std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,bool>::type = true>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE double matVAV(const double *__restrict__ mat, const double *__restrict__ v)
{
  double scratch[LDA];

  matMultInv<LDA,F>(mat,v,scratch); // scratch <- A @ vec
  return vecDot<LDA>(v,scratch);    // dot <- sratch @ vec
}

// constexpr square root for doubles using the newton-raphson method
MFEM_HOST_DEVICE double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
{
  return curr == prev ? curr : sqrtNewtonRaphson(x,0.5*(curr+x/curr),curr);
}

MFEM_HOST_DEVICE double constexpr constexprSqrt(double x)
{
  return x >= 0 && x < 99999999999999.9 ? sqrtNewtonRaphson(x,x,0) : std::nan("1");
}

template <int LDA, MATRIX_FACTOR_TYPE F, typename std::enable_if<MATRIX_FACTOR_TYPE::CHOLESKY==F,bool>::type = true>
MFEM_HOST_DEVICE MFEM_FORCE_INLINE void	computeLowestEigenVector(const double *__restrict__ mat, double *__restrict__ vec, const double atol = 1e-6)
{
  MFEM_UNROLL(LDA)
  for (int i = 0; i < LDA; ++i)	vec[i] = 1.0/constexprSqrt(static_cast<double>(LDA));

  double matfactored[LDA*LDA];
  matFactor<LDA,F>(mat,matfactored); // matfactored <- chol(mat)

  double val = matVAV<LDA,F>(matfactored,vec); // val <- vec @ A @ vec

  int iter = 0;
  MFEM_UNROLL(5)
  do {
    matMultInvInPlace<LDA,F>(matfactored,vec);      // vec <- A @ vec
    vecNormalize<LDA>(vec);                         // vec <- vec/norm(vec)
    double valTmp = matVAV<LDA,F>(matfactored,vec); // valTmp <- vec @ A @ vec
    if (fabs(val-valTmp) < atol) break;
    val = valTmp;
  } while (++iter < 30); // can't just use while (true) as nvcc has ICE when using debug
			 // flag with pseudo-infinite inlined loops
  return;
}
#undef MFEM_MAT_COL_MAJOR
#undef MFEM_MAT_ROW_MAJOR

} // namespace kernels

template <int LDA>
static void forAllDispatchCoeffs(const int size,
				 const int *__restrict__ I,
				 const int *__restrict__ J,
				 const double *__restrict__ data,
				 const int *__restrict__ clusters,
				 double *__restrict__ ret)
{
  if (LDA <= 3) {
#define SUBMAT membuf
#define EIGVAL (SUBMAT+(LDA*LDA))
#define EIGVEC (EIGVAL+LDA)
   MFEM_FORALL(group, size,
   {
      using namespace kernels;

      double membuf[(LDA+LDA+1)*LDA]; // unified memory buffer
      int    minEigIdx = 0;

      // must zero the submatrix, theres no guarantee the super-matrix has a full NxN's
      // worth of stuff.
      MFEM_UNROLL(LDA*LDA)
      for (int i = 0; i < LDA*LDA; ++i) SUBMAT[i] = 0.0;

      loadSubmatLDG<LDA>(I,J,data,clusters+LDA*group,SUBMAT);
      CalcEigenvalues<LDA>(SUBMAT,EIGVAL,EIGVEC);
      {
         // search the eigenvalues for the smallest eigenvalue, note the "index" here is
         // just multiples of LDA to make indexing easier
         double smallestEigVal = EIGVAL[0];
         MFEM_UNROLL(LDA-1)
         for (int i = 1; i < LDA; ++i)
         {
            if (EIGVAL[i] < smallestEigVal)
            {
               smallestEigVal = EIGVAL[i];
               minEigIdx += LDA;
            }
         }
      }
      vecNormalize<LDA>(EIGVEC+minEigIdx);

      // compute g^T A g, reusing eigVal in place of results
      GTAGDiag<LDA>(EIGVEC+minEigIdx,SUBMAT,EIGVAL);

      // stream to global memory, g^TAg in slots [0,LDA) and coefficients in slots
      // [LDA,2*LDA)
      MFEM_UNROLL(LDA)
      for (int i = 0; i < LDA; ++i) ret[2*LDA*group+i]     = 1.0/(EIGVAL[i]);

      MFEM_UNROLL(LDA)
      for (int i = 0; i < LDA; ++i) ret[2*LDA*group+LDA+i] = EIGVEC[minEigIdx+i];
   });
#undef SUBMAT
#undef EIGVEC
#undef EIGVAL
  } else {
#define SUBMAT membuf
#define	EIGVEC (SUBMAT+(LDA*LDA))
#define DIAGS  (EIGVEC+LDA)
   MFEM_FORALL(group, size,
   {
     using namespace kernels;

     double membuf[(LDA+2)*LDA]; // unified memory buffer

     MFEM_UNROLL(LDA*LDA)
     for (int i = 0; i < LDA*LDA; ++i) SUBMAT[i] = 0.0;

     loadSubmatLDG<LDA>(I,J,data,clusters+LDA*group,SUBMAT);
     computeLowestEigenVector<LDA,MATRIX_FACTOR_TYPE::CHOLESKY>(SUBMAT,EIGVEC,1e-7);
     vecNormalize<LDA>(EIGVEC);

     // compute g^T A g
     GTAGDiag<LDA>(EIGVEC,SUBMAT,DIAGS);

     // stream to global memory, g^TAg in slots [0,LDA) and coefficients in slots
     // [LDA,2*LDA), same as above
     MFEM_UNROLL(LDA)
     for (int i = 0; i < LDA; ++i) ret[2*LDA*group+i]     = 1.0/(DIAGS[i]);

     MFEM_UNROLL(LDA)
     for (int i = 0; i < LDA; ++i) ret[2*LDA*group+LDA+i] = EIGVEC[i];
   });
#undef SUBMAT
#undef EIGVEC
#undef DIAGS
  }
  MFEM_DEVICE_SYNC;
  return;
}

// For LDA = 1 the coeffs are always = 1, (so no point storing them in the interleaved
// format detailed above). Since NVCC is not able to elide the loading of the initial
// coefficients   even if they aren't used we must specialize
template <>
void forAllDispatchCoeffs<1>(const int size,
			     const int *__restrict__ I,
			     const int *__restrict__ J,
			     const double *__restrict__ data,
			     const int *__restrict__ clusters,
			     double *__restrict__ ret)
{
   MFEM_FORALL(group, size,
   {
      double diag_value;

      kernels::loadSubmatLDG<1>(I,J,data,clusters+group,&diag_value);
      ret[group] = 1.0/diag_value;
   });
   MFEM_DEVICE_SYNC;
}

void DRSmoother::FormG(const DisjointSets *clustering)
{
   const Array<int> &bounds = clustering->GetBounds();
   const Array<int> &elems  = clustering->GetElems();
   const Array<int> &sizeCounter = clustering->GetSizeCounter();
   const int         sizeCtrSize = sizeCounter.Size();
   // vector of clusters arranged by size. Entry i contains all the clusters of size i+1
   // (as there are no 0 sized clusters) in the order that they appear in elems.
   clusterPack.resize(sizeCtrSize);
   // an 'i' for each size
   std::vector<int> clusterIter(sizeCtrSize,0);
   // clusterSize[i]
   std::vector<int> clusterSize(sizeCtrSize);

   {
      int totalCoeffSize = 0;
      // loop over all the packed vectors setting size, ignore size 0
      for (int i = 0; i < sizeCtrSize; ++i)
      {
         const int csize = (i+1)*sizeCounter[i];

         clusterPack[i].SetSize(csize);
         clusterSize[i] = totalCoeffSize;
         totalCoeffSize += csize;
      }

      // permutation to unpack clusterPack
      //devicePerm.SetSize(totalCoeffSize);

      // can allocate the full coefficient array now. Due to interleaving of the coefficients
      // and g^TAg we allocate twice as many
      diagonal_scaling.SetSize(2*totalCoeffSize);
   }

   // now we loop over all clusters
   for (int i = 0; i < bounds.Size()-1; ++i)
   {
      // get size of cluster and adjust it
      const int csize = bounds[i+1]-bounds[i], adjustedCsize = csize-1;
      const int ci    = clusterIter[adjustedCsize];

      // append the cluster to the appropriate packed vector
      for (int j = 0; j < csize; ++j)
      {
         //perm[elems[bounds[i]+j]]         = 2*(clusterSize[adjustedCsize]+ci)+j;
         clusterPack[adjustedCsize][ci+j] = elems[bounds[i]+j];
      }
      clusterIter[adjustedCsize] += csize;
   }

   // get device-side memory
   auto devCoeffArray = diagonal_scaling.Write();
   auto devData = A->ReadData();
   auto devI    = A->ReadI(), devJ = A->ReadJ();

   // running total of all the coefficients transfered so far
   for (int i = 0, totalSize = 0; i < clusterPack.size(); ++i)
   {
      const int cpackSize = clusterPack[i].Size();

      if (cpackSize)
      {
         auto devCluster = clusterPack[i].Read();

         switch (i+1)
         {
            case 1:
               forAllDispatchCoeffs<1>(cpackSize,devI,devJ,devData,devCluster,devCoeffArray);
               totalSize += cpackSize; // no interleaving
               break;
            case 2:
               forAllDispatchCoeffs<2>(cpackSize/2,devI,devJ,devData,devCluster,
                                       devCoeffArray+totalSize);
               totalSize += 2*cpackSize;
               break;
            case 3:
               forAllDispatchCoeffs<3>(cpackSize/3,devI,devJ,devData,devCluster,
                                       devCoeffArray+totalSize);
               totalSize += 2*cpackSize;
               break;
	    case 4:
	      forAllDispatchCoeffs<4>(cpackSize/4,devI,devJ,devData,devCluster,devCoeffArray+totalSize);
	       totalSize += 2*cpackSize;
	       break;
            default:
               MFEM_ABORT("Clustering of size "<<i+1<<" not yet implemented");
               break;
         }
      }
   }
   return;
}

void PrintClusteringStats(std::ostream &out, const DisjointSets *clustering)
{
   Array<int> unique_group_sizes;
   Array<int> group_size_counts;
   const Array<int> &bounds = clustering->GetBounds();

   for (int i = 0; i < bounds.Size()-1; ++i)
   {
      const int size = bounds[i+1] - bounds[i];

      int j = 0;
      for (; j < unique_group_sizes.Size(); ++j)
      {
         if (size == unique_group_sizes[j])
         {
            group_size_counts[j]++;
            break;
         }
      }

      if (j == unique_group_sizes.Size())
      {
         unique_group_sizes.Append(size);
         group_size_counts.Append(1);
      }
   }

   if (unique_group_sizes.Size() == 0) { return; }

   out << "Vertex groupings: ";
   out << group_size_counts[0] << " of size " << unique_group_sizes[0];
   for (int i = 1; i < unique_group_sizes.Size(); ++i)
   {
      out << ", " << group_size_counts[i] << " of size " << unique_group_sizes[i];
   }
   out << std::endl;
}

DisjointSets *LORInfo::Cluster() const
{
   const Array<int> &dof_arr = *dofs;

   int dofs_per_elem = 1;
   for (int d = 0; d < dim; ++d) { dofs_per_elem *= order + 1; }

   MFEM_VERIFY(dof_arr.Size() % dofs_per_elem == 0,
               "DOF array for order " << order << " and dimension " << dim <<
               " should be a multiple of " << dofs_per_elem);
   MFEM_VERIFY(order > 1, "Order must be greater than 1");

   DisjointSets *clustering = new DisjointSets(num_dofs);

   if (dim == 1)
   {
      for (int base = 0; base < dof_arr.Size(); base += dofs_per_elem)
      {
         clustering->Union(dof_arr[base], dof_arr[base+1]);
         clustering->Union(dof_arr[base], dof_arr[base+2]);
         clustering->Union(dof_arr[base], dof_arr[base+3]);
         clustering->Union(dof_arr[base+dofs_per_elem-1], dof_arr[base+dofs_per_elem-2]);
         clustering->Union(dof_arr[base+dofs_per_elem-1], dof_arr[base+dofs_per_elem-3]);
         clustering->Union(dof_arr[base+dofs_per_elem-1], dof_arr[base+dofs_per_elem-4]);
      }
   }
   else if (dim == 2)
   {
      for (int base = 0; base < dof_arr.Size(); base += dofs_per_elem)
      {
         for (int d = 0; d < 2; ++d)
         {
            int li, lj;
            li = order+1;
            lj = 1;

            if (d == 1) { std::swap(li, lj); }

            for (int i = 0; i <= order-1; i += order-1)
            {
               for (int j = 2; j <= order-2; ++j)
               {
                  const int v1 = dof_arr[base + i * li + j * lj];
                  const int v2 = dof_arr[base + (i+1) * li + j * lj];

                  clustering->Union(v1, v2);
               }
            }
         }
      }
   }
   else if (dim == 3)
   {
      const int order_plus_1_sq = (order+1)*(order+1);

      for (int base = 0; base < dof_arr.Size(); base += dofs_per_elem)
      {
         for (int d = 0; d < 6; ++d)
         {
            int li, lj, lk;

            switch (d)
            {
               case 0:
                  li = 1; lj = order+1; lk = order_plus_1_sq; break;
               case 1:
                  lj = 1; li = order+1; lk = order_plus_1_sq; break;
               case 2:
                  lk = 1; lj = order+1; li = order_plus_1_sq; break;
               case 3:
                  li = 1; lk = order+1; lj = order_plus_1_sq; break;
               case 4:
                  lj = 1; lk = order+1; li = order_plus_1_sq; break;
               case 5:
                  lk = 1; li = order+1; lj = order_plus_1_sq; break;
            }

            for (int i = 0; i <= order-1; i += order-1)
            {
               for (int j = 2; j <= order-2; ++j)
               {
                  for (int k = 2; k <= order-2; ++k)
                  {
                     const int v1 = dof_arr[base + i * li + j * lj + k*lk];
                     const int v2 = dof_arr[base + (i+1) * li + j * lj + k*lk];

                     clustering->Union(v1, v2);
                  }
               }
            }
         }

#ifdef DRSMOOTHER_3D_EDGES
         for (int d = 0; d < 3; ++d)
         {
            int li = 1;
            int lj = order+1;
            int lk = order_plus_1_sq;

            if (d == 1)
            {
               std::swap(li, lk);
            }
            else if (d == 2)
            {
               std::swap(lj, lk);
            }

            for (int k = 2; k <= order-2; ++k)
            {
               for (int i = 0; i <= order-1; i += order-1)
               {
                  for (int j = 0; j <= order-1; j += order-1)
                  {
                     const int v1 = dof_arr[base + i*li + j*lj + k*lk];
                     const int v2 = dof_arr[base + (i+1)*li + j*lj + k*lk];
                     const int v3 = dof_arr[base + i*li + (j+1)*lj + k*lk];
                     const int v4 = dof_arr[base + (i+1)*li + (j+1)*lj + k*lk];

                     clustering->Union(v1, v2);
                     clustering->Union(v1, v3);
                     clustering->Union(v1, v4);
                  }
               }
            }
         }
#endif
      }
   }
   else
   {
      MFEM_ABORT("Only supported for dimensions 1, 2, and 3");
   }

   clustering->Finalize();

   const Array<int> &bounds = clustering->GetBounds();
   MFEM_VERIFY(bounds.Size() > 0,
               "Finalized clustering should have established bounds array");

   return clustering;
}

void PrintClusteringForVis(std::ostream &out, const DisjointSets *clustering,
                           const Mesh *mesh)
{
   int dim = mesh->SpaceDimension();

   MFEM_VERIFY(clustering != NULL, "Clustering must be non-null");

   const Array<int> &elems  = clustering->GetElems();
   const Array<int> &bounds = clustering->GetBounds();

   for (int group = 0; group < bounds.Size()-1; ++group)
   {
      for (int i = bounds[group]; i < bounds[group+1]; ++i)
      {
         const double *v = mesh->GetVertex(elems[i]);
         out << v[0];
         for (int j = 1; j < dim; ++j) { out << " " << v[j]; }

         if (i == bounds[group+1]-1) { out << std::endl; }
         else { out << " "; }
      }
   }
}

void DRSmoother::DiagonalDominance(const SparseMatrix *A, double &dd1,
                                   double &dd2)
{
   const int *I = A->GetI();
   const int *J = A->GetJ();
   const double *data = A->GetData();

   dd1 = 0.0;
   dd2 = 0.0;

   for (int i = 0; i < A->Height(); ++i)
   {
      double off_diag = 0.0;
      double diag = 0.0;
      for (int k = I[i]; k < I[i+1]; ++k)
      {
         const int j = J[k];
         const double a = data[k];
         if (i == j)
         {
            diag = a;
         }
         else
         {
            off_diag += fabs(a);
         }
      }
      const double dd = off_diag / diag;
      dd1 = std::max(dd1, dd);
      dd2 += dd;
   }
   dd2 /= A->Height();
}

}
