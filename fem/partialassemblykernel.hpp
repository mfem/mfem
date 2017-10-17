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

// This file contains operator-based bilinear form integrators used
// with BilinearFormOperator.

#ifndef MFEM_PAK
#define MFEM_PAK

#include "../config/config.hpp"
#include "bilininteg.hpp"
#include <vector>

using std::vector;

namespace mfem
{

/**
*  A dummy tensor class
*/
class DummyTensor
{
protected:
   double* data;
   int dim;
   vector<int> sizes;
   vector<int> offsets;

public:
   DummyTensor(int dim) : dim(dim), sizes(dim,1), offsets(dim,1)
   {
   }

   int GetNumVal()
   {
      int result = 1;
      for (int i = 0; i < dim; ++i)
      {
         result *= sizes[i];
      }
      return result;
   }

   // Memory leak if used more than one time
   void SetSize(int* _sizes)
   {
      for (int i = 0; i < dim; ++i)
      {
         sizes[i] = _sizes[i];
         int dim_ind = 1;
         for (int j = 0; j < i; ++j)
         {
            dim_ind *= sizes[j];
         }
         offsets[i] = dim_ind;
      }
      data = (double*)malloc(GetNumVal()*sizeof(double));   
   }

   //The function to change in order to change the Layout
   int GetRealInd(int* ind)
   {
      int real_ind = 0;
      for (int i = 0; i < dim; ++i)
      {
         real_ind += ind[i]*offsets[i];
      }
      return real_ind;
   }

   // really unsafe!
   void SetVal(int* ind, double val)
   {
      int real_ind = GetRealInd(ind);
      data[real_ind] = val;
   }

   double GetVal(int real_ind)
   {
      return data[real_ind];
   }

   double GetVal(int* ind)
   {
      int real_ind = GetRealInd(ind);
      return data[real_ind];
   }

   // Assumes that elements/faces indice is always the last indice
   double* GetElmtData(int e){
      return &data[ e * offsets[dim-1] ];
   }

};

/*static void ComputeBasis1d(const FiniteElement *fe, int ir_order,
                           DenseMatrix &shape1d)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   // Compute the 1d shape functions and gradients
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, ir_order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs = fe->GetOrder() + 1;

   shape1d.SetSize(dofs, quads1d);

   Vector u(dofs);
   for (int k = 0; k < quads1d; k++)
   {
      const IntegrationPoint &ip = ir1d.IntPoint(k);
      basis1d.Eval(ip.x, u);
      for (int i = 0; i < dofs; i++)
      {
         shape1d(i, k) = u(i);
      }
   }
}*/

static void ComputeBasis1d(const FiniteElement *fe, int ir_order,
                           DenseMatrix &shape1d, DenseMatrix &dshape1d)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   // Compute the 1d shape functions and gradients
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, ir_order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs = fe->GetOrder() + 1;

   shape1d.SetSize(dofs, quads1d);
   dshape1d.SetSize(dofs, quads1d);

   Vector u(dofs);
   Vector d(dofs);
   for (int k = 0; k < quads1d; k++)
   {
      const IntegrationPoint &ip = ir1d.IntPoint(k);
      basis1d.Eval(ip.x, u, d);
      for (int i = 0; i < dofs; i++)
      {
         shape1d(i, k) = u(i);
         dshape1d(i, k) = d(i);
      }
   }
}

void MultBtDB1(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U);
void MultBtDB2(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U);
void MultBtDB3(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U);
void MultGtDG1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultGtDG2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultGtDG3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultBtDG1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultBtDG2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultBtDG3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultGtDB1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultGtDB2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultGtDB3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);

/**
*  A dummy partial assembly kernel class for domain integrals
*/
class DummyDomainPAK
{
public:
   typedef DummyTensor Tensor;

protected:
   FiniteElementSpace *fes;
   int dim;
   DenseMatrix shape1d, dshape1d;
   Tensor D;

public:

   DummyDomainPAK(FiniteElementSpace *_fes, int ir_order, int tensor_dim)
   : fes(_fes), dim(fes->GetFE(0)->GetDim()), D(tensor_dim)
   {
      // Store the 1d shape functions and gradients
      ComputeBasis1d(fes->GetFE(0), ir_order, shape1d, dshape1d);
   }

   // Returns the tensor D
   Tensor& GetD() { return D; }

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   void MultBtDB(const Vector &U, Vector &V)
   {
      switch(dim)
      {
      case 1:MultBtDB1(fes,shape1d,D,U,V);break;
      case 2:MultBtDB2(fes,shape1d,D,U,V);break;
      case 3:MultBtDB3(fes,shape1d,D,U,V);break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

   /**
   * Computes V = G^T D G U where G is a tensor product of shape1d and dshape1d. 
   */
   void MultGtDG(const Vector &U, Vector &V)
   {
      
      switch(dim)
      {
      case 1:MultGtDG1(fes,shape1d,dshape1d,D,U,V);break;
      case 2:MultGtDG2(fes,shape1d,dshape1d,D,U,V);break;
      case 3:MultGtDG3(fes,shape1d,dshape1d,D,U,V);break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

   /**
   * Computes V = B^T D G U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void MultBtDG(const Vector &U, Vector &V)
   {
      
      switch(dim)
      {
      case 1:MultBtDG1(fes,shape1d,dshape1d,D,U,V);break;
      case 2:MultBtDG2(fes,shape1d,dshape1d,D,U,V);break;
      case 3:MultBtDG3(fes,shape1d,dshape1d,D,U,V);break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

   /**
   * Computes V = G^T D B U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void MultGtDB(const Vector &U, Vector &V)
   {
      switch(dim)
      {
      case 1:MultGtDB1(fes,shape1d,dshape1d,D,U,V);break;
      case 2:MultGtDB2(fes,shape1d,dshape1d,D,U,V);break;
      case 3:MultGtDB3(fes,shape1d,dshape1d,D,U,V);break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }
};

/**
*  A dummy partial assembly kernel class for face integrals
*/
class DummyFacePAK
{
public:
   typedef DummyTensor Tensor;

protected:
   FiniteElementSpace *fes;
   int dim;
   DenseMatrix shape1d, dshape1d;
   Tensor D11,D12,D21,D22;

public:

   DummyFacePAK(FiniteElementSpace *_fes, int ir_order, int tensor_dim)
   : fes(_fes), dim(fes->GetFE(0)->GetDim()),
   D11(tensor_dim), D12(tensor_dim), D21(tensor_dim), D22(tensor_dim)
   {
      // Store the 1d shape functions and gradients
      ComputeBasis1d(fes->GetFE(0), ir_order, shape1d, dshape1d);
   }

   void InitPb(const DenseMatrix& P) { }

   IntegrationPoint& IntPoint(int k){
      //TODO!!!
   }

   // Returns the tensor D11
   Tensor& GetD11() { return D11; }
   // Returns the tensor D12
   Tensor& GetD12() { return D12; }
   // Returns the tensor D11
   Tensor& GetD21() { return D21; }
   // Returns the tensor D12
   Tensor& GetD22() { return D22; }

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   void MultBtDB(const Vector &U, Vector &V)
   {
      // switch(dim)
      // {
      // case 1:MultBtDB1(fes,shape1d,D,U,V);break;
      // case 2:MultBtDB2(fes,shape1d,D,U,V);break;
      // case 3:MultBtDB3(fes,shape1d,D,U,V);break;
      // default: mfem_error("More than # dimension not yet supported"); break;
      // }
   }

   /**
   * Computes V = G^T D G U where G is a tensor product of shape1d and dshape1d. 
   */
   void MultGtDG(const Vector &U, Vector &V)
   {
      
      // switch(dim)
      // {
      // case 1:MultGtDG1(fes,shape1d,dshape1d,D,U,V);break;
      // case 2:MultGtDG2(fes,shape1d,dshape1d,D,U,V);break;
      // case 3:MultGtDG3(fes,shape1d,dshape1d,D,U,V);break;
      // default: mfem_error("More than # dimension not yet supported"); break;
      // }
   }

   /**
   * Computes V = B^T D G U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void MultBtDG(const Vector &U, Vector &V)
   {
      
      // switch(dim)
      // {
      // case 1:MultBtDG1(fes,shape1d,dshape1d,D,U,V);break;
      // case 2:MultBtDG2(fes,shape1d,dshape1d,D,U,V);break;
      // case 3:MultBtDG3(fes,shape1d,dshape1d,D,U,V);break;
      // default: mfem_error("More than # dimension not yet supported"); break;
      // }
   }

   /**
   * Computes V = G^T D B U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void MultGtDB(const Vector &U, Vector &V)
   {
      // switch(dim)
      // {
      // case 1:MultGtDB1(fes,shape1d,dshape1d,D,U,V);break;
      // case 2:MultGtDB2(fes,shape1d,dshape1d,D,U,V);break;
      // case 3:MultGtDB3(fes,shape1d,dshape1d,D,U,V);break;
      // default: mfem_error("More than # dimension not yet supported"); break;
      // }
   }
};


}

#endif //MFEM_PAK