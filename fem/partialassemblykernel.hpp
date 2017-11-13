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

#include "fem.hpp"
#include "../config/config.hpp"
#include "bilininteg.hpp"
#include "dalg.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//using namespace std;
//using std::vector;

namespace mfem
{

/**
* Gives the evaluation of the 1d basis functions and their derivative at one point @param x
*/
template <typename Tensor>
static void ComputeBasis0d(const FiniteElement *fe, double x,
                     Tensor& shape0d, Tensor& dshape0d)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis0d = tfe->GetBasis1D();

   const int quads0d = 1;
   const int dofs = fe->GetOrder() + 1;

   // We use Matrix and not Vector because we don't want shape0d and dshape0d to have
   // a different treatment than shape1d and dshape1d
   shape0d  = Tensor(dofs, quads0d);
   dshape0d = Tensor(dofs, quads0d);

   Vector u(dofs);
   Vector d(dofs);
   basis0d.Eval(x, u, d);
   for (int i = 0; i < dofs; i++)
   {
      shape0d(i, 0) = u(i);
      dshape0d(i, 0) = d(i);
   }
}

/**
* Gives the evaluation of the 1d basis functions and their derivative at all quadrature points
*/
template <typename Tensor>
static void ComputeBasis1d(const FiniteElement *fe, int order, Tensor& shape1d,
                           Tensor& dshape1d, bool backward=false)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs = fe->GetOrder() + 1;

   shape1d  = Tensor(dofs, quads1d);
   dshape1d = Tensor(dofs, quads1d);

   Vector u(dofs);
   Vector d(dofs);
   for (int k = 0; k < quads1d; k++)
   {
      int ind = backward ? quads1d -1 - k : k;
      const IntegrationPoint &ip = ir1d.IntPoint(k);
      basis1d.Eval(ip.x, u, d);
      for (int i = 0; i < dofs; i++)
      {
         shape1d(i, ind) = u(i);
         dshape1d(i, ind) = d(i);
      }
   }
}


/////////////////////////////////////////////////
//                                             //
//                                             //
//                 DUMMY KERNEL                //
//                                             //
//                                             //
/////////////////////////////////////////////////


/**
*  The Kernels for BtDB in 1d,2d and 3d.
*/
void MultBtDB1(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U);
void MultBtDB2(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U);
void MultBtDB3(FiniteElementSpace* fes, DenseMatrix const & shape1d,
   DummyTensor & D, const Vector &V, Vector &U);
/**
*  The Kernels for GtDG in 1d,2d and 3d.
*/
void MultGtDG1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultGtDG2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultGtDG3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
/**
*  The Kernels for BtDG in 1d,2d and 3d.
*/
void MultBtDG1(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultBtDG2(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
void MultBtDG3(FiniteElementSpace* fes, DenseMatrix const& shape1d,
   DenseMatrix const& dshape1d, DummyTensor & D, const Vector &V, Vector &U);
/**
*  The Kernels for GtDB in 1d,2d and 3d.
*/
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

   DummyDomainPAK(FiniteElementSpace *_fes, int order, int tensor_dim)
   : fes(_fes), dim(fes->GetFE(0)->GetDim()), D(tensor_dim)
   {
      // Store the 1d shape functions and gradients
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
      cout << "quads1d=" << shape1d.Width() << endl;
   }

   // Returns the tensor D
   Tensor& GetD() { return D; }

   /**
   *  Sets the dimensions of the tensor D
   */
   void SetSizeD(int* sizes)
   {
      D.SetSize(sizes);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   */
   void SetValD(int* ind, double val)
   {
      D(ind) = val;
   }

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

void MultBtDB2int(int ind_trial, FiniteElementSpace* fes,
   DenseMatrix & shape1d, DenseMatrix & shape0d0, DenseMatrix & shape0d1,
   DummyTensor & D, const Vector &U, Vector &V);
void MultBtDB2ext(int ind_trial, FiniteElementSpace* fes,
   DenseMatrix & shape1d, DenseMatrix & shape0d0, DenseMatrix & shape0d1,
   IntMatrix & coord_change, IntMatrix & backward, 
   DummyTensor & D, const Vector &U, Vector &V);
void MultBtDB3(FiniteElementSpace* fes,
   DenseMatrix & shape1d, DenseMatrix & shape0d0, DenseMatrix & shape0d1,
   IntMatrix & coord_change, IntMatrix & backward, 
   DummyTensor & D, const Vector &U, Vector &V);

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
   DenseMatrix shape0d0, shape0d1, dshape0d0, dshape0d1;
   IntMatrix coord_change1, backward1;
   IntMatrix coord_change2, backward2;
   Tensor D11,D12,D21,D22;
   DummyMatrix<IntegrationPoint> intPts;

public:

   DummyFacePAK(FiniteElementSpace *_fes, int order, int tensor_dim)
   : fes(_fes), dim(fes->GetFE(0)->GetDim()),
   coord_change1(dim,fes->GetMesh()->GetNumFaces()),backward1(dim,fes->GetMesh()->GetNumFaces()),
   coord_change2(dim,fes->GetMesh()->GetNumFaces()),backward2(dim,fes->GetMesh()->GetNumFaces()),
   D11(tensor_dim), D12(tensor_dim), D21(tensor_dim), D22(tensor_dim)
   {
      // Store the two 0d shape functions and gradients
      // in x = 0.0
      ComputeBasis0d(fes->GetFE(0), 0.0, shape0d0, dshape0d0);
      // in y = 0.0
      ComputeBasis0d(fes->GetFE(0), 1.0, shape0d1, dshape0d1);
      // Store the 1d shape functions and gradients
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
      // Creates the integration points for each face
      const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);
      const int quads1d = ir1d.GetNPoints();
      intPts = DummyMatrix<IntegrationPoint>(pow(quads1d,dim-1),2*dim);
      switch(dim){
      case 1:
         intPts(0,0).x = 0.0;
         intPts(0,0).weight = 1.0;
         intPts(1,0).x = 1.0;
         intPts(1,0).weight = 1.0;
         break;
      case 2:
         for (int i = 0; i < quads1d; ++i)
         {
            //SOUTH
            intPts(i,0).x = ir1d.IntPoint(i).x;
            intPts(i,0).y = 0.0;
            intPts(i,0).weight = ir1d.IntPoint(i).weight;
            //EAST
            intPts(i,1).x = 1.0;
            intPts(i,1).y = ir1d.IntPoint(i).x;
            intPts(i,1).weight = ir1d.IntPoint(i).weight;
            //NORTH
            intPts(i,2).x = ir1d.IntPoint(i).x;
            intPts(i,2).y = 1.0;
            intPts(i,2).weight = ir1d.IntPoint(i).weight;
            //WEST
            intPts(i,3).x = 0.0;
            intPts(i,3).y = ir1d.IntPoint(i).x;
            intPts(i,3).weight = ir1d.IntPoint(i).weight;
         }
         break;
      case 3:
      //TODO verify that order doesn't matter
         for (int j = 0; j < quads1d; ++j){
            for (int i = 0; i < quads1d; ++i){
               //BOTTOM
               intPts(i+j*quads1d,0).x = ir1d.IntPoint(i).x;
               intPts(i+j*quads1d,0).y = ir1d.IntPoint(j).x;
               intPts(i+j*quads1d,0).z = 0.0;
               intPts(i+j*quads1d,0).weight = ir1d.IntPoint(i).weight * ir1d.IntPoint(j).weight;
               //SOUTH
               intPts(i+j*quads1d,1).x = ir1d.IntPoint(i).x;
               intPts(i+j*quads1d,1).y = 0.0;
               intPts(i+j*quads1d,1).z = ir1d.IntPoint(j).x;
               intPts(i+j*quads1d,1).weight = ir1d.IntPoint(i).weight * ir1d.IntPoint(j).weight;
               //EAST
               intPts(i+j*quads1d,2).x = 1.0;
               intPts(i+j*quads1d,2).y = ir1d.IntPoint(i).x;
               intPts(i+j*quads1d,2).z = ir1d.IntPoint(j).x;
               intPts(i+j*quads1d,2).weight = ir1d.IntPoint(i).weight * ir1d.IntPoint(j).weight;
               //NORTH
               intPts(i+j*quads1d,3).x = ir1d.IntPoint(i).x;
               intPts(i+j*quads1d,3).y = 1.0;
               intPts(i+j*quads1d,3).z = ir1d.IntPoint(j).x;
               intPts(i+j*quads1d,3).weight = ir1d.IntPoint(i).weight * ir1d.IntPoint(j).weight;
               //WEST
               intPts(i+j*quads1d,4).x = 0.0;
               intPts(i+j*quads1d,4).y = ir1d.IntPoint(i).x;
               intPts(i+j*quads1d,4).z = ir1d.IntPoint(j).x;
               intPts(i+j*quads1d,4).weight = ir1d.IntPoint(i).weight * ir1d.IntPoint(j).weight;
               //TOP
               intPts(i+j*quads1d,5).x = ir1d.IntPoint(i).x;
               intPts(i+j*quads1d,5).y = ir1d.IntPoint(j).x;
               intPts(i+j*quads1d,5).z = 1.0;
               intPts(i+j*quads1d,5).weight = ir1d.IntPoint(i).weight * ir1d.IntPoint(j).weight;
            }
         }
         break;
      default:
         mfem_error("Face of that dimension not handled");
         break;
      }
   }

   void InitPb(int face, const IntMatrix& P) {
      // P gives base_e1 to base_e2
      for (int j = 0; j < dim; ++j)
      {
         for (int i = 0; i < dim; ++i)
         {
            // base_e1 -> base_e2
            if (P(i,j)!=0)
            {
               coord_change1(j,face) = i;
               // Checks if the basis vectors are in the same or opposite direction
               backward1(j,face) = P(i,j)<0;
            }
            // base_e2 -> base_e1
            if (P(j,i)!=0)
            {
               coord_change2(j,face) = i;
               // Checks if the basis vectors are in the same or opposite direction
               backward2(j,face) = P(j,i)<0;
            }
         }
      }
   }
   
   // Returns the k-th IntegrationPoint on the face
   IntegrationPoint& IntPoint(int face, int k){
      return intPts(k,face);
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
   * Computes V = B^T D B U where B is a tensor product of shape1d and shape0d. 
   */
   void MultBtDB(const Vector &U, Vector &V)
   {
      /*cout << "coord_change1=\n" << coord_change1 << "-----------" << endl;
      cout << "backward1=\n" << backward1 << "-----------" << endl;
      cout << "coord_change2=\n" << coord_change2 << "-----------" << endl;
      cout << "backward2=\n" << backward2 << "-----------" << endl;*/
      switch(dim)
      {
      // case 1:MultBtDB1(fes,shape1d,D,U,V);break;
      case 2:
         MultBtDB2int(1,fes,shape1d,shape0d0,shape0d1,D11,U,V);
         MultBtDB2int(2,fes,shape1d,shape0d0,shape0d1,D22,U,V);
         MultBtDB2ext(1,fes,shape1d,shape0d0,shape0d1,coord_change1,backward1,D21,U,V);
         MultBtDB2ext(2,fes,shape1d,shape0d0,shape0d1,coord_change2,backward2,D12,U,V);
         break;
      // case 3:
      //    MultBtDB3(fes,shape1d,shape0d0,shape0d1,coord_change,backward,D11,U,V);
      //    break;
      default: mfem_error("Dimension not yet supported"); break;
      }
   }

   /**
   * Computes V = G^T D G U where G is a tensor product of shape1d and dshape1d. 
   */
   void MultGtDG(const Vector &U, Vector &V)
   {
      
      switch(dim)
      {
      // case 1:MultGtDG1(fes,shape1d,dshape1d,D,U,V);break;
      // case 2:MultGtDG2(fes,shape1d,dshape1d,D,U,V);break;
      // case 3:MultGtDG3(fes,shape1d,dshape1d,D,U,V);break;
      default: mfem_error("Dimension not yet supported"); break;
      }
   }

   /**
   * Computes V = B^T D G U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void MultBtDG(const Vector &U, Vector &V)
   {
      
      switch(dim)
      {
      // case 1:MultBtDG1(fes,shape1d,dshape1d,D,U,V);break;
      // case 2:MultBtDG2(fes,shape1d,dshape1d,D,U,V);break;
      // case 3:MultBtDG3(fes,shape1d,dshape1d,D,U,V);break;
      default: mfem_error("Dimension not yet supported"); break;
      }
   }

   /**
   * Computes V = G^T D B U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void MultGtDB(const Vector &U, Vector &V)
   {
      switch(dim)
      {
      // case 1:MultGtDB1(fes,shape1d,dshape1d,D,U,V);break;
      // case 2:MultGtDB2(fes,shape1d,dshape1d,D,U,V);break;
      // case 3:MultGtDB3(fes,shape1d,dshape1d,D,U,V);break;
      default: mfem_error("Dimension not yet supported"); break;
      }
   }
};


/////////////////////////////////////////////////
//                                             //
//                                             //
//                 EIGEN KERNEL                //
//                                             //
//                                             //
/////////////////////////////////////////////////


enum PAOp { BtDB, BtDG, GtDB, GtDG };

/**
*  A shortcut for EigenOp type names
*/
template <int Dim, PAOp OpName>
class EigenOp;

/**
*  A Domain Partial Assembly Kernel using Eigen's Tensor library
*/
template <int Dim, PAOp OpName >
class EigenDomainPAK
{
public:
   typedef typename EigenOp<Dim,OpName>::Op Op;
   typedef typename Op::DTensor DTensor;
   typedef typename Op::Tensor2d Tensor2d;

private:
   FiniteElementSpace* fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   EigenDomainPAK(FiniteElementSpace* _fes, int order)
   : fes(_fes)
   {
      // Store the 1d shape functions and gradients
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   /**
   *  Sets the dimensions of the tensor D
   */
   void SetSizeD(int* sizes)
   {
      Op::SetSizeD(D, sizes);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   */
   void SetValD(int* ind, double val)
   {
      Op::SetValD(D, ind, val);
   }

   /**
   *  Applies the partial assembly operator
   */
   void Mult(const Vector& U, Vector& V)
   {
      Op::eval(fes, shape1d, dshape1d, D, U, V);
   }
};

template <int Dim>
class EigenMultBtDB;

template <int Dim>
class EigenMultBtDG;

template <int Dim>
class EigenMultGtDB;

template <int Dim>
class EigenMultGtDG;

template <int Dim>
class EigenOp<Dim,PAOp::BtDB>
{
public:
   using Op = EigenMultBtDB<Dim>;
};

template <int Dim>
class EigenOp<Dim,PAOp::BtDG>
{
public:
   using Op = EigenMultBtDG<Dim>;
};

template <int Dim>
class EigenOp<Dim,PAOp::GtDB>
{
public:
   using Op = EigenMultGtDB<Dim>;
};

template <int Dim>
class EigenOp<Dim,PAOp::GtDG>
{
public:
   using Op = EigenMultGtDG<Dim>;
};

template <>
class EigenMultBtDB<2>{
public:
   static const int tensor_dim = 2;
   using Tensor2d = Eigen::Tensor<double,2>;
   using DTensor = Eigen::Tensor<double,2>;

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   static void eval(FiniteElementSpace* fes, Tensor2d const & shape1d, Tensor2d const& dshape1d,
                        DTensor & D, const Vector &U, Vector &V)
   {
      const int dofs1d = shape1d.dimension(0);
      const int dofs = dofs1d*dofs1d;
      const int quads1d = shape1d.dimension(1);
      //const int quads  = quads1d*quads1d;
      array<Eigen::IndexPair<int>, 1> cont_ind;
      for (int e = 0; e < fes->GetNE(); e++){
         Eigen::TensorMap<Eigen::Tensor<double,2>> T0(U.GetData() + e*dofs, dofs1d, dofs1d);
         Eigen::TensorMap<Eigen::Tensor<double,2>> R(V.GetData() + e*dofs, dofs1d, dofs1d);
         // T1 = B.T0
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T1 = T0.contract(shape1d, cont_ind);
         // T2 = B.T1
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T2 = T1.contract(shape1d, cont_ind);
         // T3 = D*T2
         array<int, 2> dims{{quads1d, quads1d}};
         auto T3 = D.chip(e,1).reshape(dims) * T2;
         // T4 = Bt.T3
         cont_ind = { Eigen::IndexPair<int>(0, 1) };
         auto T4 = T3.contract(shape1d, cont_ind);
         // R = Bt.T4
         cont_ind = { Eigen::IndexPair<int>(0, 1) };
         R += T4.contract(shape1d, cont_ind);
      }
   }

   /**
   *  Sets the dimensions of the tensor D
   */
   static void SetSizeD(DTensor& D, int* sizes)
   {
      D = DTensor(sizes[0],sizes[1]);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   */
   static void SetValD(DTensor& D, int* ind, double val)
   {
      D(ind[0],ind[1]) = val;
   }

};

template <>
class EigenMultBtDB<3>{
public:
   static const int tensor_dim = 2;
   using Tensor2d = Eigen::Tensor<double,2>;
   using DTensor = Eigen::Tensor<double,2>;

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   static void eval(FiniteElementSpace* fes, Tensor2d const & shape1d, Tensor2d const& dshape1d,
                        DTensor & D, const Vector &U, Vector &V)
   {
      const int dofs1d = shape1d.dimension(0);
      const int dofs = dofs1d*dofs1d*dofs1d;
      const int quads1d = shape1d.dimension(1);
      //const int quads  = quads1d*quads1d;
      array<Eigen::IndexPair<int>, 1> cont_ind;
      for (int e = 0; e < fes->GetNE(); e++){
         Eigen::TensorMap<Eigen::Tensor<double,3>> T0(U.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
         Eigen::TensorMap<Eigen::Tensor<double,3>> R(V.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
         // T1 = B.T0
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T1 = T0.contract(shape1d, cont_ind);
         // T2 = B.T1
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T2 = T1.contract(shape1d, cont_ind);
         // T3 = B.T2
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T3 = T2.contract(shape1d, cont_ind);
         // T4 = D*T3
         array<int, 3> dims{{quads1d, quads1d, quads1d}};
         auto T4 = D.chip(e,1).reshape(dims) * T3;
         // T5 = Bt.T4
         cont_ind = { Eigen::IndexPair<int>(0, 1) };
         auto T5 = T4.contract(shape1d, cont_ind);
         // T6 = Bt.T5
         cont_ind = { Eigen::IndexPair<int>(0, 1) };
         auto T6 = T5.contract(shape1d, cont_ind);
         // R = Bt.T6
         cont_ind = { Eigen::IndexPair<int>(0, 1) };
         R += T6.contract(shape1d, cont_ind);
      }
   }

   /**
   *  Sets the dimensions of the tensor D
   */
   static void SetSizeD(DTensor& D, int* sizes)
   {
      D = DTensor(sizes[0],sizes[1]);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   */
   static void SetValD(DTensor& D, int* ind, double val)
   {
      D(ind[0],ind[1]) = val;
   }

};

template <>
class EigenMultBtDG<2>{
public:
   static const int tensor_dim = 2;
   using Tensor2d = Eigen::Tensor<double,2>;
   using DTensor = Eigen::Tensor<double,3>;

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   static void eval(FiniteElementSpace* fes, Tensor2d const & shape1d, Tensor2d const& dshape1d,
                        DTensor & D, const Vector &U, Vector &V)
   {
      const int dofs1d = shape1d.dimension(0);
      const int dofs = dofs1d*dofs1d;
      const int quads1d = shape1d.dimension(1);
      //const int quads  = quads1d*quads1d;
      array<Eigen::IndexPair<int>, 1> cont_ind;
      array<int, 2> dims{{quads1d, quads1d}};
      for (int e = 0; e < fes->GetNE(); e++){
         Eigen::TensorMap<Eigen::Tensor<double,2>> T0(U.GetData() + e*dofs, dofs1d, dofs1d);
         Eigen::TensorMap<Eigen::Tensor<double,2>> R(V.GetData() + e*dofs, dofs1d, dofs1d);
         // V = B G U
         // T1 = G.T0
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T1 = T0.contract(dshape1d, cont_ind);
         // T2 = B.T1
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T2 = T1.contract(shape1d, cont_ind);
         // T3 = D*T2
         auto T3 = D.chip(e,2).chip(0,1).reshape(dims) * T2;
         // V = G B U
         // T1 = B.T0
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T1b = T0.contract(shape1d, cont_ind);
         // T2 = G.T1
         cont_ind = { Eigen::IndexPair<int>(0, 0) };
         auto T2b = T1b.contract(dshape1d, cont_ind);
         // V = ( D_0 B G + D_1 G B ) U
         // T3 = D*T2
         auto T3b = T3 + D.chip(e,2).chip(1,1).reshape(dims) * T2b;
         // T4 = Bt.T3
         cont_ind = { Eigen::IndexPair<int>(0, 1) };
         auto T4 = T3b.contract(shape1d, cont_ind);
         // V = B B D ( B G + G B ) U
         // R = Bt.T4
         cont_ind = { Eigen::IndexPair<int>(0, 1) };
         R += T4.contract(shape1d, cont_ind);
      }
   }

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   static void feval(FiniteElementSpace* fes, Tensor2d const & shape1d, Tensor2d const& dshape1d,
                        DTensor & D, const Vector &U, Vector &V)
   {
      using Tensor3d = Eigen::Tensor<double,3>;
      const int nb_e = fes->GetNE();
      const int dofs1d = shape1d.dimension(0);
      const int dofs = dofs1d*dofs1d;
      const int quads1d = shape1d.dimension(1);
      //const int quads  = quads1d*quads1d;
      array<Eigen::IndexPair<int>, 1> cont_ind;
      array<int, 3> dims{{quads1d, quads1d, nb_e}};
      array<int, 3> new_order{{0, 2, 1}};
      Eigen::TensorMap<Eigen::Tensor<double,3>> T0(U.GetData(), dofs1d, dofs1d, nb_e);
      Eigen::TensorMap<Eigen::Tensor<double,3>> R(V.GetData(), dofs1d, dofs1d, nb_e);
      // TODO swap T and B
      // V = B G U
      // T1 = G.T0
      cont_ind = { Eigen::IndexPair<int>(0, 0) };
      Tensor3d T1 = T0.contract(dshape1d, cont_ind).shuffle(new_order);
      // T2 = B.T1
      cont_ind = { Eigen::IndexPair<int>(0, 0) };
      Tensor3d T2 = T1.contract(shape1d, cont_ind).shuffle(new_order);
      // T3 = D*T2
      Tensor3d T3 = D.chip(0,1).reshape(dims) * T2;
      // V = G B U
      // T1 = B.T0
      cont_ind = { Eigen::IndexPair<int>(0, 0) };
      Tensor3d T1b = T0.contract(shape1d, cont_ind).shuffle(new_order);
      // T2 = G.T1
      cont_ind = { Eigen::IndexPair<int>(0, 0) };
      Tensor3d T2b = T1b.contract(dshape1d, cont_ind).shuffle(new_order);
      // V = ( D_0 B G + D_1 G B ) U
      // T3 = D*T2
      Tensor3d T3b = T3 + D.chip(1,1).reshape(dims) * T2b;
      // T4 = Bt.T3
      cont_ind = { Eigen::IndexPair<int>(0, 1) };
      Tensor3d T4 = T3b.contract(shape1d, cont_ind).shuffle(new_order);
      // V = B B D ( B G + G B ) U
      // R = Bt.T4
      cont_ind = { Eigen::IndexPair<int>(0, 1) };
      R += T4.contract(shape1d, cont_ind).shuffle(new_order);
   }


   /**
   *  Sets the dimensions of the tensor D
   */
   static void SetSizeD(DTensor& D, int* sizes)
   {
      D = DTensor(sizes[1],sizes[0],sizes[2]);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   */
   static void SetValD(DTensor& D, int* ind, double val)
   {
      D(ind[1],ind[0],ind[2]) = val;
   }

};

}

#endif //MFEM_PAK