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
#include "dgfacefunctions.hpp"
#include <iostream>

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
//        PARTIAL ASSEMBLY INTEGRATORS         //
//                                             //
//                                             //
/////////////////////////////////////////////////

/**
*  The different operators available for the Kernels
*/
enum PAOp { BtDB, BtDG, GtDB, GtDG };

  //////////////////////////////
 // Available Domain Kernels //
//////////////////////////////

/**
*  A shortcut for DummyDomainPAK type names
*/
template <PAOp OpName>
class DummyDomainPAK;

  /////////////////////////////
 // Domain Kernel Interface //
/////////////////////////////

/**
*  A dummy partial assembly Integrator interface class for domain integrals
*/
template <PAOp OpName, template<PAOp> class IMPL = DummyDomainPAK >
class PADomainIntegrator: protected IMPL<OpName>::Op
{
public:
   typedef typename IMPL<OpName>::Op Op;
   typedef typename Op::DTensor DTensor;
   typedef typename Op::Tensor2d Tensor2d;

public:
   // ToFix: the initialization of D should be done in the Kernel
   PADomainIntegrator(FiniteElementSpace *fes, int order)
   // : fes(_fes), D()
   :  Op(fes,order)
   {
   }

   /**
   *  Sets the dimensions of the tensor D
   */
   template <typename... Args>
   void SetSizeD(Args... args)
   {
      this->D.setSize(args...);
   }

   /*
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size
   *  the dimension of the tensor D.
   */
   template<typename... Args>
   void SetValD(double val, Args... args)
   {
      this->D(args...) = val;
   }

   /**
   *  Applies the partial assembly operator. Not really necessary,
   *  just to make the interface more clear.
   */
   void Mult(const Vector& U, Vector& V)
   {
      this->eval(U, V);
   }

};

  ////////////////////////////
 // Available Face Kernels //
////////////////////////////

/**
*  The Operator selector class
*/
template <PAOp Op>
class FacePAK;

  ///////////////////////////
 // Face Kernel Interface //
///////////////////////////

/**
*  A dummy partial assembly Integrator interface class for face integrals
*/
template <PAOp OpName, template<PAOp> class IMPL = FacePAK>
class PAFaceIntegrator
{
public:
   typedef typename IMPL<OpName>::Op Op;
   typedef typename Op::KData KData;
   typedef typename Op::DTensor DTensor;
   typedef typename Op::Tensor2d Tensor2d;

private:
   FiniteElementSpace *fes;
   const int dim;
   Tensor2d shape1d, dshape1d;
   Tensor2d shape0d0, shape0d1, dshape0d0, dshape0d1;
   DTensor Dint, Dext;
   DummyMatrix<IntegrationPoint> intPts;
   KData kernel_data;// Data needed by the Kernel

public:
   /**
   *  Creates a Partial Assembly Face Integrator given a finite element space and the total
   *  order of the functions to integrate.
   */
   PAFaceIntegrator(FiniteElementSpace* _fes, int order)
   : fes(_fes), dim(fes->GetFE(0)->GetDim())
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

   /**
   *  Set the sizes of Dint and Dext tensors, and of KData.
   *  Contains: quads, nb_elemets, nb_faces_elt and eventually two times dim
   */
   void SetSize(int (&sizes)[Op::dimD])
   {
      Op::SetSize(Dint, Dext, kernel_data, sizes);
   }

   /**
   *  Set a value of Dint tensor.
   *  The ind array contains for:
   *  -BtDB: quad, element, face_id
   *  -BtDG and GtDB: dim, quad, element, face_id
   *  -GtDG: dim, dim, quad, element, face_id
   */
   void SetValDint(int (&ind)[Op::dimD], double val)
   {
      Op::SetValDint(Dint, ind, val);
   }

   /**
   *  Set a value of Dext tensor.
   *  The ind array contains for:
   *  -BtDB: quad, elt_trial, face_id_trial, elt_test, face_id_test
   *  -BtDG and GtDB: dim, quad, elt_trial, face_id_trial, elt_test, face_id_test
   *  -GtDG: dim, dim, quad, elt_trial, face_id_trial, elt_test, face_id_test
   */
   void SetValDext(int (&ind)[Op::dimD+2], double val)
   {
      Op::SetValDext(Dext, kernel_data, ind, val);
   }

   /**
   *  Returns k-th integration point on the face "face"
   */
   IntegrationPoint& IntPoint(int face, int k)
   {
      return intPts(k,face);
   }

   /**
   *  Calls the Operators to compute internal and external fluxes.
   */
   void Mult(const Vector& U, Vector& V)
   {
      Op::EvalInt(fes, shape1d, dshape1d, shape0d0, shape0d1, dshape0d0, dshape0d1,
                  Dint, U, V);
      Op::EvalExt(fes, shape1d, dshape1d, shape0d0, shape0d1, dshape0d0, dshape0d1,
                  kernel_data, Dext, U, V);
   }
};


/////////////////////////////////////////////////
//                                             //
//                                             //
//            DUMMY DOMAIN KERNEL              //
//                                             //
//                                             //
/////////////////////////////////////////////////


class DummyMultBtDB
{
public:
   static const int dimD = 2;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;

protected:
   FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   DummyMultBtDB(FiniteElementSpace* _fes, int order)
   : fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultBtDB1(fes,shape1d,D,U,V); break;
      case 2:MultBtDB2(fes,shape1d,D,U,V); break;
      case 3:MultBtDB3(fes,shape1d,D,U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for BtDB in 1d,2d and 3d.
   */
   static void MultBtDB1(FiniteElementSpace* fes, const Tensor2d & shape1d,
      DTensor & D, const Vector &V, Vector &U);
   static void MultBtDB2(FiniteElementSpace* fes, const Tensor2d & shape1d,
      DTensor & D, const Vector &V, Vector &U);
   static void MultBtDB3(FiniteElementSpace* fes, const Tensor2d & shape1d,
      DTensor & D, const Vector &V, Vector &U);  

};

class DummyMultGtDG
{
public:
   static const int dimD = 4;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;

protected:
   FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   DummyMultGtDG(FiniteElementSpace* _fes, int order)
   : fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   /**
   * Computes V = G^T D G U where G is a tensor product of shape1d and dshape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultGtDG1(fes,shape1d,dshape1d,D,U,V); break;
      case 2:MultGtDG2(fes,shape1d,dshape1d,D,U,V); break;
      case 3:MultGtDG3(fes,shape1d,dshape1d,D,U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for GtDG in 1d,2d and 3d.
   */
   static void MultGtDG1(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);
   static void MultGtDG2(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);
   static void MultGtDG3(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);

};

class DummyMultBtDG
{
public:
   static const int dimD = 3;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;

protected:
   FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   DummyMultBtDG(FiniteElementSpace* _fes, int order)
   : fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   /**
   * Computes V = B^T D G U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultBtDG1(fes,shape1d,dshape1d,D,U,V); break;
      case 2:MultBtDG2(fes,shape1d,dshape1d,D,U,V); break;
      case 3:MultBtDG3(fes,shape1d,dshape1d,D,U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for BtDG in 1d,2d and 3d.
   */
   static void MultBtDG1(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);
   static void MultBtDG2(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);
   static void MultBtDG3(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);

};

class DummyMultGtDB
{
public:
   static const int dimD = 3;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;

protected:
   FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   DummyMultGtDB(FiniteElementSpace* _fes, int order)
   : fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   /**
   * Computes V = G^T D B U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultGtDB1(fes,shape1d,dshape1d,D,U,V); break;
      case 2:MultGtDB2(fes,shape1d,dshape1d,D,U,V); break;
      case 3:MultGtDB3(fes,shape1d,dshape1d,D,U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for GtDB in 1d,2d and 3d.
   */
   static void MultGtDB1(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);
   static void MultGtDB2(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);
   static void MultGtDB3(FiniteElementSpace* fes, const Tensor2d & shape1d,
      const Tensor2d & dshape1d, DTensor & D, const Vector &V, Vector &U);

};

template <>
class DummyDomainPAK<PAOp::BtDB>{
public:
   using Op = DummyMultBtDB;
};

template <>
class DummyDomainPAK<PAOp::BtDG>{
public:
   using Op = DummyMultBtDG;
};

template <>
class DummyDomainPAK<PAOp::GtDB>{
public:
   using Op = DummyMultGtDB;
};

template <>
class DummyDomainPAK<PAOp::GtDG>{
public:
   using Op = DummyMultGtDG;
};


/////////////////////////////////////////////////
//                                             //
//                                             //
//             DUMMY FACE KERNEL               //
//                                             //
//                                             //
/////////////////////////////////////////////////



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
//            DUMMY FACE KERNEL 2              //
//                                             //
//                                             //
/////////////////////////////////////////////////

  ///////////////////
 // BtDB Operator //
///////////////////

/**
*  A face kernel to compute BtDB
*/
class DummyFaceMultBtDB{

/**
*  A structure that stores the indirection and permutation on dofs for an external flux on a face.
*/
struct PermIndir{
   int indirection;
   int permutation;   
};

public:
   static const int dimD = 3;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;
   using Tensor3d = Tensor<3,double>;
   using KData = Tensor<2,PermIndir>;

   /**
   *  Sets the dimensions of the tensor D
   */
   static void SetSize(DTensor& Dint, DTensor& Dext, KData& kernel_data, int* sizes)
   {
      Dint.setSize(sizes[0],sizes[1],sizes[2]);
      Dext.setSize(sizes[0],sizes[1],sizes[2]);
      int nb_elts  = sizes[1];
      int nb_faces = sizes[2];
      kernel_data.setSize(nb_elts,nb_faces);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt, face_id
   */
   static void SetValDint(DTensor& Dint, int (&ind)[dimD], double val)
   {
      int quad(ind[0]), elt(ind[1]), face_id(ind[2]);
      Dint(quad,elt,face_id) = val;
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt_trial, face_id_trial, elt_test, face_id_test
   */
   static void SetValDext(DTensor& Dext, KData& kernel_data,
                           int (&ind)[dimD+2], double val)
   {
      int quad(ind[0]), elt_trial(ind[1]), face_id_trial(ind[2]), elt_test(ind[3]), face_id_test(ind[4]);
      // We do the indirections and permutations at the beginning so that each element receives
      // one and only one flux per face. So this is a per face for test element approach.
      // TODO should not be here, done too many times
      kernel_data(elt_test,face_id_test).indirection = elt_trial;
      // cout << "face_id_trial=" << face_id_trial << ", face_id_test=" << face_id_test
      //       << ", permutation=" << Permutation2D(face_id_trial,face_id_test) << endl;
      kernel_data(elt_test,face_id_test).permutation = Permutation2D(face_id_trial,face_id_test);
      Dext(quad,elt_test,face_id_test) = val;
   }

   /**
   *  Computes internal fluxes
   */
   static void EvalInt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        DTensor& Dint, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id = 2;
      MultBtDBintY(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // South Faces
      face_id = 0;
      MultBtDBintY(fes,shape1d,shape0d0,Dint,face_id,U,V);
      // East Faces
      face_id = 1;
      MultBtDBintX(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // West Faces
      face_id = 3;
      MultBtDBintX(fes,shape1d,shape0d0,Dint,face_id,U,V);
   }

   /**
   *  Computes external fluxes
   */
   static void EvalExt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        KData& kernel_data ,DTensor& Dext, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id_test = 2;
      MultBtDBextY(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // South Faces
      face_id_test = 0;
      MultBtDBextY(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
      // East Faces
      face_id_test = 1;
      MultBtDBextX(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // West Faces
      face_id_test = 3;
      MultBtDBextX(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);      
   }

private:
   static void MultBtDBintX(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBintY(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextX(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextY(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void Permutation(int face_id, int nbe, int dofs1d, KData& kernel_data,
                        const Tensor3d& T0, Tensor3d& T0p);
};

  ///////////////////
 // GtDB Operator //
///////////////////

/**
*  A face kernel to compute GtDB
*/
class DummyFaceMultGtDB{

/**
*  A structure that stores the indirection and permutation on dofs for an external flux on a face.
*/
struct PermIndir{
   int indirection;
   int permutation;  
};

public:
   static const int dimD = 4;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;
   using Tensor3d = Tensor<3,double>;
   using KData = Tensor<2,PermIndir>;

   /**
   *  Sets the dimensions of the tensor D
   */
   static void SetSize(DTensor& Dint, DTensor& Dext, KData& kernel_data, int* sizes)
   {
      int dim(sizes[0]), quad(sizes[1]), nb_elts(sizes[2]), nb_faces(sizes[3]);
      Dint.setSize(dim,quad,nb_elts,nb_faces);
      Dext.setSize(dim,quad,nb_elts,nb_faces);
      kernel_data.setSize(nb_elts,nb_faces);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt, face_id
   */
   static void SetValDint(DTensor& Dint, int (&ind)[dimD], double val)
   {
      int dim(ind[0]), quad(ind[1]), elt(ind[2]), face_id(ind[3]);
      Dint(dim,quad,elt,face_id) = val;
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt_trial, face_id_trial, elt_test, face_id_test
   */
   static void SetValDext(DTensor& Dext, KData& kernel_data,
                           int (&ind)[dimD+2], double val)
   {
      int dim(ind[0]), quad(ind[1]), elt_trial(ind[2]), face_id_trial(ind[3]), elt_test(ind[4]), face_id_test(ind[5]);
      // We do the indirections and permutations at the beginning so that each element receives
      // one and only one flux per face. So this is a per face for test element approach.
      // TODO should not be here, done too many times
      kernel_data(elt_test,face_id_test).indirection = elt_trial;
      // cout << "face_id_trial=" << face_id_trial << ", face_id_test=" << face_id_test
      //       << ", permutation=" << Permutation2D(face_id_trial,face_id_test) << endl;
      kernel_data(elt_test,face_id_test).permutation = Permutation2D(face_id_trial,face_id_test);
      Dext(dim,quad,elt_test,face_id_test) = val;
   }

   /**
   *  Computes internal fluxes
   */
   static void EvalInt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        DTensor& Dint, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id = 2;
      MultBtDBintY(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // South Faces
      face_id = 0;
      MultBtDBintY(fes,shape1d,shape0d0,Dint,face_id,U,V);
      // East Faces
      face_id = 1;
      MultBtDBintX(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // West Faces
      face_id = 3;
      MultBtDBintX(fes,shape1d,shape0d0,Dint,face_id,U,V);
   }

   /**
   *  Computes external fluxes
   */
   static void EvalExt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        KData& kernel_data ,DTensor& Dext, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id_test = 2;
      MultBtDBextY(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // South Faces
      face_id_test = 0;
      MultBtDBextY(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
      // East Faces
      face_id_test = 1;
      MultBtDBextX(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // West Faces
      face_id_test = 3;
      MultBtDBextX(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);      
   }

private:
   static void MultBtDBintX(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBintY(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextX(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextY(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void Permutation(int face_id, int nbe, int dofs1d, KData& kernel_data,
                        const Tensor3d& T0, Tensor3d& T0p);
};

  ///////////////////
 // BtDG Operator //
///////////////////

/**
*  A face kernel to compute BtDG
*/
class DummyFaceMultBtDG{

/**
*  A structure that stores the indirection and permutation on dofs for an external flux on a face.
*/
struct PermIndir{
   int indirection;
   int permutation;  
};

public:
   static const int dimD = 4;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;
   using Tensor3d = Tensor<3,double>;
   using KData = Tensor<2,PermIndir>;

   /**
   *  Sets the dimensions of the tensor D
   */
   static void SetSize(DTensor& Dint, DTensor& Dext, KData& kernel_data, int* sizes)
   {
      int dim(sizes[0]), quad(sizes[1]), nb_elts(sizes[2]), nb_faces(sizes[3]);
      Dint.setSize(dim,quad,nb_elts,nb_faces);
      Dext.setSize(dim,quad,nb_elts,nb_faces);
      kernel_data.setSize(nb_elts,nb_faces);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt, face_id
   */
   static void SetValDint(DTensor& Dint, int (&ind)[dimD], double val)
   {
      int dim(ind[0]), quad(ind[1]), elt(ind[2]), face_id(ind[3]);
      Dint(dim,quad,elt,face_id) = val;
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt_trial, face_id_trial, elt_test, face_id_test
   */
   static void SetValDext(DTensor& Dext, KData& kernel_data,
                           int (&ind)[dimD+2], double val)
   {
      int dim(ind[0]), quad(ind[1]), elt_trial(ind[2]), face_id_trial(ind[3]), elt_test(ind[4]), face_id_test(ind[5]);
      // We do the indirections and permutations at the beginning so that each element receives
      // one and only one flux per face. So this is a per face for test element approach.
      // TODO should not be here, done too many times
      kernel_data(elt_test,face_id_test).indirection = elt_trial;
      // cout << "face_id_trial=" << face_id_trial << ", face_id_test=" << face_id_test
      //       << ", permutation=" << Permutation2D(face_id_trial,face_id_test) << endl;
      kernel_data(elt_test,face_id_test).permutation = Permutation2D(face_id_trial,face_id_test);
      Dext(dim,quad,elt_test,face_id_test) = val;
   }

   /**
   *  Computes internal fluxes
   */
   static void EvalInt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        DTensor& Dint, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id = 2;
      MultBtDBintY(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // South Faces
      face_id = 0;
      MultBtDBintY(fes,shape1d,shape0d0,Dint,face_id,U,V);
      // East Faces
      face_id = 1;
      MultBtDBintX(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // West Faces
      face_id = 3;
      MultBtDBintX(fes,shape1d,shape0d0,Dint,face_id,U,V);
   }

   /**
   *  Computes external fluxes
   */
   static void EvalExt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        KData& kernel_data ,DTensor& Dext, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id_test = 2;
      MultBtDBextY(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // South Faces
      face_id_test = 0;
      MultBtDBextY(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
      // East Faces
      face_id_test = 1;
      MultBtDBextX(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // West Faces
      face_id_test = 3;
      MultBtDBextX(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);      
   }

private:
   static void MultBtDBintX(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBintY(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextX(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextY(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void Permutation(int face_id, int nbe, int dofs1d, KData& kernel_data,
                        const Tensor3d& T0, Tensor3d& T0p);
};

  ///////////////////
 // GtDG Operator //
///////////////////

/**
*  A face kernel to compute GtDG
*/
class DummyFaceMultGtDG{

/**
*  A structure that stores the indirection and permutation on dofs for an external flux on a face.
*/
struct PermIndir{
   int indirection;
   int permutation;  
};

public:
   static const int dimD = 5;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;
   using Tensor3d = Tensor<3,double>;
   using KData = Tensor<2,PermIndir>;

   /**
   *  Sets the dimensions of the tensor D
   */
   static void SetSize(DTensor& Dint, DTensor& Dext, KData& kernel_data, int* sizes)
   {
      int dim1(sizes[0]), dim2(sizes[1]), quad(sizes[2]), nb_elts(sizes[3]), nb_faces(sizes[4]);
      Dint.setSize(dim1,dim2,quad,nb_elts,nb_faces);
      Dext.setSize(dim1,dim2,quad,nb_elts,nb_faces);
      kernel_data.setSize(nb_elts,nb_faces);
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt, face_id
   */
   static void SetValDint(DTensor& Dint, int (&ind)[dimD], double val)
   {
      int dim1(ind[0]), dim2(ind[1]), quad(ind[2]), elt(ind[3]), face_id(ind[4]);
      Dint(dim1,dim2,quad,elt,face_id) = val;
   }

   /**
   *  Sets the value at indice @a ind to @a val. @a ind is a raw integer array of size 2.
   *  The ind array contains: quad, elt_trial, face_id_trial, elt_test, face_id_test
   */
   static void SetValDext(DTensor& Dext, KData& kernel_data,
                           int (&ind)[dimD+2], double val)
   {
      int dim1(ind[0]), dim2(ind[1]), quad(ind[2]), elt_trial(ind[3]),
         face_id_trial(ind[4]), elt_test(ind[5]), face_id_test(ind[6]);
      // We do the indirections and permutations at the beginning so that each element receives
      // one and only one flux per face. So this is a per face for test element approach.
      // TODO should not be here, done too many times
      kernel_data(elt_test,face_id_test).indirection = elt_trial;
      // cout << "face_id_trial=" << face_id_trial << ", face_id_test=" << face_id_test
      //       << ", permutation=" << Permutation2D(face_id_trial,face_id_test) << endl;
      kernel_data(elt_test,face_id_test).permutation = Permutation2D(face_id_trial,face_id_test);
      Dext(dim1,dim2,quad,elt_test,face_id_test) = val;
   }

   /**
   *  Computes internal fluxes
   */
   static void EvalInt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        DTensor& Dint, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id = 2;
      MultBtDBintY(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // South Faces
      face_id = 0;
      MultBtDBintY(fes,shape1d,shape0d0,Dint,face_id,U,V);
      // East Faces
      face_id = 1;
      MultBtDBintX(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // West Faces
      face_id = 3;
      MultBtDBintX(fes,shape1d,shape0d0,Dint,face_id,U,V);
   }

   /**
   *  Computes external fluxes
   */
   static void EvalExt(FiniteElementSpace* fes, Tensor2d& shape1d, Tensor2d& dshape1d,
                        Tensor2d& shape0d0, Tensor2d& shape0d1,
                        Tensor2d& dshape0d0, Tensor2d& dshape0d1,
                        KData& kernel_data ,DTensor& Dext, const Vector& U, Vector& V)
   {
      // North Faces
      int face_id_test = 2;
      MultBtDBextY(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // South Faces
      face_id_test = 0;
      MultBtDBextY(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
      // East Faces
      face_id_test = 1;
      MultBtDBextX(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // West Faces
      face_id_test = 3;
      MultBtDBextX(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);      
   }

private:
   static void MultBtDBintX(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBintY(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextX(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultBtDBextY(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void Permutation(int face_id, int nbe, int dofs1d, KData& kernel_data,
                        const Tensor3d& T0, Tensor3d& T0p);
};


template <>
class FacePAK<PAOp::BtDB>{
public:
   using Op = DummyFaceMultBtDB;
};

template <>
class FacePAK<PAOp::GtDB>{
public:
   using Op = DummyFaceMultGtDB;
};

template <>
class FacePAK<PAOp::BtDG>{
public:
   using Op = DummyFaceMultGtDB;
};

template <>
class FacePAK<PAOp::GtDG>{
public:
   using Op = DummyFaceMultGtDB;
};


}

#endif //MFEM_PAK