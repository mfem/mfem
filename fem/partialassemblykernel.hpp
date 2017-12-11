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

/**
* Gives the evaluation of the 1d basis functions at one point @param x
*/
template <typename Tensor>
static void ComputeBasis0d(const FiniteElement *fe, double x, Tensor& shape0d)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis0d = tfe->GetBasis1D();

   const int quads0d = 1;
   const int dofs = fe->GetOrder() + 1;

   // We use Matrix and not Vector because we don't want shape0d and dshape0d to have
   // a different treatment than shape1d and dshape1d
   // Well that was before, we might want to reconsider this.
   shape0d  = Tensor(dofs, quads0d);

   Vector u(dofs);
   Vector d(dofs);
   basis0d.Eval(x, u, d);
   for (int i = 0; i < dofs; i++)
   {
      shape0d(i, 0) = u(i);
   }
}

/**
* Gives the evaluation of the 1d basis functions at all quadrature points
*/
template <typename Tensor>
static void ComputeBasis1d(const FiniteElement *fe, int order, Tensor& shape1d, bool backward=false)
{
   const TensorBasisElement* tfe(dynamic_cast<const TensorBasisElement*>(fe));
   const Poly_1D::Basis &basis1d = tfe->GetBasis1D();
   const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);

   const int quads1d = ir1d.GetNPoints();
   const int dofs = fe->GetOrder() + 1;

   shape1d  = Tensor(dofs, quads1d);

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
template <typename Equation, PAOp OpName>
// template <PAOp OpName>
class DummyDomainPAK;

  /////////////////////////////
 // Domain Kernel Interface //
/////////////////////////////

/**
*  A partial assembly Integrator class for domain integrals.
*  Takes an Equation template, that must contain OpName of type PAOp
*  and an evalD function, that receives a res vector, the element
*  transformation and the integration point, and then whatever is needed
*  to compute at the point (Coefficient, VectorCoeffcient, etc...)
*/
template < typename Equation,
            template<typename,PAOp> class IMPL = DummyDomainPAK>
class PADomainInt
: public IMPL<Equation,Equation::OpName>::Op
{
private:
   typedef typename IMPL<Equation,Equation::OpName>::Op Op;

public:
   template <typename... Args>
   PADomainInt(FiniteElementSpace *fes, const int order, Args... args)
   : Op(fes,order,args...)
   {
   }

   /**
   *  Applies the partial assembly operator.
   */
   virtual void AssembleVector(const FiniteElementSpace &fes, const Vector &fun, Vector &vect)
   {
      this->eval(fun, vect);  
   }

};


  ////////////////////////////
 // Available Face Kernels //
////////////////////////////

/**
*  The Operator selector class
*/
template <typename Equation, PAOp Op>
class FacePAK;

  ///////////////////////////
 // Face Kernel Interface //
///////////////////////////

/**
*  A dummy partial assembly Integrator interface class for face integrals
*/
template <typename Equation, template<typename,PAOp> class IMPL = FacePAK>
class PAFaceInt
: public IMPL<Equation,Equation::FaceOpName>::Op
{
private:
   typedef typename IMPL<Equation,Equation::FaceOpName>::Op Op;
public:
   template <typename... Args>
   PAFaceInt(FiniteElementSpace* fes, const int order, Args... args)
   : Op(fes, order, args...)
   {}

   // Perform the action of the BilinearFormIntegrator
   virtual void AssembleVector(const FiniteElementSpace &fes, const Vector &fun, Vector &vect)
   {
      this->EvalInt(fun, vect);
      this->EvalExt(fun, vect);
   }
};

/////////////////////////////////////////////////
//                                             //
//                                             //
//            DUMMY DOMAIN KERNEL              //
//                                             //
//                                             //
/////////////////////////////////////////////////


/**
*  A class that implement a BtDB partial assembly Kernel
*/
template <typename Equation>
class DummyMultBtDB: public BilinearFormIntegrator, Equation
{
public:
   static const int dimD = 2;
   using DTensor = Tensor<dimD,double>;
   using Tensor2d = DenseMatrix;

protected:
   FiniteElementSpace *fes;
   Tensor2d shape1d;
   DTensor D;

public:
   template <typename... Args>
   DummyMultBtDB(FiniteElementSpace* _fes, int order, Args... args)
   : BilinearFormIntegrator(&IntRules.Get(_fes->GetFE(0)->GetGeomType(), order)),
     fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d);
      const int nb_elts = fes->GetNE();
      const int quads  = IntRule->GetNPoints();
      const FiniteElement* fe = fes->GetFE(0);
      int dim = fe->GetDim();
      this->D.setSize(dim,quads,nb_elts);
      double res = 0.0;
      for (int e = 0; e < nb_elts; ++e)
      {
         ElementTransformation *Tr = fes->GetElementTransformation(e);
         for (int k = 0; k < quads; ++k)
         {
            const IntegrationPoint &ip = IntRule->IntPoint(k);
            Tr->SetIntPoint(&ip);
            this->evalD(res, Tr, ip, args...);
            this->D(k,e) = res;
         }
      }
   }

   /**
   * Computes V = B^T D B U where B is a tensor product of shape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultBtDB1(U,V); break;
      case 2:MultBtDB2(U,V); break;
      case 3:MultBtDB3(U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for BtDB in 1d,2d and 3d.
   */
   void MultBtDB1(const Vector &V, Vector &U);
   void MultBtDB2(const Vector &V, Vector &U);
   void MultBtDB3(const Vector &V, Vector &U);  

};


/**
*  A class that implement a GtDG partial assembly Kernel
*/
template <typename Equation>
class DummyMultGtDG: public BilinearFormIntegrator, Equation
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
   template <typename... Args>
   DummyMultGtDG(FiniteElementSpace* _fes, int order, Args... args)
   : BilinearFormIntegrator(&IntRules.Get(_fes->GetFE(0)->GetGeomType(), order)),
     fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
      const int nb_elts = fes->GetNE();
      const int quads  = IntRule->GetNPoints();
      const FiniteElement* fe = fes->GetFE(0);
      int dim = fe->GetDim();
      Tensor<2> res(dim,dim);
      this->D.setSize(dim,quads,nb_elts);
      for (int e = 0; e < nb_elts; ++e)
      {
         ElementTransformation *Tr = fes->GetElementTransformation(e);
         for (int k = 0; k < quads; ++k)
         {
            const IntegrationPoint &ip = IntRule->IntPoint(k);
            Tr->SetIntPoint(&ip);
            this->evalD(res, Tr, ip, args...);
            for (int i = 0; i < dim; ++i)
            {
               for (int j = 0; j < dim; ++j)
               {
                  this->D(i,j,k,e) = res(i,j);
               }
            }
         }
      }
   }

   /**
   * Computes V = G^T D G U where G is a tensor product of shape1d and dshape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultGtDG1(U,V); break;
      case 2:MultGtDG2(U,V); break;
      case 3:MultGtDG3(U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for GtDG in 1d,2d and 3d.
   */
   void MultGtDG1(const Vector &V, Vector &U);
   void MultGtDG2(const Vector &V, Vector &U);
   void MultGtDG3(const Vector &V, Vector &U);

};


/**
*  A class that implement a BtDG partial assembly Kernel
*/
template <typename Equation>
class DummyMultBtDG: public BilinearFormIntegrator, Equation
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
   template <typename... Args>
   DummyMultBtDG(FiniteElementSpace* _fes, int order, Args... args)
   : BilinearFormIntegrator(&IntRules.Get(_fes->GetFE(0)->GetGeomType(), order)),
     fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
      const int nb_elts = fes->GetNE();
      const int quads  = IntRule->GetNPoints();
      const FiniteElement* fe = fes->GetFE(0);
      int dim = fe->GetDim();
      Tensor<1> res(dim);
      this->D.setSize(dim,quads,nb_elts);
      for (int e = 0; e < nb_elts; ++e)
      {
         ElementTransformation *Tr = fes->GetElementTransformation(e);
         for (int k = 0; k < quads; ++k)
         {
            const IntegrationPoint &ip = IntRule->IntPoint(k);
            Tr->SetIntPoint(&ip);
            this->evalD(res, Tr, ip, args...);
            for (int i = 0; i < dim; ++i)
            {
               this->D(i,k,e) = res(i);
            }
         }
      }
   }

   /**
   * Computes V = B^T D G U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultBtDG1(U,V); break;
      case 2:MultBtDG2(U,V); break;
      case 3:MultBtDG3(U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for BtDG in 1d,2d and 3d.
   */
   void MultBtDG1(const Vector &V, Vector &U);
   void MultBtDG2(const Vector &V, Vector &U);
   void MultBtDG3(const Vector &V, Vector &U);

};

/**
*  A class that implement a GtDB partial assembly Kernel
*/
template <typename Equation>
class DummyMultGtDB: public BilinearFormIntegrator, Equation
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
   template <typename... Args>
   DummyMultGtDB(FiniteElementSpace* _fes, int order, Args... args)
   : BilinearFormIntegrator(&IntRules.Get(_fes->GetFE(0)->GetGeomType(), order)),
     fes(_fes), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
      const int nb_elts = fes->GetNE();
      const int quads  = IntRule->GetNPoints();
      const FiniteElement* fe = fes->GetFE(0);
      int dim = fe->GetDim();
      Tensor<1> res(dim);
      this->D.setSize(dim,quads,nb_elts);
      for (int e = 0; e < nb_elts; ++e)
      {
         ElementTransformation *Tr = fes->GetElementTransformation(e);
         for (int k = 0; k < quads; ++k)
         {
            const IntegrationPoint &ip = IntRule->IntPoint(k);
            Tr->SetIntPoint(&ip);
            this->evalD(res, Tr, ip, args...);
            for (int i = 0; i < dim; ++i)
            {
               this->D(i,k,e) = res(i);
            }
         }
      }
   }

   /**
   * Computes V = G^T D B U where B and G are a tensor product of shape1d and dshape1d. 
   */
   void eval(const Vector &U, Vector &V)
   {
      int dim = fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:MultGtDB1(U,V); break;
      case 2:MultGtDB2(U,V); break;
      case 3:MultGtDB3(U,V); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

private:
   /**
   *  The Dummy Kernels for GtDB in 1d,2d and 3d.
   */
   void MultGtDB1(const Vector &V, Vector &U);
   void MultGtDB2(const Vector &V, Vector &U);
   void MultGtDB3(const Vector &V, Vector &U);

};

template <typename Equation>
class DummyDomainPAK<Equation,PAOp::BtDB>{
public:
   using Op = DummyMultBtDB<Equation>;
};

template <typename Equation>
class DummyDomainPAK<Equation,PAOp::BtDG>{
public:
   using Op = DummyMultBtDG<Equation>;
};

template <typename Equation>
class DummyDomainPAK<Equation,PAOp::GtDB>{
public:
   using Op = DummyMultGtDB<Equation>;
};

template <typename Equation>
class DummyDomainPAK<Equation,PAOp::GtDG>{
public:
   using Op = DummyMultGtDG<Equation>;
};





/////////////////////////////////////////////////
//                                             //
//                                             //
//            DUMMY FACE KERNEL                //
//                                             //
//                                             //
/////////////////////////////////////////////////


template <typename Matrix>
void InitIntegrationPoints(Matrix& intPts, const int order,
                           const IntegrationRule& ir1d, const int dim){
   // const IntegrationRule &ir1d = IntRules.Get(Geometry::SEGMENT, order);
   const int quads1d = ir1d.GetNPoints();
   intPts.setSize(pow(quads1d,dim-1),2*dim);
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


  ///////////////////
 // BtDB Operator //
///////////////////


/**
*  A face kernel to compute BtDB
*/
template<typename Equation>
class DummyFaceMultBtDB
: public BilinearFormIntegrator, Equation
{
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

private:
   FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   Tensor2d shape0d0, shape0d1, dshape0d0, dshape0d1;
   DTensor Dint, Dext;
   Tensor<2,IntegrationPoint> intPts;
   KData kernel_data;// Data needed by the Kernel

public:
   template <typename... Args>
   DummyFaceMultBtDB(FiniteElementSpace* _fes, int order, Args... args)
   : BilinearFormIntegrator(&IntRules.Get(_fes->GetFE(0)->GetGeomType(), order)),
     fes(_fes), Dint(), Dext()
   {
      // Store the two 0d shape functions and gradients
      // in x = 0.0
      ComputeBasis0d(fes->GetFE(0), 0.0, shape0d0, dshape0d0);
      // in y = 0.0
      ComputeBasis0d(fes->GetFE(0), 1.0, shape0d1, dshape0d1);
      // Store the 1d shape functions and gradients
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
      const int dim = fes->GetFE(0)->GetDim();
      const IntegrationRule& ir1d = IntRules.Get(Geometry::SEGMENT, order);
      InitIntegrationPoints(intPts, order, ir1d, dim);
      Mesh* mesh = fes->GetMesh();
      const int nb_elts = fes->GetNE();
      const int nb_faces_elt = 2*dim;
      const int nb_faces = mesh->GetNumFaces();
      int geom;
      switch(dim){
         case 1:geom = Geometry::POINT;break;
         case 2:geom = Geometry::SEGMENT;break;
         case 3:geom = Geometry::SQUARE;break;
      }
      const int quads  = IntRules.Get(geom, order).GetNPoints();
      Vector qvec(dim);
      Vector n(dim);
      Dint.setSize(quads,nb_elts,nb_faces_elt);
      Dext.setSize(quads,nb_elts,nb_faces_elt);
      kernel_data.setSize(nb_elts,nb_faces_elt);
      double res11, res21, res22, res12;
      // We have a per face approach for the fluxes, so we should initialize the four different
      // fluxes.
      for (int face = 0; face < nb_faces; ++face)
      {
         int ind_elt1, ind_elt2;
         // We collect the indices of the two elements on the face, element1 is the master element,
         // the one that defines the normal to the face.
         mesh->GetFaceElements(face,&ind_elt1,&ind_elt2);
         int info_elt1, info_elt2;
         // We collect the informations on the face for the two elements.
         mesh->GetFaceInfos(face,&info_elt1,&info_elt2);
         int nb_rot1, nb_rot2;
         int face_id1, face_id2;
         GetIdRotInfo(info_elt1,face_id1,nb_rot1);
         GetIdRotInfo(info_elt2,face_id2,nb_rot2);
         FaceElementTransformations* face_tr = mesh->GetFaceElementTransformations(face);
         kernel_data(ind_elt2,face_id2).indirection = ind_elt1;
         kernel_data(ind_elt2,face_id2).permutation = Permutation2D(face_id1,face_id2);
         kernel_data(ind_elt1,face_id1).indirection = ind_elt2;
         kernel_data(ind_elt1,face_id1).permutation = Permutation2D(face_id2,face_id1);
         for (int k = 0; k < quads; ++k)
         {
            const IntegrationRule& ir = IntRules.Get(geom, order);
            const IntegrationPoint& ip = ir.IntPoint(k);
            if(face_tr->Elem2No!=-1){
               // TODO: maybe pak.IntPoint should return two points, one for each side
               // in case any field is discontinuous
               IntegrationPoint& eip1 = this->IntPoint( face_id1, k );
               // IntegrationPoint& eip2 = pak.IntPoint( face_id2, k );
               face_tr->Face->SetIntPoint( &ip );
               face_tr->Elem1->SetIntPoint( &eip1 );
               CalcOrtho( face_tr->Face->Jacobian(), n );
               // face_tr->Elem2->SetIntPoint( &eip2 );
               // FIXME: Should not take two times eip1!!!
               this->evalFaceD(res11,res21,res22,res12,face_tr,n,eip1,eip1,args...);
               this->Dint(k, ind_elt1, face_id1) = res11;
               this->Dext(k, ind_elt2, face_id2) = res21;
               this->Dint(k, ind_elt2, face_id2) = res22;
               this->Dext(k, ind_elt1, face_id1) = res12;             
            }else{//Boundary face
               // D11(ind) = 0;  
            }
         }
      }
   }

   /**
   *  Returns k-th integration point on the face "face"
   */
   IntegrationPoint& IntPoint(int face, int k)
   {
      return intPts(k,face);
   }

   /**
   *  Computes internal fluxes
   */
   void EvalInt(const Vector& U, Vector& V)
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
   void EvalExt(const Vector& U, Vector& V)
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

template <typename Equation>
class FacePAK<Equation,PAOp::BtDB>{
public:
   using Op = DummyFaceMultBtDB<Equation>;
};

/*template <>
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
};*/

      //////////////////////////////////////
     ///                                ///
    ///                                ///
   /// IMPLEMENTATION OF THE KERNELS  ///
  ///                                ///
 ///                                ///
//////////////////////////////////////

template<typename Equation>
void DummyMultBtDB<Equation>::MultBtDB1(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      shape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_q[k] *= D(k,e); }

      // Q_k1 = dshape_j1_k1 * Q_k1
      shape1d.AddMult(Q, Umat);
   }
}

template<typename Equation>
void DummyMultBtDB<Equation>::MultBtDB2(const Vector &V, Vector &U)
{
   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d*quads1d;

   DenseMatrix QQ(quads1d, quads1d);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, shape1d, QQ);

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_qq[k] *= D(k,e); }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ, DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }
}

template<typename Equation>
void DummyMultBtDB<Equation>::MultBtDB3(const Vector &V, Vector &U)
{
   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   Vector Q(quads1d);
   DenseMatrix QQ(quads1d, quads1d);
   DenseTensor QQQ(quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // QQQ_k1_k2_k3 = shape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
      QQQ = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2) += Q(k1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ(k1, k2, k3) += QQ(k1, k2) * shape1d(j3, k3);
               }
      }

      // QQQ_k1_k2_k3 = Dmat_k1_k2_k3 * QQQ_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      double *data_qqq = QQQ.GetData(0);
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_qqq[k] *= D(k,e); }

      // Apply transpose of the first operator that takes V -> QQQ -- QQQ -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1) += QQQ(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2) += Q(i1) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) += shape1d(i3, k3) * QQ(i1, i2);
               }
      }

      // increment offset
      offset += dofs;
   }  
}

template<typename Equation>
void DummyMultGtDG<Equation>::MultGtDG1(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      dshape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= D(0,0,k,e);
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      dshape1d.AddMult(Q, Umat);
   }   
}

template<typename Equation>
void DummyMultGtDG<Equation>::MultGtDG2(const Vector &V, Vector &U)
{
   const int dim = 2;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * dshape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, dshape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, dshape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind00[] = {0,0,k,e};
         const double D00 = D(0,0,k,e);
         // int ind10[] = {1,0,k,e};
         const double D10 = D(1,0,k,e);
         // int ind01[] = {0,1,k,e};
         const double D01 = D(0,1,k,e);
         // int ind11[] = {1,1,k,e};
         const double D11 = D(1,1,k,e);

         const double q0 = data_qq[0*quads + k];
         const double q1 = data_qq[1*quads + k];

         data_qq[0*quads + k] = D00 * q0 + D01 * q1;
         data_qq[1*quads + k] = D10 * q0 + D11 * q1;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(dshape1d, DQ, Umat);

      // DQ_i2_k1   = dshape_i2_k2 * QQ_1_k1_k2
      // U_i1_i2   += shape_i1_k1  * DQ_i2_k1
      MultABt(dshape1d, QQ(1), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }   
}

template<typename Equation>
void DummyMultGtDG<Equation>::MultGtDG3(const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // QQQ_0_k1_k2_k3 = dshape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_1_k1_k2_k3 = shape_j1_k1  * dshape_j2_k2 * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_2_k1_k2_k3 = shape_j1_k1  * shape_j2_k2  * dshape_j3_k3 * Vmat_j1_j2_j3
      QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * dshape1d(j1, k1);
                  Q(k1, 1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
                  QQ(k1, k2, 1) += Q(k1, 1) * dshape1d(j2, k2);
                  QQ(k1, k2, 2) += Q(k1, 1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
                  QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * shape1d(j3, k3);
                  QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * dshape1d(j3, k3);
               }
      }

      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind00[] = {0,0,k,e};
         const double D00 = D(0,0,k,e);
         // int ind10[] = {1,0,k,e};
         const double D10 = D(1,0,k,e);
         // int ind20[] = {2,0,k,e};
         const double D20 = D(2,0,k,e);
         // int ind01[] = {0,1,k,e};
         const double D01 = D(0,1,k,e);
         // int ind11[] = {1,1,k,e};
         const double D11 = D(1,1,k,e);
         // int ind21[] = {2,1,k,e};
         const double D21 = D(2,1,k,e);
         // int ind02[] = {0,2,k,e};
         const double D02 = D(0,2,k,e);
         // int ind12[] = {1,2,k,e};
         const double D12 = D(1,2,k,e);
         // int ind22[] = {2,2,k,e};
         const double D22 = D(2,2,k,e);

         const double q0 = data_qqq[0*quads + k];
         const double q1 = data_qqq[1*quads + k];
         const double q2 = data_qqq[2*quads + k];

         data_qqq[0*quads + k] = D00 * q0 + D01 * q1 + D02 * q2;
         data_qqq[1*quads + k] = D10 * q0 + D11 * q1 + D12 * q2;
         data_qqq[2*quads + k] = D20 * q0 + D21 * q1 + D22 * q2;
      }

      // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * dshape1d(i1, k1);
                  Q(i1, 1) += QQQ1(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 2) += QQQ2(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
                  QQ(i1, i2, 1) += Q(i1, 1) * dshape1d(i2, k2);
                  QQ(i1, i2, 2) += Q(i1, 2) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) +=
                     QQ(i1, i2, 0) * shape1d(i3, k3) +
                     QQ(i1, i2, 1) * shape1d(i3, k3) +
                     QQ(i1, i2, 2) * dshape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }
}

template<typename Equation>
void DummyMultBtDG<Equation>::MultBtDG1(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      dshape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= D(0,k,e);
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      shape1d.AddMult(Q, Umat);
   }   
}

template<typename Equation>
void DummyMultBtDG<Equation>::MultBtDG2(const Vector &V, Vector &U)
{
   const int dim = 2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * dshape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, dshape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, dshape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      for (int k = 0; k < quads; ++k)
      {
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);

         const double q0 = data_qq[0*quads + k];
         const double q1 = data_qq[1*quads + k];

         data_qq[0*quads + k] = D0 * q0 + D1 * q1;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }
}

template<typename Equation>
void DummyMultBtDG<Equation>::MultBtDG3(const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // QQQ_0_k1_k2_k3 = dshape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_1_k1_k2_k3 = shape_j1_k1  * dshape_j2_k2 * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_2_k1_k2_k3 = shape_j1_k1  * shape_j2_k2  * dshape_j3_k3 * Vmat_j1_j2_j3
      QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * dshape1d(j1, k1);
                  Q(k1, 1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
                  QQ(k1, k2, 1) += Q(k1, 1) * dshape1d(j2, k2);
                  QQ(k1, k2, 2) += Q(k1, 1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
                  QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * shape1d(j3, k3);
                  QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * dshape1d(j3, k3);
               }
      }

      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);
         // int ind2[] = {2,k,e};
         const double D2 = D(2,k,e);

         const double q0 = data_qqq[0*quads + k];
         const double q1 = data_qqq[1*quads + k];
         const double q2 = data_qqq[2*quads + k];

         data_qqq[0*quads + k] = D0 * q0 + D1 * q1 + D2 * q2;
      }

      // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) += QQ(i1, i2, 0) * shape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }
}

template<typename Equation>
void DummyMultGtDB<Equation>::MultGtDB1(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = shape_j1_k1 * V_i1
      shape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= D(0,k,e);
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      dshape1d.AddMult(Q, Umat);
   }    
}

template<typename Equation>
void DummyMultGtDB<Equation>::MultGtDB2(const Vector &V, Vector &U)
{
   const int dim = 2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      //TODO One QQ should be enough
      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1 -- contract in x direction
      // QQ_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      //Can be optimized since this is the same data
      //MultAtB(Vmat, shape1d, DQ);
      //MultAtB(DQ, shape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      //const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // const double D0 = data_d[terms*k + 0];
         // const double D1 = data_d[terms*k + 1];
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);
         const double q = data_qq[0*quads + k];

         data_qq[0*quads + k] = D0 * q;
         data_qq[1*quads + k] = D1 * q;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(dshape1d, DQ, Umat);

      // DQ_i2_k1   = dshape_i2_k2 * QQ_1_k1_k2
      // U_i1_i2   += shape_i1_k1  * DQ_i2_k1
      MultABt(dshape1d, QQ(1), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }    
}

template<typename Equation>
void DummyMultGtDB<Equation>::MultGtDB3(const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*(dim+1)/2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // TODO One QQQ should be enough
      // QQQ_0_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      // QQQ_1_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      // QQQ_2_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
               }
      }

      //TODO insert the three QQQ only here
      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);
         // int ind2[] = {2,k,e};
         const double D2 = D(2,k,e);

         const double q0 = data_qqq[0*quads + k];

         data_qqq[0*quads + k] = D0 * q0;
         data_qqq[1*quads + k] = D1 * q0;
         data_qqq[2*quads + k] = D2 * q0;
      }

      // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * dshape1d(i1, k1);
                  Q(i1, 1) += QQQ1(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 2) += QQQ2(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
                  QQ(i1, i2, 1) += Q(i1, 1) * dshape1d(i2, k2);
                  QQ(i1, i2, 2) += Q(i1, 2) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) +=
                     QQ(i1, i2, 0) * shape1d(i3, k3) +
                     QQ(i1, i2, 1) * shape1d(i3, k3) +
                     QQ(i1, i2, 2) * dshape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }   
}

/// FACE KERNELS

template <typename Equation>
void DummyFaceMultBtDB<Equation>::MultBtDBintX(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   DTensor T0(U.GetData(),dofs1d,dofs1d,nbe);
   DTensor R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1> T1(dofs1d),T2(quads1d),T3(dofs1d);
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i2 = 0; i2 < dofs1d; ++i2)
      {
         T1(i2) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i2) += B0d(i1,0) * T0(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T2(k2) += B(i2,k2) * T1(i2);
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = D(k2,e,face_id) * T2(k2);
      }
      // T3.zero();
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         T3(j2) = 0.0;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T3(j2) += B(j2,k2) * T2(k2);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0d(j1,0) * T3(j2);
         }
      }
   }
}

template <typename Equation>
void DummyFaceMultBtDB<Equation>::MultBtDBintY(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   DTensor T0(U.GetData(),dofs1d,dofs1d,nbe);
   DTensor R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1> T1(dofs1d),T2(quads1d),T3(dofs1d);
   //T1_i1 = B0d^i2 U_i1i2
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i1 = 0; i1 < dofs1d; ++i1)
      {
         T1(i1) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T1(i1) += B0d(i2,0) * T0(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T2(k1) += B(i1,k1) * T1(i1);
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = D(k1,e,face_id) * T2(k1);
      }
      // T3.zero();
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         T3(j1) = 0.0;
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(j1) += B(j1,k1) * T2(k1);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0d(j2,0) * T3(j1);
         }
      }
   }
}

template <typename Equation>
void DummyFaceMultBtDB<Equation>::Permutation(int face_id, int nbe, int dofs1d, KData& kernel_data,
                                    const Tensor3d& T0, Tensor3d& T0p)
{
   for (int e = 0; e < nbe; ++e)
   {
      const int trial = kernel_data(e,face_id).indirection;
      const int permutation = kernel_data(e,face_id).permutation;
      if(trial!=-1)
      {
         if(permutation==0)
         {
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  T0p(i1,i2,e) = T0(i1,i2,trial);
               }
            }
         }else if(permutation==1){
            for (int i2 = 0, j1 = dofs1d-1; i2 < dofs1d; ++i2, --j1)
            {
               for (int i1 = 0, j2 = 0; i1 < dofs1d; ++i1, ++j2)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
         }else if(permutation==2){
            for (int i2 = 0, j2 = dofs1d-1; i2 < dofs1d; ++i2, --j2)
            {
               for (int i1 = 0, j1 = dofs1d-1; i1 < dofs1d; ++i1, --j1)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
         }else if(permutation==3){
            // cout << "perm" << permutation << endl;
            for (int i2 = 0, j1 = 0; i2 < dofs1d; ++i2, ++j1)
            {
               for (int i1 = 0, j2 = dofs1d-1; i1 < dofs1d; ++i1, --j2)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
         }else{
            mfem_error("This permutation id does not exist");
         }
      }else{
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T0p(i1,i2,e) = 0.0;
            }
         }
      }
   }
}

template <typename Equation>
void DummyFaceMultBtDB<Equation>::MultBtDBextX(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dTest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor3d T0(U.GetData(),dofs1d,dofs1d,nbe);
   Tensor3d R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1,double> T1(dofs1d),T2(quads1d),T3(dofs1d);
   // Indirections
   Tensor3d T0p(dofs1d,dofs1d,nbe);
   Permutation(face_id,nbe,dofs1d,kernel_data,T0,T0p);
   //T1_i2 = B0d^i1 U_i1i2
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i2 = 0; i2 < dofs1d; ++i2)
      {
         T1(i2) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i2) += B0dTrial(i1,0) * T0p(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T2(k2) += B(i2,k2) * T1(i2);
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = D(k2,e,face_id) * T2(k2);
      }
      // T3.zero();
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         T3(j2) = 0.0;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T3(j2) += B(j2,k2) * T2(k2);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0dTest(j1,0) * T3(j2);
         }
      }
   }
}

template <typename Equation>
void DummyFaceMultBtDB<Equation>::MultBtDBextY(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dTest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   DTensor T0(U.GetData(),dofs1d,dofs1d,nbe);
   DTensor R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1,double> T1(dofs1d),T2(quads1d),T3(dofs1d);
   // Indirections
   DTensor T0p(dofs1d,dofs1d,nbe);
   Permutation(face_id,nbe,dofs1d,kernel_data,T0,T0p);
   //T1_i1 = B0d^i2 U_i1i2
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i1 = 0; i1 < dofs1d; ++i1)
      {
         T1(i1) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T1(i1) += B0dTrial(i2,0) * T0p(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T2(k1) += B(i1,k1) * T1(i1);
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = D(k1,e,face_id) * T2(k1);
      }
      // T3.zero();
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         T3(j1) = 0.0;
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(j1) += B(j1,k1) * T2(k1);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0dTest(j2,0) * T3(j1);
         }
      }
   }
}

}

#endif //MFEM_PAK