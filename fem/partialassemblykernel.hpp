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
#include "domainkernels.hpp"
#include "facekernels.hpp"
#include "solverkernels.hpp"
#include <iostream>

namespace mfem
{

/////////////////////////////////////////////////
//                                             //
//                                             //
//        PARTIAL ASSEMBLY INTEGRATORS         //
//                                             //
//                                             //
/////////////////////////////////////////////////


  /////////////////////////////
 // Domain Kernel Interface //
/////////////////////////////

struct ElementInfo
{
   int dim;
   int k;
   int e;
   ElementTransformation* tr;
   IntegrationPoint ip;
   Tensor<2> J_ek;
}

/**
*  A partial assembly Integrator class for domain integrals.
*  Takes an 'Equation' template parameter, that must contain 'OpName' of
*  type 'PAOp' and a function named 'evalD', that receives a 'res' vector,
*  the element transformation and the integration point, and then whatever
*  is needed to compute at the point (Coefficient, VectorCoeffcient, etc...).
*  The 'IMPL' template parameter allows to switch between different implementations
*  of the tensor contraction kernels.
*/
template < typename Equation,
            template<typename,PAOp> class IMPL = DomainMult>
class PADomainInt
: public LinearFESpaceIntegrator, public IMPL<Equation,Equation::OpName>
{
private:
   typedef IMPL<Equation,Equation::OpName> Op;

public:
   /**
   *  The constructor is templated so that the argument needed for 'evalD' can be
   *  packed arbitrarily ('evalD' with the corresponding signature must exist).
   */
   template <typename Args>
   PADomainInt(FiniteElementSpace *fes, const int order, const Args& args)
   : LinearFESpaceIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     Op(fes,order,args)
   {
      const int nb_elts = fes->GetNE();
      const int quads  = IntRule->GetNPoints();
      const FiniteElement* fe = fes->GetFE(0);
      const int dim = fe->GetDim();
      this->InitD(dim,quads,nb_elts);
      Tensor<1> Jac1D(dim*dim*quads*nb_elts);
      EvalJacobians(dim,fes,order,Jac1D);
      Tensor<4> Jac(Jac1D.getData(),dim,dim,quads,nb_elts);
      for (int e = 0; e < nb_elts; ++e)
      {
         ElementTransformation *Tr = fes->GetElementTransformation(e);
         for (int k = 0; k < quads; ++k)
         {
            Tensor<2> J_ek(&Jac(0,0,k,e),dim,dim);
            const IntegrationPoint &ip = IntRule->IntPoint(k);
            Tr->SetIntPoint(&ip);
            this->evalEq(dim, k, e, Tr, ip, J_ek, args);
         }
      }
   }

   const typename Op::DTensor& getD() const
   {
      return Op::getD();
   }

   /**
   *  Applies the partial assembly operator.
   */
   virtual void AddMult(const Vector &fun, Vector &vect)
   {
      int dim = this->fes->GetFE(0)->GetDim();
      switch(dim)
      {
      case 1:this->Mult1d(fun,vect); break;
      case 2:this->Mult2d(fun,vect); break;
      case 3:this->Mult3d(fun,vect); break;
      default: mfem_error("More than # dimension not yet supported"); break;
      }
   }

};

  ///////////////////////////
 // Face Kernel Interface //
///////////////////////////

struct FaceInfo
{
   int dim; // The problem dimension
   int k1, k2; // The indices of 
   IntegrationPoint eip1, eip2; // The integration points on each element
   Vector* normal;  // The normal to the face
   int ind_elt1, ind_elt2; // The indices of the elements
   int face_id1, face_id2; // The face ID for the face according to each element
   FaceElementTransformations* face_tr; // The Face transformation
   Tensor<2> J_e1, J_e2; // The Jacobians for each element at their respective quadrature point
};

/**
*  A partial assembly Integrator interface class for face integrals.
*  The template parameters have the same role as for 'PADomainInt'.
*/
template <typename Equation, template<typename,PAOp> class IMPL = FaceMult>
class PAFaceInt
: public LinearFESpaceIntegrator, public IMPL<Equation,Equation::FaceOpName>
{
private:
   typedef IMPL<Equation,Equation::FaceOpName> Op;

public:
   template <typename Args>
   PAFaceInt(FiniteElementSpace* fes, const int order, Args& args)
   : LinearFESpaceIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     Op(fes, order, args)
   {
      const int dim = fes->GetFE(0)->GetDim();
      const int quads1d = fes->GetNQuads1d(order);
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
      const IntegrationRule& ir = IntRules.Get(geom, order);
      const int quads  = ir.GetNPoints();
      Vector qvec(dim);
      Tensor<1> normal(dim);
      Vector n(normal.getData(),dim);
      // Vector n(dim);
      this->init(dim,quads,nb_elts,nb_faces_elt);
      // !!! Should not be recomputed... !!!
      Tensor<1> Jac1D(dim*dim*quads*quads1d*nb_elts);
      EvalJacobians(dim,fes,order,Jac1D);
      Tensor<4> Jac(Jac1D.getData(),dim,dim,quads*quads1d,nb_elts);
      // !!!                             !!!
      // We have a per face approach for the fluxes
      for (int face = 0; face < nb_faces; ++face)
      {
         int ind_elt1, ind_elt2;
         int face_id1, face_id2;
         int nb_rot1, nb_rot2;
         GetFaceInfo(mesh, face, ind_elt1, ind_elt2, face_id1, face_id2, nb_rot1, nb_rot2);
         FaceElementTransformations* face_tr = mesh->GetFaceElementTransformations(face);
         int perm1, perm2;
         // cout << "ind_elt1=" << ind_elt1 << ", face_id1=" << face_id1 << ", nb_rot1=" << nb_rot1 << ", ind_elt2=" << ind_elt2 << ", face_id2=" << face_id2 << ", nb_rot2=" << nb_rot2 << endl;
         for (int kf = 0; kf < quads; ++kf)
         {
            const IntegrationPoint& ip = ir.IntPoint(kf);
            if(ind_elt2!=-1){//Not a boundary face
               Tensor<1,int> ind_f1(dim-1), ind_f2(dim-1);
               // We compute the lexicographical index on each face
               int k1 = GetFaceQuadIndex(dim,face_id1,nb_rot1,kf,quads1d,ind_f1);
               int k2 = GetFaceQuadIndex(dim,face_id2,nb_rot2,kf,quads1d,ind_f2);
               this->initFaceData(dim,ind_elt1,face_id1,nb_rot1,perm1,ind_elt2,face_id2,nb_rot2,perm2);
               face_tr->Face->SetIntPoint( &ip );
               IntegrationPoint eip1;
               face_tr->Loc1.Transform(ip,eip1);
               eip1.weight = ip.weight;//Sets the weight since Transform doesn't do it...
               // face_tr->Elem1->SetIntPoint( &eip1 );
               IntegrationPoint eip2;
               face_tr->Loc2.Transform(ip,eip2);
               eip2.weight = ip.weight;//Sets the weight since Transform doesn't do it...
               // face_tr->Elem2->SetIntPoint( &eip2 );
               int kg1 = GetGlobalQuadIndex(dim,face_id1,quads1d,ind_f1);
               int kg2 = GetGlobalQuadIndex(dim,face_id2,quads1d,ind_f2);
               Tensor<2> J_e1(&Jac(0,0,kg1,ind_elt1),dim,dim);
               Tensor<2> J_e2(&Jac(0,0,kg2,ind_elt2),dim,dim);
               Tensor<2> Adj(dim,dim);
               adjugate(J_e1,Adj);
               calcOrtho( Adj, face_id1, normal); // normal*determinant (risky, bug prone)
               this->evalEq(dim,k1,k2,n,ind_elt1,face_id1,ind_elt2,face_id2,face_tr,eip1,eip2,J_e1,J_e2,args);
               // FaceInfo face_info = {dim,k1,k2,n,ind_elt1,face_id1,ind_elt2,face_id2,face_tr,eip1,eip2,J_e1,J_e2};
               // this->evalEq(face_info,args);
            }else{//Boundary face
               this->initBoundaryFaceData(ind_elt1,face_id1);
               // TODO: Something should be done here when there is boundary conditions!
               // D11(ind) = 0;  
            }
         }
      }
   }

   // Perform the action of the BilinearFormIntegrator
   virtual void AddMult(const Vector &fun, Vector &vect)
   {
      int dim = this->fes->GetFE(0)->GetDim();
      switch(dim)
      {
         case 1:
            mfem_error("Not yet implemented");
            break;
         case 2:
            this->EvalInt2D(fun, vect);
            this->EvalExt2D(fun, vect);
            break;
         case 3:
            this->EvalInt3D(fun, vect);
            this->EvalExt3D(fun, vect);
            break;
         default:
            mfem_error("Face Kernel does not exist for this dimension.");
            break;
      }
   }

};

}

#endif //MFEM_PAK