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
   PADomainInt(FiniteElementSpace *fes, const int order, const typename Equation::Args& args)
   : LinearFESpaceIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     Op(fes,order,args)
   {
      const int nb_elts = fes->GetNE();
      const int quads  = IntRule->GetNPoints();
      const FiniteElement* fe = fes->GetFE(0);
      const int dim = fe->GetDim();
      this->InitD(dim,quads,nb_elts);
      for (int e = 0; e < nb_elts; ++e)
      {
         ElementTransformation *Tr = fes->GetElementTransformation(e);
         for (int k = 0; k < quads; ++k)
         {
            const IntegrationPoint &ip = IntRule->IntPoint(k);
            Tr->SetIntPoint(&ip);
            this->evalEq(dim, k, e, Tr, ip, args);
         }
      }
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
   PAFaceInt(FiniteElementSpace* fes, const int order, const typename Equation::Args& args)
   : LinearFESpaceIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     Op(fes, order, args)
   {
      const int dim = fes->GetFE(0)->GetDim();
      const IntegrationRule& ir1d = IntRules.Get(Geometry::SEGMENT, order);
      // InitIntegrationPoints(intPts, order, ir1d, dim);
      const int quads1d = ir1d.GetNPoints();
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
      this->InitD(dim,quads,nb_elts,nb_faces_elt);
      this->kernel_data.setSize(nb_elts,nb_faces_elt);
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
         int perm1, perm2;
         // cout << "ind_elt1=" << ind_elt1 << ", face_id1=" << face_id1 << ", nb_rot1=" << nb_rot1 << ", ind_elt2=" << ind_elt2 << ", face_id2=" << face_id2 << ", nb_rot2=" << nb_rot2 << endl;
         for (int kf = 0; kf < quads; ++kf)
         {
            const IntegrationRule& ir = IntRules.Get(geom, order);
            const IntegrationPoint& ip = ir.IntPoint(kf);
            if(ind_elt2!=-1){//Not a boundary face
               int k1 = GetFaceQuadIndex(dim,face_id1,nb_rot1,kf,quads1d);
               int k2 = GetFaceQuadIndex(dim,face_id2,nb_rot2,kf,quads1d);
               Permutation(dim,face_id1,face_id2,nb_rot2,perm1,perm2);
               // Initialization of indirection and permutation identification
               this->kernel_data(ind_elt2,face_id2).indirection = ind_elt1;
               this->kernel_data(ind_elt2,face_id2).permutation = perm1;
               this->kernel_data(ind_elt1,face_id1).indirection = ind_elt2;
               this->kernel_data(ind_elt1,face_id1).permutation = perm2;
               face_tr->Face->SetIntPoint( &ip );
               // IntegrationPoint& eip1 = this->IntPoint( face_id1, k );
               IntegrationPoint eip1;
               face_tr->Loc1.Transform(ip,eip1);
               eip1.weight = ip.weight;//Sets the weight since Transform doesn't do it...
               IntegrationPoint eip2;
               face_tr->Loc2.Transform(ip,eip2);
               eip2.weight = ip.weight;//Sets the weight since Transform doesn't do it...
               face_tr->Elem1->SetIntPoint( &eip1 );
               CalcOrtho( face_tr->Face->Jacobian(), n );
               this->evalEq(dim,k1,k2,n,ind_elt1,face_id1,ind_elt2,face_id2,face_tr,eip1,eip2,args);
            }else{//Boundary face
               this->kernel_data(ind_elt1,face_id1).indirection = ind_elt2;
               this->kernel_data(ind_elt1,face_id1).permutation = 0;
               // FIXME: Something should be done here!
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