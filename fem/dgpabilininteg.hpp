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


// This file contains a prototype version for Discontinuous Galerkin Partial assembly

#ifndef MFEM_DGPABILININTEG
#define MFEM_DGPABILININTEG

#include "../config/config.hpp"
#include "bilininteg.hpp"
#include "partialassemblykernel.hpp"
#include "dgfacefunctions.hpp"

#include "fem.hpp"
#include <cmath>
#include <algorithm>
#include "../linalg/vector.hpp"

namespace mfem
{

/**
* PAK (Partial Assembly Kernel) dependent partial assembly for Convection Integrator.
* Assumes:
*  - InitPb, GetD, MultGtDB for Device
*  - SetSize, SetVal for the D tensor inside Device
*/
template <typename PAK>
class PAConvectionIntegrator : public BilinearFormIntegrator
{
protected:
  PAK& pak;

public:
  PAConvectionIntegrator(PAK& _pak, FiniteElementSpace *fes, const int ir_order,
                        VectorCoefficient &q, double a = 1.0)
  : BilinearFormIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), ir_order)),
    pak(_pak)
  {
  	const int nb_elts = fes->GetNE();
	const int quads  = IntRule->GetNPoints();

   const FiniteElement* fe = fes->GetFE(0);
   int dim = fe->GetDim();
	Vector qvec(dim);
    //device initialization prototype (should evolve)
    //int tensorDim = 4;
    //pak.InitPb(fes,tensorDim,ir_order);
    //gets the D tensor, whatever his type
    typename PAK::Tensor& D = pak.GetD();
    int sizes[] = {dim,dim,quads,nb_elts};
    D.SetSize(sizes);
    //the local D tensor
    DenseMatrix locD(dim, dim);
    for (int e = 0; e < nb_elts; ++e)
    {
      ElementTransformation *Tr = fes->GetElementTransformation(e);
      for (int k = 0; k < quads; ++k)
      {
        const IntegrationPoint &ip = IntRule->IntPoint(k);
  	    Tr->SetIntPoint(&ip);
        DenseMatrix locD = Tr->AdjugateJacobian();
        locD *= ip.weight;
        q.Eval(qvec, *Tr, ip);
        for (int j = 0; j < dim; ++j)
        {
          for (int i = 0; i < dim; ++i)
          {
            double val = - a * qvec(j) * locD(i,j);
            int ind[] = {i,j,k,e};
            D.SetVal(ind,val);
          }
        }
      }
    }
  }

  virtual void AssembleVector(const FiniteElementSpace &fes, const Vector &fun, Vector &vect)
  {
    //We assume that the device has such a method.
    pak.MultGtDB(fun,vect);
  }
};

/** Class for computing the action of 
    (alpha < rho_q (q.n) {u},[v] > + beta < rho_q |q.n| [u],[v] >),
    where u and v are the trial and test variables, respectively, and rho_q are
    given scalar/vector coefficients. The vector coefficient, q, is assumed to
    be continuous across the faces and when given the scalar coefficient, rho,
    is assumed to be discontinuous. The integrator uses the upwind value of rho,
    rho_q, which is value from the side into which the vector coefficient, q,
    points. This uses a partially assembled operator at quadrature points.
* */
template <typename PAK>
class PADGConvectionFaceIntegrator : public BilinearFormIntegrator
{
protected:
   PAK& pak;
	//TODO check what's needed
   FiniteElementSpace *fes;
   const FiniteElement *fe;
   const int dim;

public:
	//using Tensor = DummyTensor;
	typedef DummyTensor Tensor;

  	PADGConvectionFaceIntegrator(PAK& _pak, FiniteElementSpace *_fes, const int ir_order,
                        VectorCoefficient &q, double a = 1.0, double b = 1.0)
  	:BilinearFormIntegrator(&IntRules.Get(_fes->GetFE(0)->GetGeomType(), ir_order)),
  		pak(_pak),
   	fes(_fes),
   	fe(fes->GetFE(0)),
   	dim(fe->GetDim())
	{
	   Mesh* mesh = fes->GetMesh();
	   const int nb_faces = mesh->GetNumFaces();
		const int quads  = pow(ir_order,dim-1);
		Vector qvec(dim);
	   // D11 and D22 are the matrices for the flux of the element on himself for elemt 1 and 2
	   // respectively (we can add together the matrices for the different faces,
	   // (element approach > face approach?)
	   // D12 and D21 are the flux matrices for Element 1 on Element 2, and Element 2 on Element 1
	   // respectively.
	   Tensor& D11 = pak.GetD11();
	   Tensor& D12 = pak.GetD12();
	   Tensor& D21 = pak.GetD21();
	   Tensor& D22 = pak.GetD22();
	   int sizes[] = {quads,nb_faces};
	   D11.SetSize(sizes);
	   D12.SetSize(sizes);
	   D21.SetSize(sizes);
	   D22.SetSize(sizes);
	   DenseMatrix P(dim,dim);
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
	      GetIdRotInfo(ind_elt1,face_id1,nb_rot1);
	      GetIdRotInfo(ind_elt2,face_id2,nb_rot2);
	      int nb_rot = nb_rot1 - nb_rot2;//TODO check that it's correct!!!
	      DenseMatrix base_E1, base_E2;
	      vector<pair<int,int> > map;
	      switch(dim){
	      	case 1:mfem_error("1D Advection not yet implemented");break;
	      	case 2:
	      		GetLocalCoordMap2D(map,nb_rot);
	      		InitFaceCoord2D(face_id1,base_E1);
	      		InitFaceCoord2D(face_id2,base_E2);
	      		break;
	      	case 3:
	      		GetLocalCoordMap3D(map,nb_rot);
	      		InitFaceCoord3D(face_id1,base_E1);
	      		InitFaceCoord3D(face_id2,base_E2);
	      		break;
	      	default:
	      		mfem_error("Wrong dimension");break;
	      }
	      GetChangeOfBasis(base_E1,base_E2,map,P);
	      pak.InitPb(P);
	      for (int k = 0; k < quads; ++k)
	      {
        		const IntegrationPoint &ip = pak.IntPoint(k);//2D point?
        		FaceElementTransformations* face_tr = mesh->GetFaceElementTransformations(face);
	         face_tr->Face->SetIntPoint(&ip);
      		IntegrationPoint eip1;
      		face_tr->Loc1.Transform(ip, eip1);
				//face_tr->Elem1->SetIntPoint(&eip1);
	         q.Eval(qvec, *(face_tr->Elem1), eip1);
	         Vector n;
	         CalcOrtho(face_tr->Face->Jacobian(), n);//includes the determinant?
	         double res = qvec*n;
	         int ind[] = {face,k};
	         double val = ip.weight * ( a/2 * res + b * abs(res) );
	         D11.SetVal(ind,val);
	         D22.SetVal(ind,val);
	         val = ip.weight * ( a/2 * res + b * abs(res) );
	         D21.SetVal(ind,val);
	         val = ip.weight * ( a/2 * res - b * abs(res) );
	         D12.SetVal(ind,val);
	      }
	   }
	}
   
   // Perform the action of the BilinearFormIntegrator
   virtual void AssembleVector(const FiniteElementSpace &fes, const Vector &fun, Vector &vect)
   {
  		pak.MultBtDB(fun,vect);
   }
};

}

#endif //MFEM_DGPABILININTEG