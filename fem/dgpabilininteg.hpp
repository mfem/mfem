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
*  - MultGtDB Operation and SetSizeD, SetValD for the D tensor inside the Kernel
*/
template <template<PAOp> class PAK = DummyDomainPAK >
class PAConvectionIntegrator : public BilinearFormIntegrator
{
protected:
  	PADomainIntegrator<PAOp::BtDG, PAK> pak;

public:
  	PAConvectionIntegrator(FiniteElementSpace *fes, const int order,
                        VectorCoefficient &q, double a = 1.0)
  	: BilinearFormIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     pak(fes,order)
  	{
	  	const int nb_elts = fes->GetNE();
		const int quads  = IntRule->GetNPoints();
	   const FiniteElement* fe = fes->GetFE(0);
	   int dim = fe->GetDim();
		Vector qvec(dim);
		// Initialization of the size of the D tensor
		//ToSelf: should initialization change since we know the number of args
		int sizes[] = {dim,quads,nb_elts};
		pak.SetSizeD(sizes);
		for (int e = 0; e < nb_elts; ++e)
		{
		   ElementTransformation *Tr = fes->GetElementTransformation(e);
		   for (int k = 0; k < quads; ++k)
		   {
		     	const IntegrationPoint &ip = IntRule->IntPoint(k);
			   Tr->SetIntPoint(&ip);
		     	const DenseMatrix& locD = Tr->AdjugateJacobian();
		     	q.Eval(qvec, *Tr, ip);
		     	for (int i = 0; i < dim; ++i)
		     	{
		     		double val = 0;
		     		for (int j = 0; j < dim; ++j)
		       	{
		         	val += locD(i,j) * qvec(j);
		       	}
					//ToSelf: should SetValD change since we know the number of args
	         	int ind[] = {i,k,e};
	         	pak.SetValD(ind, ip.weight * a * val);
		     	}
		   }
		}
  	}

  virtual void AssembleVector(const FiniteElementSpace &fes, const Vector &fun, Vector &vect)
  {
    //We assume that the kernel has such a method.
    pak.Mult(fun,vect);
  }

};

/**
*	A Mass Integrator using Partial Assembly based on Eigen Tensors.
*  TODO: should be merged with legacy one
*/
template <int Dim, template<int,PAOp> class PAK = EigenDomainPAK>
class EigenPAMassIntegrator : public BilinearFormIntegrator
{
protected:
	// BtDB specifies which operation in the kernel we're using
  	PAK<Dim,PAOp::BtDB> pak;

public:
	EigenPAMassIntegrator(FiniteElementSpace* fes, const int order)
	: BilinearFormIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     pak(fes,order)
	{
		
	   const int nelem   = fes->GetNE();
	   const int quads   = IntRule->GetNPoints();
	   int sizes[2] = {quads,nelem};
	   pak.SetSizeD(sizes);

	   for (int e = 0; e < fes->GetNE(); e++)
	   {
	      ElementTransformation *Tr = fes->GetElementTransformation(e);
	      for (int k = 0; k < quads; k++)
	      {
	         const IntegrationPoint &ip = IntRule->IntPoint(k);
	         Tr->SetIntPoint(&ip);
	         double val = ip.weight * Tr->Weight();
	         int ind[2] = {k,e};
	         pak.SetValD(ind,val);
	      }
	   }
	}

	EigenPAMassIntegrator(FiniteElementSpace* fes, const int order, Coefficient& coeff)
	: BilinearFormIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     pak(fes,order)
	{
		
	   const int nelem   = fes->GetNE();
	   const int quads   = IntRule->GetNPoints();
	   int sizes[2] = {quads,nelem};
	   pak.SetSizeD(sizes);

	   for (int e = 0; e < fes->GetNE(); e++)
	   {
	      ElementTransformation *Tr = fes->GetElementTransformation(e);
	      for (int k = 0; k < quads; k++)
	      {
	         const IntegrationPoint &ip = IntRule->IntPoint(k);
	         Tr->SetIntPoint(&ip);
	         const double weight = ip.weight * Tr->Weight();
	         double val = coeff.Eval(*Tr, ip) * weight;
	         int ind[2] = {k,e};
	         pak.SetValD(ind,val);
	      }
	   }
	}

	virtual void AssembleVector(const FiniteElementSpace &fes, const Vector &fun, Vector &vect)
	{
		pak.Mult(fun,vect);
	}
};

/**
*	A Convection Integrator using Partial Assembly based on Eigen Tensors.
*  TODO: should be merged with legacy one
*/
template <int Dim, template<int,PAOp> class PAK = EigenDomainPAK>
class EigenPAConvectionIntegrator : public BilinearFormIntegrator
{
protected:
	// BtDG specifies which operation in the kernel we're using
  	PAK<Dim,PAOp::BtDG> pak;

public:
  	EigenPAConvectionIntegrator(FiniteElementSpace *fes, const int order,
                        VectorCoefficient &q, double a = 1.0)
  	: BilinearFormIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
     pak(fes,order)
  	{
	  	const int nb_elts = fes->GetNE();
		const int quads  = IntRule->GetNPoints();
	   const FiniteElement* fe = fes->GetFE(0);
	   int dim = fe->GetDim();
		Vector qvec(dim);
		// Initialization of the size of the D tensor
		//ToSelf: should initialization change since we know the number of args
		int sizes[] = {dim,quads,nb_elts};
		pak.SetSizeD(sizes);
		for (int e = 0; e < nb_elts; ++e)
		{
		   ElementTransformation *Tr = fes->GetElementTransformation(e);
		   for (int k = 0; k < quads; ++k)
		   {
		     	const IntegrationPoint &ip = IntRule->IntPoint(k);
			   Tr->SetIntPoint(&ip);
		     	const DenseMatrix& locD = Tr->AdjugateJacobian();
		     	q.Eval(qvec, *Tr, ip);
		     	for (int i = 0; i < dim; ++i)
		     	{
		     		double val = 0;
		     		for (int j = 0; j < dim; ++j)
		       	{
		         	val += locD(i,j) * qvec(j);
		       	}
					//ToSelf: should SetValD change since we know the number of args
	         	int ind[] = {i,k,e};
	         	pak.SetValD(ind, ip.weight * a * val);
		     	}
		   }
		}
  	}

  virtual void AssembleVector(const FiniteElementSpace &fes, const Vector &fun, Vector &vect)
  {
    //We assume that the kernel has such a method. Operation is set in the type of PAK.
    pak.Mult(fun,vect);
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
    Assumes:
      - IntegrationRule for PAK
* */
template <typename PAK>
class PADGConvectionFaceIntegrator : public BilinearFormIntegrator
{
protected:
   PAK pak;

public:

  	PADGConvectionFaceIntegrator(FiniteElementSpace *fes, const int order,
                        VectorCoefficient &q, double a = 1.0, double b = 1.0)
  	:BilinearFormIntegrator(&IntRules.Get(fes->GetFE(0)->GetGeomType(), order)),
  	pak(fes,order,2)
	{
		const int dim = fes->GetFE(0)->GetDim();
	   Mesh* mesh = fes->GetMesh();
	   const int nb_faces = mesh->GetNumFaces();
	   int geom;
	   switch(dim){
	   	case 1:geom = Geometry::POINT;break;
	   	case 2:geom = Geometry::SEGMENT;break;
	   	case 3:geom = Geometry::SQUARE;break;
	   }
		const int quads  = IntRules.Get(geom, order).GetNPoints();//pow(ir_order,dim-1);
		Vector qvec(dim);
	   // D11 and D22 are the matrices for the flux of the element on himself for elemt 1 and 2
	   // respectively
	   // (ToSelf: element approach > face approach?)
	   // D12 and D21 are the flux matrices for Element 1 on Element 2, and Element 2 on Element 1
	   // respectively.
	   typename PAK::Tensor& D11 = pak.GetD11();
	   typename PAK::Tensor& D12 = pak.GetD12();
	   typename PAK::Tensor& D21 = pak.GetD21();
	   typename PAK::Tensor& D22 = pak.GetD22();
	   int sizes[] = {quads,nb_faces};
	   D11.SetSize(sizes);
	   D12.SetSize(sizes);
	   D21.SetSize(sizes);
	   D22.SetSize(sizes);
	   IntMatrix P(dim,dim);
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
	      // Trial basis shouldn't rotate, since we apply B1d to dofs directly
	      //int nb_rot = nb_rot1 - nb_rot2;//TODO check that it's correct!!!
	      //IntMatrix base_E1(dim,dim), base_E2(dim,dim);
	      // The mapping "map" stores the cahnge of basis from element e1 to element e2
	      //vector<pair<int,int> > map;
	      // //TODO: This code should be factorized and put somewhere else
	      switch(dim){
	      	case 1:mfem_error("1D Advection not yet implemented");break;
	      	case 2:
	      		GetChangeOfBasis2D(face_id1,face_id2,P);
	      		/*GetLocalCoordMap2D(map,nb_rot);
	      		InitFaceCoord2D(face_id1,base_E1);
	      		InitFaceCoord2D(face_id2,base_E2);*/
	      		break;
	      	case 3:
	      		mfem_error("3D Advection not yet implemented");
	      		/*GetLocalCoordMap3D(map,nb_rot);
	      		InitFaceCoord3D(face_id1,base_E1);
	      		InitFaceCoord3D(face_id2,base_E2);*/
	      		break;
	      	default:
	      		mfem_error("Wrong dimension");break;
	      }
	      //GetChangeOfBasis(base_E1,base_E2,map,P);
	      pak.InitPb(face,P);
   		FaceElementTransformations* face_tr = mesh->GetFaceElementTransformations(face);
	      for (int k = 0; k < quads; ++k)
	      {
	      	// We need to be sure that we go in the same order as for the partial assembly.
	      	// So we take the points on the face for the element that has the trial function.
	      	// So we can only compute 2 of the 4 matrices with a set of points.
	      	// IntPoint are not the same from e1 to e2 than from e2 to e1, because
	      	// dofs are usually not oriented the same way on the face.
	      	// IntPoints are the same for e1 to e2 (D21) and for e1 to e1 (D11).
	         // Shared parameters
	         int ind[] = {k,face};
	         double val = 0;
	         Vector n(dim);
	         double res;
	      	const IntegrationRule& ir = IntRules.Get(geom, order);
        		const IntegrationPoint &ip = ir.IntPoint(k);
	      	// We compute D11 and D21
      		IntegrationPoint eip1;
        		eip1 = pak.IntPoint( face_id1, k );//2D point ordered according to coord on element 1
      		//face_tr->Loc1.Transform( ip, eip1 );
	         face_tr->Face->SetIntPoint( &ip );
				face_tr->Elem1->SetIntPoint( &eip1 );
	         q.Eval( qvec, *(face_tr->Elem1), eip1 );
	         CalcOrtho( face_tr->Face->Jacobian(), n );
	         res = qvec * n;
	         if(face_tr->Elem2No!=-1){
		         val = eip1.weight * (   a/2 * res + b * abs(res) );
		         D11(ind) = val;
		         val = eip1.weight * (   a/2 * res - b * abs(res) );
		         D21(ind) = val;
		         // We compute D12 and D22
	      		IntegrationPoint eip2;
	      		eip2 = pak.IntPoint( face_id2, k );
	      		//face_tr->Loc2.Transform( ip, eip2 );
		        	face_tr->Face->SetIntPoint( &ip );
					face_tr->Elem2->SetIntPoint( &eip2 );
			      q.Eval( qvec, *(face_tr->Elem2), eip2 );
			      CalcOrtho( face_tr->Face->Jacobian(), n );
			      res = qvec * n;
			      val = eip2.weight * ( - a/2 * res + b * abs(res) );
			      D22(ind) = val;
			      val = eip2.weight * ( - a/2 * res - b * abs(res) );
			      D12(ind) = val;
		      }else{//Boundary face
		        	D11(ind) = 0;	      	
		      }
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