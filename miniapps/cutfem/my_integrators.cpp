// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "my_integrators.hpp"
using namespace std;


namespace mfem
{

void MySimpleDiffusionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   real_t w;

   dshape.SetSize(nd, dim);
   dshapedxt.SetSize(nd, spaceDim);


   elmat.SetSize(nd);

   int order = 4; //Todo: set this differently 
   const IntegrationRule *ir = nullptr;

   ir = &IntRules.Get(el.GetGeomType(),order);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = Trans.Weight();
      w = ip.weight / (w);
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);

      AddMult_a_AAt(w, dshapedxt, elmat);
      
   }
}


void MyDiffusionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   real_t w;

   dshape.SetSize(nd, dim);
   dshapedxt.SetSize(nd, spaceDim);


   elmat.SetSize(nd);

   int order = 2*el.GetOrder();  
   const IntegrationRule *ir = nullptr;
   ir = &IntRules.Get(el.GetGeomType(),order);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = Trans.Weight();
      w = ip.weight / (w);
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);

      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }
      AddMult_a_AAt(w, dshapedxt, elmat);
      
   }
}


void MyVectorDiffusionIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   dim = el.GetDim();
   sdim = Trans.GetSpaceDim();

   // If vdim is not set, set it to the space dimension;
   vdim = (vdim <= 0) ? sdim : vdim;
   const bool square = (dim == sdim);

   if (VQ)
   {
      vcoeff.SetSize(vdim);
   }
   else if (MQ)
   {
      mcoeff.SetSize(vdim);
   }

   dshape.SetSize(dof, dim);
   dshapedxt.SetSize(dof, sdim);

   elmat.SetSize(vdim * dof);
   pelmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &DiffusionIntegrator::GetRule(el,el);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {

      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      real_t w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);

      if (MQ)
      {
         MQ->Eval(mcoeff, Trans, ip);
         for (int ii = 0; ii < vdim; ++ii)
         {
            for (int jj = 0; jj < vdim; ++jj)
            {
               Mult_a_AAt(w*mcoeff(ii,jj), dshapedxt, pelmat);
               elmat.AddMatrix(pelmat, dof*ii, dof*jj);
            }
         }
      }
      else if (VQ)
      {
         VQ->Eval(vcoeff, Trans, ip);
         for (int k = 0; k < vdim; ++k)
         {
            Mult_a_AAt(w*vcoeff(k), dshapedxt, pelmat);
            elmat.AddMatrix(pelmat, dof*k, dof*k);
         }
      }
      else
      {
         if (Q) { w *= Q->Eval(Trans, ip); }
         Mult_a_AAt(w, dshapedxt, pelmat);
         for (int k = 0; k < vdim; ++k)
         {
            elmat.AddMatrix(pelmat, dof*k, dof*k);
         }
      }
   }
}


void GhostPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe1,
                                                const FiniteElement &fe2,
                                                FaceElementTransformations &Tr,
                                                DenseMatrix &elmat)
{
    const int ndim=Tr.GetSpaceDim();
    int elem2 = Tr.Elem2No;
    if(elem2<0){
        elmat.SetSize(fe1.GetDof()*ndim);
        elmat=0.0;
        return;
    }

    const int ndof1 = fe1.GetDof();
    const int ndof2 = fe2.GetDof();
    const int ndofs = ndof1+ndof2;

    elmat.SetSize(ndofs*ndim);
    elmat=0.0;

    int order=std::max(fe1.GetOrder(), fe2.GetOrder());

    int ndofg;
    if(ndim==1){ndofg=order+1;}
    else if(ndim==2){ ndofg=(order+1)*(order+2)/2;}
    else if(ndim==3){ ndofg=(order+1)*(order+2)*(order+3)/6;}

    Vector sh1(ndof1);
    Vector sh2(ndof2);
    Vector shg(ndofg);

    Vector xx(ndim);

    DenseMatrix Mge(ndofg,ndofs); Mge=0.0;
    DenseMatrix Mgg(ndofg,ndofg); Mgg=0.0;
    DenseMatrix Mee(ndofs,ndofs); Mee=0.0;

    ElementTransformation &Tr1 = Tr.GetElement1Transformation();
    ElementTransformation &Tr2 = Tr.GetElement2Transformation();

    const IntegrationRule* ir;


    //element 1
    double w;
    ir=&IntRules.Get(Tr1.GetGeometryType(), 2*order+2);

    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr1.SetIntPoint(&ip);
        Tr1.Transform(ip,xx);
        fe1.CalcPhysShape(Tr1,sh1);
        Shape(xx,order,shg);

        w = Tr1.Weight();
        w = ip.weight * w;
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        for(int i=0;i<ndof1;i++){
            for(int j=0;j<i;j++){
                Mee(i,j)=Mee(i,j)+sh1(i)*sh1(j)*w;
                Mee(j,i)=Mee(j,i)+sh1(i)*sh1(j)*w;
            }
            Mee(i,i)=Mee(i,i)+sh1(i)*sh1(i)*w;
        }

        for(int i=0;i<ndof1;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,i)=Mge(j,i)+shg(j)*sh1(i)*w;
            }}
    }


    //element 2
    ir=&IntRules.Get(Tr2.GetGeometryType(), 2*order+2);
    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr2.SetIntPoint(&ip);
        Tr2.Transform(ip,xx);

        fe2.CalcPhysShape(Tr2,sh2);
        Shape(xx,order,shg);

        w = Tr2.Weight();
        w = ip.weight * w;

        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        for(int i=0;i<ndof2;i++){
            for(int j=0;j<i;j++){
                Mee(ndof1+i,ndof1+j)=Mee(ndof1+i,ndof1+j)+sh2(i)*sh2(j)*w;
                Mee(ndof1+j,ndof1+i)=Mee(ndof1+j,ndof1+i)+sh2(i)*sh2(j)*w;
            }
            Mee(ndof1+i,ndof1+i)=Mee(ndof1+i,ndof1+i)+sh2(i)*sh2(i)*w;
        }

        for(int i=0;i<ndof2;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,ndof1+i)=Mge(j,ndof1+i)+shg(j)*sh2(i)*w;
            }}
    }

    DenseMatrixInverse Mii(Mgg);
    DenseMatrix Mre(ndofg,ndofs);
    DenseMatrix Mff(ndofs,ndofs);
    Mii.Mult(Mge,Mre);
    MultAtB(Mge,Mre,Mff);

    double tv;
    for(int i=0;i<ndofs;i++){
        for(int j=0;j<ndofs;j++){
            tv=penal*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
            for(int d=0;d<ndim;d++){
                elmat(i+d*ndofs,j+d*ndofs)=tv;
            }
        }
    }
}






} // end mfem namespace
