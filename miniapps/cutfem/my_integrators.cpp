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
#include "iostream"
#include "fstream"
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
void VolGhostPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe1,
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

    elmat.SetSize(ndofs);
    elmat=0.0;

    int order=fe1.GetOrder();
    if(order>fe2.GetOrder()){order=fe2.GetOrder();} //order=min(fe1.order, fe2.order)

    int ndofg;
    if(ndim==1){ndofg=order+1;}
    else if(ndim==2){ ndofg=(order+1)*(order+2)/2;}
    else if(ndim==3){ ndofg=(order+1)*(order+2)*(order+3)/6;}

    Vector sh1(ndof1);
    Vector sh2(ndof2);
    Vector shg(ndofg);

    Vector xx(ndim);

    DenseMatrix Mge1(ndofg,ndof1); Mge1=0.0; //mixed mass matrix over element 1
    DenseMatrix Mge2(ndofg,ndof2); Mge2=0.0; //mixed mass matrix over element 2

    DenseMatrix Mgg1(ndofg,ndofg); Mgg1=0.0; //global mass matrix over element 1
    DenseMatrix Mgg2(ndofg,ndofg); Mgg2=0.0; //global mass matrix over element 2

    DenseMatrix Me11(ndof1,ndof1); Me11=0.0; //mass matrix element 1
    DenseMatrix Me22(ndof2,ndof2); Me22=0.0; //mass matrix element 2

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

        //compute the contribution from element 1 to Mgg
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg1(i,j)=Mgg1(i,j)+shg(i)*shg(j)*w;
                Mgg1(j,i)=Mgg1(j,i)+shg(i)*shg(j)*w;
            }
            Mgg1(i,i)=Mgg1(i,i)+shg(i)*shg(i)*w;
        }

        //compute the contribution from element 1 to Mee
        for(int i=0;i<ndof1;i++){
            for(int j=0;j<i;j++){
                Me11(i,j)=Me11(i,j)+sh1(i)*sh1(j)*w;
                Me11(j,i)=Me11(j,i)+sh1(i)*sh1(j)*w;
            }
            Me11(i,i)=Me11(i,i)+sh1(i)*sh1(i)*w;
        }

        //compute the contribution from element 1 to Mge
        for(int i=0;i<ndof1;i++){
            for(int j=0;j<ndofg;j++){
                Mge1(j,i)=Mge1(j,i)+shg(j)*sh1(i)*w;
            }
        }
    }


    //element 2
    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr2.SetIntPoint(&ip);
        Tr2.Transform(ip,xx);
        fe1.CalcPhysShape(Tr2,sh2);
        Shape(xx,order,shg);

        w = Tr2.Weight();
        w = ip.weight * w;

        //compute the contribution from element 2 to Mgg
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg2(i,j)=Mgg2(i,j)+shg(i)*shg(j)*w;
                Mgg2(j,i)=Mgg2(j,i)+shg(i)*shg(j)*w;
            }
            Mgg2(i,i)=Mgg2(i,i)+shg(i)*shg(i)*w;
        }

        //compute the contribution from element 1 to Mee
        for(int i=0;i<ndof1;i++){
            for(int j=0;j<i;j++){
                Me22(i,j)=Me22(i,j)+sh2(i)*sh2(j)*w;
                Me22(j,i)=Me22(j,i)+sh2(i)*sh2(j)*w;
            }
            Me22(i,i)=Me22(i,i)+sh2(i)*sh2(i)*w;
        }

        //compute the contribution from element 1 to Mge
        for(int i=0;i<ndof1;i++){
            for(int j=0;j<ndofg;j++){
                Mge2(j,i)=Mge2(j,i)+shg(j)*sh2(i)*w;
            }
        }
    }

    DenseMatrixInverse Mii1(Mgg1,true);
    DenseMatrixInverse Mii2(Mgg2,true);


    DenseMatrix Mff(ndofg,ndofg);
    DenseMatrix Mf1(ndofg,ndof1), Mf1t(ndofg,ndof1);
    DenseMatrix Mf2(ndofg,ndof2), Mf2t(ndofg,ndof2);

    DenseMatrix M11(ndof1,ndof1);
    DenseMatrix M12(ndof1,ndof2);
    DenseMatrix M22(ndof2,ndof2);


    Mii1.Mult(Mge1,Mf1);
    MultAtB(Mgg2,Mf1,Mf1t);
    MultAtB(Mf1t,Mf1,M11);

    MultAtB(Mf1,Mge2,M12);

    //add the matricess to elmat
    for(int ii=0;ii<ndof1;ii++){
        for(int jj=0;jj<ndof1;jj++){
            elmat(ii,jj)=elmat(ii,jj)+penal*(M11(ii,jj)+M11(jj,ii))/2.0;
        }
    }
    //add the off-diagonal terms
    for(int ii=0;ii<ndof1;ii++){
        for(int jj=0;jj<ndof2;jj++){
            elmat(ii,ndof1+jj)=elmat(ii,ndof1+jj)-penal*M12(ii,jj);
            elmat(ndof1+jj,ii)=elmat(ndof1+jj,ii)-penal*M12(ii,jj);
        }
    }
    //add the 22 matrix
    for(int ii=0;ii<ndof2;ii++){
        for(int jj=0;jj<ndof2;jj++){
            elmat(ndof1+ii,ndof1+jj)=elmat(ndof1+ii,ndof1+jj)+penal*(Me22(ii,jj)+Me22(jj,ii))/2.0;
        }
    }

    Mii2.Mult(Mge2,Mf2);
    MultAtB(Mgg1,Mf2,Mf2t);
    MultAtB(Mf2t,Mf2,M22);

    MultAtB(Mge1,Mf2,M12);

    //add the matricess to elmat
    for(int ii=0;ii<ndof1;ii++){
        for(int jj=0;jj<ndof1;jj++){
            elmat(ii,jj)=elmat(ii,jj)+penal*(Me11(ii,jj)+Me11(jj,ii))/2.0;
        }
    }
    //add the off-diagonal terms
    for(int ii=0;ii<ndof1;ii++){
        for(int jj=0;jj<ndof2;jj++){
            elmat(ii,ndof1+jj)=elmat(ii,ndof1+jj)-penal*M12(ii,jj);
            elmat(ndof1+jj,ii)=elmat(ndof1+jj,ii)-penal*M12(ii,jj);
        }
    }
    //add the 22 matrix
    for(int ii=0;ii<ndof2;ii++){
        for(int jj=0;jj<ndof2;jj++){
            elmat(ndof1+ii,ndof1+jj)=elmat(ndof1+ii,ndof1+jj)+penal*(M22(ii,jj)+M22(jj,ii))/2.0;
        }
    }



    std::cout << std::fixed;
    std::cout << std::setprecision(10);
    //cout.setf(ios::fixed);
    //cout.setf(ios::showpoint);
    //cout.setf(ios::showpos);

    Vector eval;
    elmat.Eigenvalues(eval);
    eval.Print(std::cout,10);

    std::cout<<"ndofg="<<ndofg<<" ndof1="<<ndof1<<" ndof2="<<ndof2<<" ndofs="<<ndofs<<std::endl;

}


#define MFEM_GOOD_GHOST_PENALTY
#ifdef MFEM_GOOD_GHOST_PENALTY
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

    //std::cout<<"E1="<<Tr.Elem1No<<" E2="<<Tr.Elem2No<<std::endl;

    const int ndof1 = fe1.GetDof();
    const int ndof2 = fe2.GetDof();
    const int ndofs = ndof1+ndof2;

    elmat.SetSize(ndofs);
    elmat=0.0;

    int order=fe1.GetOrder();
    if(order>fe2.GetOrder()){order=fe2.GetOrder();} //order=min(fe1.order, fe2.order)

    //order--;

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
    ElementTransformation &Tr1 = Tr.GetElement1Transformation();
    ElementTransformation &Tr2 = Tr.GetElement2Transformation();

    const IntegrationRule* ir;



    Vector xx0(ndim);xx0=0.0;
    double h0=1.0;
    {

        Vector xxm(ndim);
        double s=0;
        ir=&fe1.GetNodes();
        {
            const IntegrationPoint &ip = ir->IntPoint(0);
            Tr1.SetIntPoint(&ip);
            Tr1.Transform(ip,xxm);
        }

        for(int ii=0;ii<ir->GetNPoints();ii++){
            const IntegrationPoint &ip = ir->IntPoint(ii);
            Tr1.SetIntPoint(&ip);
            Tr1.Transform(ip,xx);

            for(int i=0;i<ndim;i++){
                if(xxm[i]<xx[i]){xxm[i]=xx[i];}
            }
            xx0.Add(1.0,xx); s=s+1.0;
        }

        ir=&IntRules.Get(Tr2.GetGeometryType(), order);
        for(int ii=0;ii<ir->GetNPoints();ii++){
            const IntegrationPoint &ip = ir->IntPoint(ii);
            Tr2.SetIntPoint(&ip);
            Tr2.Transform(ip,xx);

            for(int i=0;i<ndim;i++){
                if(xxm[i]<xx[i]){xxm[i]=xx[i];}
            }
            xx0.Add(1.0,xx); s=s+1.0;
        }

        xx0/=s;

        h0=fabs(xxm[0]-xx0[0]);
        for(int i=0;i<ndim;i++){
            if(fabs(xxm[i]-xx0[i])<h0){h0=fabs(xxm[i]-xx0[i]);}
        }

    }

    //std::cout<<"xx0="; xx0.Print(std::cout);

    //element 1
    double w;

    ir=&IntRules.Get(Tr1.GetGeometryType(), 2*order+2);
    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr1.SetIntPoint(&ip);
        Tr1.Transform(ip,xx);
        xx.Add(-1.0,xx0);//shift the coordinates with the reference point
        xx/=h0;

        fe1.CalcPhysShape(Tr1,sh1);
        Shape(xx,order,shg);

        w = Tr1.Weight();
        w = ip.weight * w;

        //compute the contribution from element 1 to Mgg
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        //compute the contribution from element 1 to Mge
        for(int i=0;i<ndof1;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,i)=Mge(j,i)+shg(j)*sh1(i)*w;
            }
        }
    }

    //element 2
    ir=&IntRules.Get(Tr2.GetGeometryType(), 2*order+2);
    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr2.SetIntPoint(&ip);
        Tr2.Transform(ip,xx);
        xx.Add(-1.0,xx0);//shift the coordinates with the reference point
        xx/=h0;

        fe2.CalcPhysShape(Tr2,sh2);
        Shape(xx,order,shg);

        w = Tr2.Weight();
        w = ip.weight * w;

        //compute the contribution from element 2 to Mgg
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        //compute the contribution from element 2 to Mge
        for(int i=0;i<ndof2;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,ndof1+i)=Mge(j,ndof1+i)+shg(j)*sh2(i)*w;
            }
        }

    }

    /*
    DenseMatrixSVD MggSVD(Mgg.Width(),Mgg.Height(),'A','A'); MggSVD.Eval(Mgg);

    if((Tr.Elem1No==104)||(Tr.Elem1No==105)){
        std::cout<<"Vector="<<std::endl;
        MggSVD.Singularvalues().Print(std::cout);
        std::cout<<"Left="<<std::endl;
        MggSVD.LeftSingularvectors().PrintMatlab(std::cout);
        std::cout<<"Right="<<std::endl;
        MggSVD.RightSingularvectors().PrintMatlab(std::cout);
        std::cout<<std::endl;
    }
    */

    DenseMatrixInverse Mii(Mgg,true);
    DenseMatrix Mre(ndofg,ndofs);
    Mii.Mult(Mge,Mre);

    /*
    if((Tr.Elem1No==1627)||(Tr.Elem1No==1626)){
        Mgg.PrintMatlab(std::cout);
    }

    if((Tr.Elem1No==104)||(Tr.Elem1No==105)){
        Mgg.PrintMatlab(std::cout);
    }
    */

    //global shape functions
    Vector gs(ndofs);
    //integrate the global shape functions over element 1
    ir=&IntRules.Get(Tr1.GetGeometryType(), 2*order+2);
    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr1.SetIntPoint(&ip);
        Tr1.Transform(ip,xx);
        xx.Add(-1.0,xx0);//shift the coordinates with the reference point
        xx/=h0;

        fe1.CalcPhysShape(Tr1,sh1);
        Shape(xx,order,shg);

        w = Tr1.Weight();
        w = ip.weight * w;

        Mre.MultTranspose(shg,gs);
        for(int i=0;i<ndof1;i++){
            gs(i)=gs(i)-sh1(i);
        }

        for(int i=0;i<ndofs;i++){
            for(int j=0;j<i;j++){
                elmat(i,j)=elmat(i,j)+w*gs(i)*gs(j);
                elmat(j,i)=elmat(j,i)+w*gs(i)*gs(j);
            }
            elmat(i,i)=elmat(i,i)+w*gs(i)*gs(i);
        }
    }

    //integrate the global shape functions over element 2
    ir=&IntRules.Get(Tr2.GetGeometryType(), 2*order+2);
    for(int ii=0;ii<ir->GetNPoints();ii++){
        const IntegrationPoint &ip = ir->IntPoint(ii);
        Tr2.SetIntPoint(&ip);
        Tr2.Transform(ip,xx);
        xx.Add(-1.0,xx0);//shift the coordinates with the reference point
        xx/=h0;

        fe2.CalcPhysShape(Tr2,sh2);
        Shape(xx,order,shg);

        w = Tr2.Weight();
        w = ip.weight * w;

        Mre.MultTranspose(shg,gs);
        for(int i=0;i<ndof2;i++){
            gs(ndof1+i)=gs(ndof1+i)-sh2(i);
        }

        for(int i=0;i<ndofs;i++){
            for(int j=0;j<i;j++){
                elmat(i,j)=elmat(i,j)+w*gs(i)*gs(j);
                elmat(j,i)=elmat(j,i)+w*gs(i)*gs(j);
            }
            elmat(i,i)=elmat(i,i)+w*gs(i)*gs(i);
        }
    }

    elmat*=penal;

    //std::cout << std::fixed;
    //std::cout << std::setprecision(10);
    //cout.setf(ios::fixed);
    //cout.setf(ios::showpoint);
    //cout.setf(ios::showpos);


    /*
    Vector eval;
    elmat.Eigenvalues(eval);
    eval.Print(std::cout,10);

    std::cout<<"ndofg="<<ndofg<<" ndof1="<<ndof1<<" ndof2="<<ndof2<<" ndofs="<<ndofs<<std::endl;
    */
}

#else
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

    elmat.SetSize(ndofs);
    elmat=0.0;

    int order=fe1.GetOrder();
    if(order>fe2.GetOrder()){order=fe2.GetOrder();} //order=min(fe1.order, fe2.order)

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

        //compute the contribution from element 1 to Mgg
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        //compute the contribution from element 1 to Mee
        for(int i=0;i<ndof1;i++){
            for(int j=0;j<i;j++){
                Mee(i,j)=Mee(i,j)+sh1(i)*sh1(j)*w;
                Mee(j,i)=Mee(j,i)+sh1(i)*sh1(j)*w;
            }
            Mee(i,i)=Mee(i,i)+sh1(i)*sh1(i)*w;
        }

        //compute the contribution from element 1 to Mge
        for(int i=0;i<ndof1;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,i)=Mge(j,i)+shg(j)*sh1(i)*w;
            }
        }
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

        //compute the contribution from element 2 to Mgg
        for(int i=0;i<ndofg;i++){
            for(int j=0;j<i;j++){
                Mgg(i,j)=Mgg(i,j)+shg(i)*shg(j)*w;
                Mgg(j,i)=Mgg(j,i)+shg(i)*shg(j)*w;
            }
            Mgg(i,i)=Mgg(i,i)+shg(i)*shg(i)*w;
        }

        //compute the contribution from element 2 to Mee
        for(int i=0;i<ndof2;i++){
            for(int j=0;j<i;j++){
                Mee(ndof1+i,ndof1+j)=Mee(ndof1+i,ndof1+j)+sh2(i)*sh2(j)*w;
                Mee(ndof1+j,ndof1+i)=Mee(ndof1+j,ndof1+i)+sh2(i)*sh2(j)*w;
            }
            Mee(ndof1+i,ndof1+i)=Mee(ndof1+i,ndof1+i)+sh2(i)*sh2(i)*w;
        }

        //compute the contribution from element 2 to Mge
        for(int i=0;i<ndof2;i++){
            for(int j=0;j<ndofg;j++){
                Mge(j,ndof1+i)=Mge(j,ndof1+i)+shg(j)*sh2(i)*w;
            }
        }
    }

    DenseMatrixInverse Mii(Mgg,true);
    DenseMatrix Mre(ndofg,ndofs);
    DenseMatrix Mff(ndofs,ndofs);
    Mii.Mult(Mge,Mre);
    MultAtB(Mge,Mre,Mff);

    double tv;
    for(int i=0;i<ndofs;i++){
        for(int j=0;j<ndofs;j++){
            tv=penal*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
            elmat(i,j)=tv;
        }
    }

    std::cout << std::fixed;
    std::cout << std::setprecision(10);
    //cout.setf(ios::fixed);
    //cout.setf(ios::showpoint);
    //cout.setf(ios::showpos);

    Vector eval;
    elmat.Eigenvalues(eval);
    eval.Print(std::cout,10);
    Mee.Eigenvalues(eval);
    eval.Print(std::cout,10);

    std::cout<<"ndofg="<<ndofg<<" ndof1="<<ndof1<<" ndof2="<<ndof2<<" ndofs="<<ndofs<<std::endl;
}
#endif




void UnfittedBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;


   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      real_t val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val*sweights(i), shape, elvect);
      
   }
}


void GhostPenaltyVectorIntegrator::AssembleFaceMatrix(const FiniteElement &fe1,
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

    DenseMatrix dummymat; 
    dummymat = 0.0;

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
    for(int i=0;i<ndof1;i++){
        for(int j=0;j<ndof1;j++){
            tv=penal*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
            for(int d=0;d<ndim;d++){
                elmat(i+d*ndof1,j+d*ndof1)=tv;
            }
        }
    }
    for(int i=ndof1;i<ndofs;i++){
        for(int j=0;j<ndof1;j++){
            tv=penal*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
            for(int d=0;d<ndim;d++){
                elmat(ndof1+i+d*ndof1,j+d*ndof1)=tv;
            }
        }
    }

    for(int i=0;i<ndof1;i++){
        for(int j=ndof1;j<ndofs;j++){
            tv=penal*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
            for(int d=0;d<ndim;d++){
                elmat(i+d*ndof1,ndof1 + j+d*ndof1)=tv;
            }
        }
    }

    for(int i=ndof1;i<ndofs;i++){
        for(int j=ndof1;j<ndofs;j++){
            tv=penal*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
            for(int d=0;d<ndim;d++){
                elmat(ndof1 + i+d*ndof1,ndof1 + j+d*ndof1)=tv;
            }
        }
    }

    // for(int i=0;i<ndofs;i++){
    //     for(int j=0;j<ndofs;j++){
    //         tv=penal*(Mee(i,j)+Mee(j,i)-Mff(i,j)-Mff(j,i))/(2.0);
    //         for(int d=0;d<ndim;d++){
    //             dummy(i+d*ndofs,j+d*ndofs)=tv;
    //         }
    //     }
    // }

    // elmat.Print();


}



void UnfittedVectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
        
      Q.Eval(vec, Tr, ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++){
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s)*sweights(i);
         }
      }
   }
}


} // end mfem namespace
