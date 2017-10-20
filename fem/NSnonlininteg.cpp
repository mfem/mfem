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

#include "fem.hpp"

namespace mfem
{
void VectorNonLinearNSIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   /*int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   int intorder = 2*el.GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   elvect = 0.0;
   model->SetTransformation(Tr);
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el.CalcDShape(ip, DSh);
      Mult(DSh, J0i, DS);
      MultAtB(PMatI, DS, J);

      model->EvalP(J, P);

      P *= ip.weight*Tr.Weight();
      AddMultABt(DS, P, PMatO);
   }*/
   int dim = el.GetDim() + 1; //1 for pressure 
   int nd  = el.GetDof();
   int sp_dim = Tr.GetSpaceDim();   
   
   dshape.SetSize(nd, sp_dim); 
   gshape.SetSize(nd, sp_dim);   
   Jinv  .SetSize(sp_dim);   
   
   PMatI.UseExternalData(elfun.GetData(), nd, dim);
   elvect.SetSize(nd*dim);
   //PMatO.UseExternalData(elvect.GetData(), nd, dim);
   
   elvect = 0.0;
   
   //Compute element parameters used in Tau's calculations 
   eleVol = Geometry::Volume[el.GetGeomType()] * Tr.Weight();
   eleLength = ((sp_dim == 3) ? (0.60046878 * pow(eleVol,0.333333333333333333333))
                : (1.128379167 * sqrt(eleVol)));      
   
   //Add tauMom * (f, -grad q)
   StabVecGradIntegrator(el, Tr, elvect);  
   
   //tauMom * (f, -u.grad v)
   StabVecAdvIntegrator(el, Tr, elvect);
   
   //tauMom * (f, nu * Lap v);
   if (el.GetOrder() > 1)
       StabVecLapIntegrator(el, Tr, elvect);
    
}

void VectorNonLinearNSIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   /*int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   int intorder = 2*el.GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   elmat = 0.0;
   model->SetTransformation(Tr);
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el.CalcDShape(ip, DSh);
      Mult(DSh, J0i, DS);
      MultAtB(PMatI, DS, J);

      model->AssembleH(J, DS, ip.weight*Tr.Weight(), elmat);
   }*/
   int dim = el.GetDim() + 1; //1 is for pressure 
   int nd  = el.GetDof();
   int sp_dim = Tr.GetSpaceDim();     
   PMatI.UseExternalData(elfun.GetData(), nd, dim); 
   
   elmat. SetSize (dim * nd);
   Jinv.  SetSize (sp_dim);
   dshape.SetSize (nd, sp_dim);
   gshape.SetSize (nd, sp_dim); 
   
   elmat = 0.0;
   
   //Add convective term
   ConvectionIntegrator(el, Tr, elmat);
   
   //Compute element parameters used in Tau's calculations 
   eleVol = Geometry::Volume[el.GetGeomType()] * Tr.Weight();
   eleLength = ((sp_dim == 3) ? (0.60046878 * pow(eleVol,0.333333333333333333333))
                : (1.128379167 * sqrt(eleVol)));   
   
   //Add tauMass * (div v, div u)
   SatbDivIntegrator(el, Tr, elmat);
   
   //Add tauMom * (grad q, grad p)
   StabLaplacianIntegrator(el, Tr, elmat); 
   
   //tauMom * (u.grad v , u.grad u)
   StabConvectionIntegrator(el, Tr, elmat);
   
   //tauMom * [(u.grad v , grad p) + (grad q , u.grad u)]
   StabConvGradIntegrator(el, Tr, elmat);

   if (el.GetOrder() > 1)
   {
    //tauMom * (nu Lap v, nu Lap u)
    StabLapLapIntegrator(el, Tr, elmat);
   
    //tauMom * [(nu Lap v, -grad p) + (-grad q, nu Lap u)]
    StabLapGradIntegrator(el, Tr, elmat);
    
    //tauMom * [(nu Lap v, -u.grad u) + (-u.grad v , nu Lap u)]
    StabLapConvIntegrator(el, Tr, elmat);  
   }
   
   
}

void VectorNonLinearNSIntegrator::StabConvectionIntegrator(
                                  const FiniteElement &el,
                                  ElementTransformation &Tr, 
                                  DenseMatrix &elmat)
{
    double norm;
    
    int dim = el.GetDim() + 1;
    int nd = el.GetDof();
    int sp_dim = Tr.GetSpaceDim();
    DenseMatrix pelemat;
    Vector advGrad(nd), AdvUP(dim), auxU(sp_dim);
    
    pelemat.SetSize(nd);
    
    int intorder = 2 * (el.GetOrder() + Tr.OrderGrad(&el));
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
  
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcShape(ip, shape);
        el.CalcDShape(ip, dshape);
        
        Tr.SetIntPoint(&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse (Tr.Jacobian(), Jinv);
        
        Mult(dshape, Jinv, gshape); 
        
        //compute tau
        double tauMom, tauMass;
        double nu = nuCoef->Eval (Tr, ip);
        PMatI.MultTranspose(shape, AdvUP);
        for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk]; 
      
        double Unorm = auxU.Norml2();
        CalculateTaus(nu, Unorm, tauMom, tauMass);      
        norm *= tauMom;
        
        gshape.Mult(auxU, advGrad);
        MultVVt(advGrad, pelemat);
        pelemat *= norm;

        for (int kk = 0; kk < sp_dim; kk++)
        {
            elmat.AddMatrix(pelemat, kk*nd, kk*nd);
        }        
    }
}

void VectorNonLinearNSIntegrator::ConvectionIntegrator(const FiniteElement &el,
                                    ElementTransformation &Trans,
                                    DenseMatrix& elmat)
{
    int dim = el.GetDim() + 1;
    int nd = el.GetDof();
    int sp_dim = Trans.GetSpaceDim();
    DenseMatrix pelemat;
    double norm; 
    Vector AdvUP(dim), advGrad(nd);
    Vector auxU(sp_dim);
    
    pelemat.SetSize (nd);
    
    int intorder = 2 * el.GetOrder() + Trans.OrderGrad(&el);
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
    
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcShape (ip, shape);
        el.CalcDShape(ip, dshape);
        
        Trans.SetIntPoint(&ip);
        norm = ip.weight * Trans.Weight();
        CalcInverse (Trans.Jacobian(), Jinv);
        
        Mult(dshape, Jinv, gshape);
        
        PMatI.MultTranspose(shape, AdvUP);
        for (int kk = 0; kk < sp_dim; kk++)
            auxU[kk] = AdvUP[kk];
        
        gshape.Mult(auxU, advGrad);
        MultVWt(shape, advGrad, pelemat);
        pelemat *= norm;
        
        for (int kk = 0; kk < sp_dim; kk++)
        {
            elmat.AddMatrix(pelemat, kk*nd, kk*nd);
        }   
        
    }
}

void VectorNonLinearNSIntegrator::StabConvGradIntegrator(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    DenseMatrix &elmat)
{
    int dim = el.GetDim() + 1;
    int nd  = el.GetDof();
    int sp_dim = Tr.GetSpaceDim();
    DenseMatrix pelemat;
    Vector AdvUP(dim), advGrad(nd), auxU(sp_dim), vec1;
    double norm;
    
    pelemat.SetSize(nd);
    
    int intorder = el.GetOrder() + 2 * Tr.OrderGrad(&el);
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
    
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcDShape(ip, dshape);
        el.CalcShape (ip, shape);
        
        Tr.SetIntPoint (&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse(Tr.Jacobian(), Jinv);
        
        Mult (dshape, Jinv, gshape);
        
        //compute tau
        double tauMom, tauMass;
        double nu = nuCoef->Eval (Tr, ip);
        PMatI.MultTranspose(shape, AdvUP);
        for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk]; 
      
        double Unorm = auxU.Norml2();
        CalculateTaus(nu, Unorm, tauMom, tauMass);      
        norm *= tauMom;   
        
        gshape.Mult(auxU, advGrad);

        for (int kk = 0; kk < sp_dim; kk++)
        {
            gshape.GetColumnReference(kk, vec1);
            MultVWt(advGrad, vec1, pelemat);
            pelemat *= norm;
            elmat.AddMatrix(pelemat, nd * kk, sp_dim * nd);
            pelemat.Transpose();
            elmat.AddMatrix(pelemat, sp_dim * nd, nd * kk);
        }         
    }
}

void VectorNonLinearNSIntegrator::StabLapConvIntegrator(
                                  const FiniteElement &el,
                                  ElementTransformation &Tr, 
                                  DenseMatrix &elmat)
{
    int dim = el.GetDim() + 1;
    int nd  = el.GetDof();
    int sp_dim = Tr.GetSpaceDim();
    DenseMatrix pelemat, Hessian, auxC;
    Vector Laplacian(nd);
    Vector AdvUP(dim), advGrad(nd), auxU(sp_dim); 
    int hess_size = (sp_dim * (sp_dim + 1)) / 2; 
    Hessian.SetSize(nd , hess_size);
    double norm;
    
    pelemat.SetSize (nd);
    auxC.SetSize (nd);
    
    int grad_order = Tr.OrderGrad(&el);
    int laporder = ((grad_order > 0) ? grad_order - 1 : 0 );
    int intorder = laporder +  Tr.OrderGrad(&el);
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
    
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcShape (ip, shape);
        el.CalcDShape(ip, dshape);
        //Only applies to 2D and up to order 3!!
        el.CalcHessian(ip, Hessian);

        Tr.SetIntPoint (&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse (Tr.Jacobian(), Jinv);
        Mult (dshape, Jinv, gshape);
        
        CalcPhysLaplacian (Hessian, nd, sp_dim, Laplacian);
        
        //compute tau
        double tauMom, tauMass;
        double nu = nuCoef->Eval (Tr, ip);
        PMatI.MultTranspose(shape, AdvUP);
        for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk]; 
      
        double Unorm = auxU.Norml2();
        CalculateTaus(nu, Unorm, tauMom, tauMass);      
        norm *= (-tauMom * nu );        
        
        gshape.Mult(auxU, advGrad);
         
         for (int kk = 0; kk < sp_dim; kk++)
        {
            MultVWt(Laplacian, advGrad, pelemat);
            pelemat *= norm;
            auxC.Transpose(pelemat);
            pelemat += auxC;        
            elmat.AddMatrix(pelemat, kk * nd, kk * nd);
        }         
         
    }    
}

void VectorNonLinearNSIntegrator::StabLapGradIntegrator(
                                    const FiniteElement &el, 
                                    ElementTransformation &Tr, 
                                    DenseMatrix &elmat)
{
    int dim = el.GetDim() + 1;
    int nd  = el.GetDof();
    int sp_dim = Tr.GetSpaceDim();
    DenseMatrix pelemat, Hessian;
    Vector Laplacian(nd);
    Vector AdvUP(dim), auxU(sp_dim), vec1; 
    int hess_size = (sp_dim * (sp_dim + 1)) / 2; 
    Hessian.SetSize(nd , hess_size);
    double norm;
    
    pelemat.SetSize (nd);
    
    int grad_order = Tr.OrderGrad(&el);
    int laporder = ((grad_order > 0) ? grad_order - 1 : 0 );
    int intorder = laporder +  Tr.OrderGrad(&el);
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
    
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcShape (ip, shape);
        el.CalcDShape(ip, dshape);
        //Only applies to 2D and up to order 3!!
        el.CalcHessian(ip, Hessian);

        Tr.SetIntPoint (&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse (Tr.Jacobian(), Jinv);
        Mult (dshape, Jinv, gshape);
        
        CalcPhysLaplacian (Hessian, nd, sp_dim, Laplacian);
        
        //compute tau
        double tauMom, tauMass;
        double nu = nuCoef->Eval (Tr, ip);
        PMatI.MultTranspose(shape, AdvUP);
        for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk]; 
      
        double Unorm = auxU.Norml2();
        CalculateTaus(nu, Unorm, tauMom, tauMass);      
        norm *= (-tauMom * nu );        
         
         for (int kk = 0; kk < sp_dim; kk++)
        {
            gshape.GetColumnReference(kk, vec1);
            MultVWt(Laplacian, vec1, pelemat);
            pelemat *= norm;
            elmat.AddMatrix(pelemat, nd * kk, sp_dim * nd);
            pelemat.Transpose();
            elmat.AddMatrix(pelemat, sp_dim * nd, nd * kk);
        }         
         
    }    
}

void VectorNonLinearNSIntegrator::StabLapLapIntegrator(
                                    const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    DenseMatrix &elmat)
{
    int dim = el.GetDim() + 1;
    int nd  = el.GetDof();
    int sp_dim = Tr.GetSpaceDim();
    DenseMatrix pelemat, Hessian;
    Vector Laplacian(nd);
    Vector AdvUP(dim), auxU(sp_dim); 
    int hess_size = (sp_dim * (sp_dim + 1)) / 2; 
    Hessian.SetSize(nd , hess_size);
    double norm;
    
    pelemat.SetSize (nd);
    
    int grad_order = Tr.OrderGrad(&el);
    int laporder = ((grad_order > 0) ? grad_order - 1 : 0 );
    int intorder = 2 * laporder;
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
    
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcShape (ip, shape);
        //Only applies to 2D and up to order 3!!
        el.CalcHessian(ip, Hessian);

        Tr.SetIntPoint (&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse (Tr.Jacobian(), Jinv);
        
        CalcPhysLaplacian (Hessian, nd, sp_dim, Laplacian);
        
        //compute tau
        double tauMom, tauMass;
        double nu = nuCoef->Eval (Tr, ip);
        PMatI.MultTranspose(shape, AdvUP);
        for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk]; 
      
        double Unorm = auxU.Norml2();
        CalculateTaus(nu, Unorm, tauMom, tauMass);      
        norm *= (tauMom * nu * nu);        
        
        for (int ii = 0; ii < sp_dim; ii++)
        {
            MultVVt (Laplacian, pelemat);
            pelemat *= norm;
            elmat.AddMatrix(pelemat, ii*nd, ii*nd);            
        }  
    }
    
}

void VectorNonLinearNSIntegrator::StabLaplacianIntegrator(
                                    const FiniteElement &el, 
                                    ElementTransformation &Trans, 
                                    DenseMatrix &elmat)
{
   int dim = el.GetDim() + 1; //1 is for pressure 
   int nd  = el.GetDof();
   int sp_dim = Trans.GetSpaceDim();   
   DenseMatrix pelemat;
   double norm;
   Vector AdvUP(dim), auxU(sp_dim);  

   pelemat.SetSize (nd);

   int intorder = 2 * Trans.OrderGrad(&el);
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      el.CalcDShape (ip, dshape);
      el.CalcShape  (ip, shape);

      Trans.SetIntPoint (&ip);
      norm = ip.weight * Trans.Weight();
      CalcInverse (Trans.Jacobian(), Jinv);

      Mult (dshape, Jinv, gshape);
 
      //compute tau
      double tauMom, tauMass;
      double nu = nuCoef->Eval (Trans, ip);
      PMatI.MultTranspose(shape, AdvUP);
      for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk]; 
      
      double Unorm = auxU.Norml2();
      CalculateTaus(nu, Unorm, tauMom, tauMass);      
      norm *= tauMom;

      MultAAt (gshape, pelemat);
      pelemat *= norm;
      
      elmat.AddMatrix(pelemat, sp_dim*nd, sp_dim*nd);
   }    
}

void VectorNonLinearNSIntegrator::SatbDivIntegrator(const FiniteElement &el,
                                                    ElementTransformation &Tr,
                                                    DenseMatrix &elmat)
{
   int dim = el.GetDim() + 1; //1 is for pressure 
   int nd  = el.GetDof();
   int sp_dim = Tr.GetSpaceDim();   
   DenseMatrix pelemat;
   double norm;
   Vector AdvUP(dim), auxU(sp_dim); 
   Vector aux1, aux2;

   pelemat.SetSize (nd);

   int intorder = 2 * Tr.OrderGrad(&el);
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      el.CalcDShape (ip, dshape);
      el.CalcShape  (ip, shape);

      Tr.SetIntPoint (&ip);
      norm = ip.weight * Tr.Weight();
      CalcInverse (Tr.Jacobian(), Jinv);

      Mult (dshape, Jinv, gshape);
 
      //compute tau
      double tauMom, tauMass;
      double nu = nuCoef->Eval (Tr, ip);
      PMatI.MultTranspose(shape, AdvUP);
      for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk]; 
      
      double Unorm = auxU.Norml2();
      CalculateTaus(nu, Unorm, tauMom, tauMass);      
      norm *= tauMass;
      
      for (int kk = 0; kk < sp_dim; ++kk)
      {
          gshape.GetColumnReference(kk, aux1);
          
          for (int jj = 0 ; jj < sp_dim; jj++)
          {
              gshape.GetColumnReference(jj, aux2);
              MultVWt(aux1, aux2, pelemat);
              pelemat *= norm;
              elmat.AddMatrix(pelemat, nd * kk, nd * jj);
          }
      }

   }       
}

void VectorNonLinearNSIntegrator::StabVecAdvIntegrator(const FiniteElement &el,
                                      ElementTransformation &Tr, Vector &elvect)
{
   double norm;
   
   int dim = el.GetDim() + 1;
   int nd  = el.GetDof();
   int sp_dim = Tr.GetSpaceDim();     
   Vector vec1(nd);
   Vector AdvUP(dim), auxU(sp_dim), advGrad(nd);     

   int intorder = el.GetOrder() + Tr.OrderGrad(&el) + 1; //Assuming linear body force 
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
   
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint (&ip);
      
      el.CalcDShape(ip, dshape);
      el.CalcShape (ip, shape);
      CalcInverse (Tr.Jacobian(), Jinv);
      Mult (dshape, Jinv, gshape);
    
      //Body force vector has dim components 
      Vector exQvec;
      exQvec.SetSize(dim);
      bdfVec->Eval(exQvec, Tr, ip);
      //Vector Qvec(exQvec.GetData() , dim-1);
      
      //compute tau
      double tauMom, tauMass;
      double nu = nuCoef->Eval (Tr, ip);
      PMatI.MultTranspose(shape, AdvUP);
      for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk];  
      
      double Unorm = auxU.Norml2();
      CalculateTaus(nu, Unorm, tauMom, tauMass);   
      
      gshape.Mult(auxU, advGrad);
      
      norm =  tauMom * ip.weight * Tr.Weight();   
 
      for (int kk = 0; kk < sp_dim; kk++)
      {
          int ro = nd * kk;
          for (int ll = 0; ll < nd; ll++)
          {
              elvect[ro + ll] -= norm * advGrad[ll] * exQvec[kk];
          }
      }      
   
   }     
}

void VectorNonLinearNSIntegrator::StabVecGradIntegrator(const FiniteElement &el,
                                      ElementTransformation &Tr, Vector &elvect)
{
   double norm;
   
   int dim = el.GetDim() + 1;
   int nd  = el.GetDof();
   int sp_dim = Tr.GetSpaceDim();     
   Vector vec1(nd);
   Vector AdvUP(dim), auxU(sp_dim); 
   shape.SetSize(nd);

   int intorder = Tr.OrderGrad(&el) + 1; //Assuming linear body force 
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);
   
   int ro = sp_dim * nd;
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint (&ip);
      
      el.CalcDShape(ip, dshape);
      el.CalcShape (ip, shape);
      CalcInverse (Tr.Jacobian(), Jinv);
      Mult (dshape, Jinv, gshape);
    
      //Body force vector has 4 components 
      //to take into account pressure. We extract the 
      //velocity part here.
      Vector exQvec;
      exQvec.SetSize(dim);
      bdfVec->Eval(exQvec, Tr, ip);
      Vector Qvec(exQvec.GetData() , dim-1);
      gshape.Mult(Qvec, vec1);
      
      //compute tau
      double tauMom, tauMass;
      double nu = nuCoef->Eval (Tr, ip);
      PMatI.MultTranspose(shape, AdvUP);
      for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk];  
      
      double Unorm = auxU.Norml2();
      CalculateTaus(nu, Unorm, tauMom, tauMass);      
      
      norm =  tauMom * ip.weight * Tr.Weight();   
 
      //Last block is pressure
      for (int kk = 0; kk < nd; ++kk)
          elvect[ro + kk] -= norm * vec1[kk];      
   
   }    
}

void VectorNonLinearNSIntegrator::StabVecLapIntegrator(const FiniteElement &el,
                                      ElementTransformation &Tr, Vector &elvect)
{
   double norm;
   
   int dim = el.GetDim() + 1;
   int nd  = el.GetDof();
   int sp_dim = Tr.GetSpaceDim();     
   Vector AdvUP(dim), auxU(sp_dim); 
   shape.SetSize(nd);
   int hess_size = (sp_dim * (sp_dim + 1)) / 2; 
   DenseMatrix Hessian;
   Vector Laplacian(nd);
   Hessian.SetSize(nd , hess_size);   

   int grad_order = Tr.OrderGrad(&el);
   int laporder = ((grad_order > 0) ? grad_order - 1 : 0 );
   int intorder = laporder + 1;//Assuming linear body force
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);   

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint (&ip);
      
      el.CalcShape (ip, shape);
      el.CalcHessian(ip, Hessian);
      CalcInverse (Tr.Jacobian(), Jinv);
   
      CalcPhysLaplacian (Hessian, nd, sp_dim, Laplacian);
    
      //Body force vector has 4 components 
      //to take into account pressure. We extract the 
      //velocity part here.
      Vector exQvec;
      exQvec.SetSize(dim);
      bdfVec->Eval(exQvec, Tr, ip);
      //Vector Qvec(exQvec.GetData() , dim-1);
      
      //compute tau
      double tauMom, tauMass;
      double nu = nuCoef->Eval (Tr, ip);
      PMatI.MultTranspose(shape, AdvUP);
      for (int kk = 0; kk < sp_dim; kk++)
           auxU[kk] = AdvUP[kk];  
      
      double Unorm = auxU.Norml2();
      CalculateTaus(nu, Unorm, tauMom, tauMass);      
      
      norm =  nu * tauMom * ip.weight * Tr.Weight();   
 
      for (int kk = 0; kk < sp_dim; kk++)
      {
          int ro = nd * kk;
          for (int ll = 0; ll < nd; ll++)
          {
              elvect[ro + ll] += norm * Laplacian[ll] * exQvec[kk];
          }
      }      
   
   }    
}

void VectorNonLinearNSIntegrator::CalculateTaus(const double nu, 
        const double normVel, double& tauMom, double& tauMass)
{
    tauMom = tauMass = 0.0;
    
    double invtau = 2.0 * normVel / eleLength + 4.0 * nu / (eleLength * eleLength);     
    tauMom = 1.0/invtau;   
    
    tauMass = nu + 0.5 * eleLength * normVel;
   
}

void VectorNonLinearNSIntegrator::CalcPhysLaplacian(DenseMatrix &Hessian, 
                                                    double nnodes,
                                                    double spaceDim,        
                                                    Vector& Laplacian)
{
    //Compute the laplacian assuming linear transformation
    // it implements Lap(N) = J^-T Hessian(N) J^-1
    // it assumes each row i of H represents second derivatives 
    // of Ni saved in this order {H00, H01, H02, H12, H22, H11}
    Vector JinTH(spaceDim);
    Vector LapComp(spaceDim);
    
    for (int nd = 0; nd < nnodes; nd++)
    {
        Vector vecHess;
        Hessian.GetRow(nd , vecHess);
        LapComp = 0.0;
        
        for (int ll = 0; ll < spaceDim; ll++)
        {
            JinTH = 0.0;
            for (int ii = 0; ii < spaceDim; ii++)
            {
                for (int jj = 0; jj < spaceDim; jj++)
                {
                  //In 3D H11 is stored at the 5th component
                  //Other components position follow the i+j
                  int id = ((ii == 1 && jj == 1 && spaceDim == 3) ? 5 : ii + jj);
                    
                  JinTH[ii] += Jinv(jj , ll) * vecHess[id]; 
                }
            }
            
            for (int kk = 0; kk < spaceDim; kk++)
                LapComp[ll] += JinTH[kk] * Jinv(kk , ll);
        }
      
        //Sum (d2Ni/dxi2)
        Laplacian[nd] = LapComp.Sum();
    }
};

VectorNonLinearNSIntegrator::~VectorNonLinearNSIntegrator()
{
   PMatI.ClearExternalData();
   PMatO.ClearExternalData();
   delete bdfVec;
   delete nuCoef;
}    
}
