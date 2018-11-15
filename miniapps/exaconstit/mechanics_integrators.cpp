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

#include "mfem.hpp"
#include "mechanics_integrators.hpp"
#include "BCManager.hpp"
#include <math.h> // log 
#include <iostream> // cerr

namespace mfem
{

using namespace std;

void computeDefGrad(const QuadratureFunction *qf, ParFiniteElementSpace *fes, 
                    const Vector &x0)
{
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();

   ParGridFunction x_gf;
   
   double* vals = x0.GetData();

   const int NE = fes->GetNE();

   int dim = x0.Size() / 3;

   x_gf.MakeTRef(fes, vals);
   x_gf.SetFromTrueVector();

   
   // loop over elements
   for (int i = 0; i < NE; ++i)
   {
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);

      // declare data to store shape function gradients 
      // and element Jacobians
      DenseMatrix Jrt, DSh, DS, PMatI, Jpt, F0, F1;
      int dof = fe->GetDof(), dim = fe->GetDim();

      if (qf_offset != (dim*dim))
      {
         mfem_error("computeDefGrd0 stride input arg not dim*dim");
      }

      DSh.SetSize(dof,dim);
      DS.SetSize(dof,dim);
      Jrt.SetSize(dim);
      Jpt.SetSize(dim);
      F0.SetSize(dim);
      F1.SetSize(dim);
      PMatI.SetSize(dof, dim);
      
      // get element physical coordinates
      //Array<int> vdofs;
      //Vector el_x;
      //fes->GetElementVDofs(i, vdofs);
      //x0.GetSubVector(vdofs, el_x);
      //PMatI.UseExternalData(el_x.GetData(), dof, dim);

      // get element physical coordinates
      Array<int> vdofs(dof * dim);
      Vector el_x(PMatI.Data(), dof * dim);
      fes->GetElementVDofs(i, vdofs);
      
      x_gf.GetSubVector(vdofs, el_x);
      
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();

      // loop over integration points where the quadrature function is 
      // stored
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr->SetIntPoint(&ip);
         CalcInverse(Ttr->Jacobian(), Jrt);
         
         fe->CalcDShape(ip, DSh);
         Mult(DSh, Jrt, DS);
         MultAtB(PMatI, DS, Jpt); 

         // store local beginning step deformation gradient for a given 
         // element and integration point from the quadrature function 
         // input argument. We want to set the new updated beginning 
         // step deformation gradient (prior to next time step) to the current
         // end step deformation gradient associated with the converged 
         // incremental solution. The converged _incremental_ def grad is Jpt 
         // that we just computed above. We compute the updated beginning 
         // step def grad as F1 = Jpt*F0; F0 = F1; We do this because we 
         // are not storing F1.
         int k = 0; 
         for (int n = 0; n < dim; ++n)
         {
            for (int m = 0; m < dim; ++m)
            {
               F0(m,n) = qf_data[i * elem_offset + j * qf_offset + k];
               ++k;
            }
         }

         // compute F1 = Jpt*F0;
         Mult(Jpt, F0, F1);

         // set new F0 = F1
         F0 = F1;
  
	 // loop over element Jacobian data and populate 
         // quadrature function with the new F0 in preparation for the next 
         // time step. Note: offset0 should be the 
         // number of true state variables. 
         k = 0; 
         for (int m = 0; m < dim; ++m)
         {
            for (int n = 0; n < dim; ++n)
            {
               qf_data[i * elem_offset + j * qf_offset + k] = 
                  F0(n,m);
               ++k;
            }
         }
	 //Temporary method to check and see what's being outputted
	 /*if( j == 0 && i == 45){
	   printf("Def Grad at elem 1 and ip 1:\n");
	   for (int m = 0; m < dim; ++m){
            for (int n = 0; n < dim; ++n){
	      printf("%lf ", F0(n,m));
            }
	   }
	   printf("\n");
	   }*/
      }
      Ttr = NULL;
   }

   fe = NULL;
   ir = NULL;
   qf_data = NULL;
   qspace = NULL;
   
   return;
}
   
void computeDefGradTest(const QuadratureFunction *qf, ParFiniteElementSpace *fes, const Vector &x0)
{
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();
   ParGridFunction x_gf;
   
   double* vals = x0.GetData();

   const int NE = fes->GetNE();

   int dim = x0.Size() / 3;

   x_gf.MakeTRef(fes, vals);
   x_gf.SetFromTrueVector();

   // loop over elements
   //Only going to test it for one element
   for (int i = 0; i < NE; ++i)
   {
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);
      
      // declare data to store shape function gradients
      // and element Jacobians
      DenseMatrix Jrt, DSh, DS, PMatI, Jpt, F0, F1;
      int dof = fe->GetDof(), dim = fe->GetDim();
      
      DSh.SetSize(dof,dim);
      DS.SetSize(dof,dim);
      Jrt.SetSize(dim);
      Jpt.SetSize(dim);
      F0.SetSize(dim);
      F1.SetSize(dim);
      PMatI.SetSize(dof, dim);
      
      // get element physical coordinates
      Array<int> vdofs(dof * dim);
      Vector el_x(PMatI.Data(), dof * dim);
      fes->GetElementVDofs(i, vdofs);
      
      x_gf.GetSubVector(vdofs, el_x);
      //PMatI.UseExternalData(el_x.GetData(), dof, dim);
      
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = 0;
      
      // loop over integration points where the quadrature function is
      // stored
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr->SetIntPoint(&ip);
         CalcInverse(Ttr->Jacobian(), Jrt);
         
         fe->CalcDShape(ip, DSh);
         Mult(DSh, Jrt, DS);
         MultAtB(PMatI, DS, Jpt);
         
         // store local beginning step deformation gradient for a given
         // element and integration point from the quadrature function
         // input argument. We want to set the new updated beginning
         // step deformation gradient (prior to next time step) to the current
         // end step deformation gradient associated with the converged
         // incremental solution. The converged _incremental_ def grad is Jpt
         // that we just computed above. We compute the updated beginning
         // step def grad as F1 = Jpt*F0; F0 = F1; We do this because we
         // are not storing F1.
	 //         int k = 0;
         for (int m = 0; m < dim; ++m)
         {
            for (int n = 0; n < dim; ++n)
            {
               F0(m,n) = 0.0;
            }
            F0(m,m) = 1.0;
         }
         
         // compute F1 = Jpt*F0;
         Mult(Jpt, F0, F1);
         
         // set new F0 = F1
         F0 = F1;
      }
   }
   
   return;
}

// member functions for the Abaqus Umat base class
int ExaModel::GetStressOffset()
{
   QuadratureFunction* qf = stress0.GetQuadFunction();
   int qf_offset = qf->GetVDim();

   qf = NULL;
   
   return qf_offset;
}

int ExaModel::GetMatGradOffset()
{
   QuadratureFunction* qf = matGrad.GetQuadFunction();
   int qf_offset = qf->GetVDim();

   qf = NULL;
   
   return qf_offset;
}

int ExaModel::GetMatVarsOffset()
{
   QuadratureFunction* qf = matVars0.GetQuadFunction();
   int qf_offset = qf->GetVDim();

   qf = NULL;

   return qf_offset;
}

// the getter simply returns the beginning step stress
void ExaModel::GetElementStress(const int elID, const int ipNum,
                                bool beginStep, double* stress, int numComps)
{
   const IntegrationRule *ir = NULL;
   double* qf_data = NULL;
   int qf_offset = 0;
   QuadratureFunction* qf = NULL;
   QuadratureSpace* qspace = NULL;

   if (beginStep)
   {
     qf = stress0.GetQuadFunction();
   }
   else
   {
     qf = stress1.GetQuadFunction();
   }

   qf_data   = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace    = qf->GetSpace();

   /* for (int i=0; i<qf->Size(); ++i)
   {
      printf("GetStress1: %f \n", qf_data[i]);
   }
   */
   
   // check offset to input number of components
   if (qf_offset != numComps)
   {
      cerr << "\nGetElementStress: number of components does not match quad func offset" 
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   /*
   Vector vals;
   qf->GetElementValues(elID, &vals);

   for (int j = 0; j < ir->GetNPoints(); ++j)
   {
      for (int k = 0; k<numComps; ++k)
      {
         printf("element stress: %f \n", vals[numComps*j + k]);
      }
   }
   */

   //printf("elID and ipNum: %d %d \n", elID, ipNum);
   for (int i=0; i<numComps; ++i)
   {
     stress[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
     //stress[i] = vals[ipNum*qf_offset + i];
     //printf("stress from get routine: %f \n", stress[i]);
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;

   return;
}

void ExaModel::SetElementStress(const int elID, const int ipNum, 
                                bool beginStep, double* stress, int numComps)
{
  //printf("inside ExaModel::SetElementStress, elID, ipNum %d %d \n", elID, ipNum);
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   if (beginStep)
   {
     qf = stress0.GetQuadFunction();
   }
   else
   {
     //printf("getting stress1 \n");
     qf = stress1.GetQuadFunction();
   }

   qf_data   = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace    = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps)
   {
      cerr << "\nSetElementStress: number of components does not match quad func offset" 
           << endl;
   }

   //printf("before setting ir \n");
   ir = &(qspace->GetElementIntRule(elID));
   //printf("after setting ir \n");
   int elem_offset = qf_offset * ir->GetNPoints();

   //printf("entering offset loop \n");
   for (int i=0; i<qf_offset; ++i)
   {
     int k = elID * elem_offset + ipNum * qf_offset + i;
     qf_data[k] = stress[i];
     //     printf("qf_data[k]: %f \n", qf_data[k]);
   }
   return;
}

void ExaModel::GetElementStateVars(const int elID, const int ipNum, 
                                   bool beginStep, double* stateVars,
                                   int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   if (beginStep)
   {
     qf = matVars0.GetQuadFunction();
   }
   else
   {
     qf = matVars1.GetQuadFunction();
   }

   qf_data   = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace    = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps)
   {
      cerr << "\nGetElementStateVars: num. components does not match quad func offset" 
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<numComps; ++i)
   {
     stateVars[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;

   return;
}

void ExaModel::SetElementStateVars(const int elID, const int ipNum, 
                                   bool beginStep, double* stateVars,
                                   int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   if (beginStep)
   {
     qf = matVars0.GetQuadFunction();
   }
   else
   {
     qf = matVars1.GetQuadFunction();
   }

   qf_data   = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace    = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps)
   {
      cerr << "\nSetElementStateVars: num. components does not match quad func offset" 
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<qf_offset; ++i)
   {
     qf_data[elID * elem_offset + ipNum * qf_offset + i] = stateVars[i];
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;
   
   return;
}

void ExaModel::GetElementMatGrad(const int elID, const int ipNum, double* grad,
                                 int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   qf = matGrad.GetQuadFunction();

   qf_data   = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace    = qf->GetSpace();

   //for (int i=0; i<qf->Size(); ++i)
   //{
   //   printf("mat grad: %f \n", qf_data[i]);   
   //}

   // check offset to input number of components
   if (qf_offset != numComps)
   {
      cerr << "\nGetElementMatGrad: num. components does not match quad func offset" 
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<numComps; ++i)
   {
     grad[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
   }

   ir =	NULL;
   qf_data = NULL;
   qf =	NULL;
   qspace = NULL;

   return;
}

void ExaModel::SetElementMatGrad(const int elID, const int ipNum, 
                                 double* grad, int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   qf = matGrad.GetQuadFunction();

   qf_data   = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace    = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps)
   {
      cerr << "\nSetElementMatGrad: num. components does not match quad func offset" 
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<qf_offset; ++i)
   {
     int k = elID * elem_offset + ipNum * qf_offset + i;
     //printf("SetElementMatGrad qf_data comp (k): %d %f \n", k, qf_data[k]);
     qf_data[k] = grad[i];
   }

   ir =	NULL;
   qf_data = NULL;
   qf =	NULL;
   qspace = NULL;
   
   return;
}

void ExaModel::GetMatProps(double* props)
{
   props = matProps.GetData();
   return;
}

void ExaModel::SetMatProps(double* props, int size)
{
   matProps.NewDataAndSize(props, size);
   return;
}

void ExaModel::GetElemDefGrad0()
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   qf = defGrad0.GetQuadFunction();

   qf_data   = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace    = qf->GetSpace();

   int dim = Ttr->GetDimension();

   // clear the Jacobian
   Jpt0.Clear();

   // set the size of the Jacobian
   Jpt0.SetSize(dim); 

   // initialize the Jacobian
   Jpt0 = 0.0;

   // loop over quadrature function data at the ip point 
   // currently set on the model, for the element id set 
   // on the model
   ir = &(qspace->GetElementIntRule(elemID));
   int elem_offset = qf_offset * ir->GetNPoints();

   int k = 0;
   for (int m=0; m<dim; ++m)
   {
      for (int n=0; n<dim; ++n)
      {
         Jpt0(n,m) = 
           qf_data[elemID * elem_offset + ipID * qf_offset + k];
         ++k;
      }
   }

   ir =	NULL;
   qf_data = NULL;
   qf =	NULL;
   qspace = NULL;
   
   return;
}
void ExaModel::CalcElemDefGrad1(const DenseMatrix& Jpt)
{
   int dim = Ttr->GetDimension();
   
   Jpt1.Clear();
   Jpt1.SetSize(dim);

   // full end step def grad, F1 = F_hat*F0, where F_hat is the Jpt passed 
   // in and represents the incremental deformation gradient associated with 
   // the incremental solution state. F0 = Jpt0 stored on the model and is the 
   // beginning step deformation gradient, which is the last step's end-step 
   // deformation gradient in the converged state.
   Mult(Jpt, Jpt0, Jpt1);
}

void ExaModel::UpdateStress(int elID, int ipNum)
{
  //printf("inside UdpateStress \n");
   const IntegrationRule *ir;
   double* qf0_data;
   double* qf1_data;
   int qf_offset;
   QuadratureFunction* qf0;
   QuadratureFunction* qf1;
   QuadratureSpace* qspace;

   qf0 = stress0.GetQuadFunction();
   qf1 = stress1.GetQuadFunction();

   qf_offset = qf0->GetVDim();
   qspace    = qf0->GetSpace();

   qf0_data  = qf0->GetData();
   qf1_data  = qf1->GetData();

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<qf_offset; ++i)
   {
     qf0_data[elID * elem_offset + ipNum * qf_offset + i] =
        qf1_data[elID * elem_offset + ipNum * qf_offset + i];
   }

   ir =	NULL;
   qf0_data = NULL;
   qf0 = NULL;
   qf1_data = NULL;
   qf1 = NULL;
   qspace = NULL;
   
   return;
}

void ExaModel::UpdateStateVars(int elID, int ipNum)
{
  //printf("inside UpdateStateVars \n");
   const IntegrationRule *ir;
   double* qf0_data;
   double* qf1_data;
   int qf_offset;
   QuadratureFunction* qf0;
   QuadratureFunction* qf1;
   QuadratureSpace* qspace;

   qf0 = matVars0.GetQuadFunction();
   qf1 = matVars1.GetQuadFunction();

   qf_offset = qf0->GetVDim();
   qspace    = qf0->GetSpace();

   qf0_data  = qf0->GetData();
   qf1_data  = qf1->GetData();

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<qf_offset; ++i)
   {
     qf0_data[elID * elem_offset + ipNum * qf_offset + i] =
        qf1_data[elID * elem_offset + ipNum * qf_offset + i];
   }
   
   ir = NULL;
   qf0_data = NULL;
   qf0 = NULL;
   qf1_data = NULL;
   qf1 = NULL;
   qspace = NULL;
   
   return;
}

void AbaqusUmatModel::UpdateModelVars( ParFiniteElementSpace *fes,
                              const Vector &x)
{
// update the beginning step deformation gradient
   QuadratureFunction* defGrad = defGrad0.GetQuadFunction();
   computeDefGrad(defGrad, fes, x);
   defGrad = NULL;
   return;
}
   
void NeoHookean::UpdateModelVars(ParFiniteElementSpace *fes,
                                 const Vector &x)
{
   const FiniteElement *fe;
   const IntegrationRule *ir;
   QuadratureFunction* defGrad = defGrad0.GetQuadFunction();
   
   // update the beginning step deformation gradient
   computeDefGrad(defGrad, fes, x);

   double* qf_data = defGrad->GetData();
   int qf_offset = defGrad->GetVDim(); // offset at each integration point                                   
   QuadratureSpace* qspace = defGrad->GetSpace();
   
   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i){
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);
      
      // set element id and attribute
      //SetTransformation(*Ttr);
      SetElementID(Ttr->ElementNo, Ttr->Attribute);
      
      // declare data to store shape function gradients
      // and element Jacobians
      DenseMatrix F0;
      int dof = fe->GetDof(), dim = fe->GetDim();
      F0.SetSize(dim);
      
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();
      
      // loop over integration points where the quadrature function
      // is stored
      for (int j = 0; j < ir->GetNPoints(); ++j){
         //We need to let the model know what IP we're currently at
         SetIpID(j);

	 //const IntegrationPoint &ip = ir->IntPoint(i);
	 //Ttr->SetIntPoint(&ip);

	 // get integration point coordinates
	 //SetIPCoords(ip.x, ip.y, ip.z);
         
         int k = 0;
         for (int n = 0; n < dim; ++n){
            for (int m = 0; m < dim; ++m){
               F0(m,n) = qf_data[i * elem_offset + j * qf_offset + k];
               ++k;
            }
         }
         EvalModel(F0);
      }
   }
   
   defGrad = NULL;
   return;
}

void ExaModel::SymVoigtToDenseMat(const double* const A, DenseMatrix& B)
{
   // note this assumes a stride of 6 in A, which are the 6 components 
   // of a 3x3 matrix (symmetric rank 2 tensor).

   int size = B.Size();

   if (size != 3)
   {
      B.SetSize(3);
      size = 3;
   }

   // set diagonal terms
   for (int i=0; i<size; ++i)
   {
      B(i,i) = A[i];
   }

   // set off-diagonal elements
   B(0,1) = A[5];
   B(0,2) = A[4];
   B(1,0) = B(0,1);
   B(2,0) = B(0,2);
   B(1,2) = A[3];
   B(2,1) = B(1,2);

   return;
}

void ExaModel::CauchyToPK1()
{
   double det;
   DenseMatrix FinvT;

   // compute the determinant of the END STEP deformation gradient
   det = Jpt1.Det();

   int size = Jpt1.Size();

   FinvT.SetSize(size);

   // set size of local PK1 stress matrix stored on the model
   P.SetSize(size);

   // calculate the inverse transpose of the matrix
   CalcInverseTranspose(Jpt1, FinvT);
   
   // get Cauchy stress
   double sig[6];
   GetElementStress(elemID, ipID, false, sig, 6);

   // populate full DenseMatrix 
   DenseMatrix Cauchy;
   Cauchy.SetSize(size);

   SymVoigtToDenseMat(sig, Cauchy); 

   // calculate first Piola Kirchhoff stress
   Mult(Cauchy, FinvT, P);
   P *= det;

   return;
}

void ExaModel::PK1ToCauchy(const DenseMatrix& P, const DenseMatrix& J, double* sigma)
{
   double det;
   DenseMatrix cauchy;
   DenseMatrix J_t;
   
   det = J.Det(); 
   int size = J.Size();
   cauchy.SetSize(size);
   J_t.SetSize(size);

   J_t.Transpose(J);

   Mult(P, J_t, cauchy);

   cauchy *= 1.0 / det;

   sigma[0] = cauchy(0,0);
   sigma[1] = cauchy(1,1);
   sigma[2] = cauchy(2,2);
   sigma[3] = cauchy(1,2);
   sigma[4] = cauchy(0,2);
   sigma[5] = cauchy(0,1);
   
   return;
}

void ExaModel::ComputeVonMises(const int elemID, const int ipID)
{
   QuadratureFunction *vm_qf = vonMises.GetQuadFunction();
   QuadratureSpace* vm_qspace = vm_qf->GetSpace();
   const IntegrationRule *ir;

   if (vm_qspace == NULL)
   {
      QuadratureFunction *qf_stress0 = stress0.GetQuadFunction();
      QuadratureSpace* qspace = qf_stress0->GetSpace();
      int vdim = 1; // scalar von Mises data at each IP
      vm_qf->SetSpace(qspace, vdim); // construct object

      qf_stress0 = NULL;
      qspace = NULL;
   }

   QuadratureSpace* qspace = vm_qf->GetSpace();
   double* vmData = vm_qf->GetData();
   int vmOffset = vm_qf->GetVDim();

   ir = &(qspace->GetElementIntRule(elemID));
   int elemVmOffset = vmOffset * ir->GetNPoints();

   double istress[6];
   GetElementStress(elemID, ipID, true, istress, 6);

   double term1 = istress[0] - istress[1];
   term1 *= term1;

   double term2 = istress[1] - istress[2];
   term2 *= term2;

   double term3 = istress[2] - istress[0];
   term3 *= term3;

   double term4 = istress[3]*istress[3] + istress[4]*istress[4] 
                  + istress[5]*istress[5];
   term4 *= 6.0;
                  
   double vm = 0.5 * sqrt(term1 + term2 + term3 + term4);

   // set the von Mises quadrature function data
   vmData[elemID * elemVmOffset + ipID * vmOffset] = vm;

   ir = NULL;
   vm_qspace = NULL;
   vm_qf = NULL;
   qspace = NULL;
   vmData = NULL;
   
   return;
}

//A helper function that takes in a 3x3 rotation matrix and converts it over
//to a unit quaternion.
void ExaModel::RMat2Quat(DenseMatrix& rmat, Vector& quat){
   double inv2 = 1.0/2.0;
   double phi = 0.0;
   static const double eps = numeric_limits<double>::epsilon();
   double tr_r = 0.0;
   double inv_sin = 0.0;
   double s = 0.0;
   
   quat = 0.0;
   
   tr_r = rmat(0, 0) + rmat(1, 1) + rmat(2, 2);
   phi = acos(inv2 * (tr_r - 1.0));
   if (abs(phi) < eps){
      quat[3] = 1.0;
   }else{
      inv_sin = 1.0 / sin(phi);
      quat[0] = phi;
      quat[1] = inv_sin * inv2 * (rmat(2, 1) - rmat(1, 2));
      quat[2] = inv_sin * inv2 * (rmat(0, 2) - rmat(2, 0));
      quat[3] = inv_sin * inv2 * (rmat(1, 0) - rmat(0, 1));
   }
   
   s = sin(inv2 * quat[0]);
   quat[0] = cos(quat[0] * inv2);
   quat[1] = s * quat[1];
   quat[2] = s * quat[2];
   quat[3] = s * quat[3];
   
   return;
   
}

//A helper function that takes in a unit quaternion and and returns a 3x3 rotation
//matrix.
void ExaModel::Quat2RMat(Vector& quat, DenseMatrix& rmat){
   double qbar = 0.0;
   
   qbar = quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
   
   rmat(0, 0) = qbar + 2.0 * quat[1] * quat[1];
   rmat(1, 0) = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
   rmat(2, 0) = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);
   
   rmat(0, 1) = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
   rmat(1, 1) = qbar + 2.0 * quat[2] * quat[2];
   rmat(2, 1) = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);
   
   rmat(0, 2) = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
   rmat(1, 2) = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
   rmat(2, 2) = qbar + 2.0 * quat[3] * quat[3];
   
   return;
}

//The below method computes the polar decomposition of a 3x3 matrix using a method
//proposed in: https://animation.rwth-aachen.de/media/papers/2016-MIG-StableRotation.pdf
//The paper listed provides a fast and robust way to obtain the rotation portion
//of a positive definite 3x3 matrix which then allows for the easy computation
//of U and V.
void ExaModel::CalcPolarDecompDefGrad(DenseMatrix& R, DenseMatrix& U,
                                      DenseMatrix& V, double err){

   DenseMatrix def_grad, omega_mat, temp;
   
   int dim = Ttr->GetDimension();
   Vector quat;
   
   int max_iter = 500;
   
   double norm, inv_norm;
   
   double ac1[3], ac2[3], ac3[3];
   double w_top[3], w[3];
   double w_bot, w_norm, w_norm_inv2, w_norm_inv;
   double cth, sth;
   double r1da1, r2da2, r3da3;
   
   quat.SetSize(4);
   omega_mat.SetSize(dim);
   temp.SetSize(dim);
   
   Mult(Jpt1, Jpt0, def_grad);
   
   RMat2Quat(def_grad, quat);
   
   norm = quat.Norml2();
   inv_norm = 1.0 / norm;
   
   quat *= inv_norm;
   
   Quat2RMat(quat, R);
   
   ac1[0] = def_grad(0,0); ac1[1] = def_grad(1,0); ac1[2] = def_grad(2,0);
   ac2[0] = def_grad(0,1); ac2[1] = def_grad(1,1); ac2[2] = def_grad(2,1);
   ac3[0] = def_grad(0,2); ac3[1] = def_grad(1,2); ac3[2] = def_grad(2,2);
   
   for (int i = 0; i < max_iter; i++) {
      //The dot products that show up in the paper
      r1da1 = R(0, 0) * ac1[0] + R(1, 0) * ac1[1] + R(2, 0) * ac1[2];
      r2da2 = R(0, 1) * ac2[0] + R(1, 1) * ac2[1] + R(2, 1) * ac2[2];
      r3da3 = R(0, 2) * ac3[0] + R(1, 2) * ac3[1] + R(2, 2) * ac3[2];
      
      //The summed cross products that show up in the paper
      w_top[0] = (-R(2, 0) * ac1[1] + R(1, 0) * ac1[2]) +
                 (-R(2, 1) * ac2[1] + R(1, 1) * ac2[2]) +
                 (-R(2, 2) * ac3[1] + R(1, 2) * ac3[2]);
      
      w_top[1] = (R(2, 0) * ac1[0] - R(0, 0) * ac1[2]) +
                 (R(2, 1) * ac2[0] - R(0, 1) * ac2[2]) +
                 (R(2, 2) * ac3[0] - R(0, 2) * ac3[2]);
      
      w_top[2] = (-R(1, 0) * ac1[0] + R(0, 0) * ac1[1]) +
                 (-R(1, 1) * ac2[0] + R(0, 1) * ac2[1]) +
                 (-R(1, 2) * ac3[0] + R(0, 2) * ac3[1]);
      
      w_bot = (1.0 / (abs(r1da1 + r2da2 + r3da3) + err));
      //The axial vector that shows up in the paper
      w[0] = w_top[0] * w_bot; w[1] = w_top[1] * w_bot; w[2] = w_top[2] * w_bot;
      //The norm of the axial vector
      w_norm = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
      //If the norm is below our desired error we've gotten our solution
      //So we can break out of the loop
      if(w_norm < err) break;
      //The exponential mapping for an axial vector
      //The 3x3 case has been explicitly unrolled here
      w_norm_inv2 = 1.0/(w_norm * w_norm);
      w_norm_inv = 1.0/w_norm;
      
      sth = sin(w_norm) * w_norm_inv;
      cth = (1.0 - cos(w_norm)) * w_norm_inv2;
      
      omega_mat(0, 0) = 1.0 - cth * (w[2] * w[2] + w[1] * w[1]);
      omega_mat(1, 1) = 1.0 - cth * (w[2] * w[2] + w[0] * w[0]);
      omega_mat(2, 2) = 1.0 - cth * (w[1] * w[1] + w[0] * w[0]);
      
      omega_mat(0, 1) = -sth * w[2] + cth * w[1] * w[0];
      omega_mat(0, 2) = sth  * w[1] + cth * w[2] * w[0];
      
      omega_mat(1, 0) = sth  * w[2] + cth * w[0] * w[1];
      omega_mat(1, 2) = -sth * w[0] + cth * w[2] * w[1];
      
      omega_mat(2, 0) = -sth * w[1] + cth * w[0] * w[2];
      omega_mat(2, 1) = sth  * w[0] + cth * w[2] * w[1];
      
      Mult(omega_mat, R, temp);
      R = temp;
   }
   
   MultAtB(R, def_grad, U);
   MultABt(def_grad, R, V);
   
   return;
}

//This method calculates the Eulerian strain which is given as:
//e = 1/2 (I - B^(-1)) = 1/2 (I - F(^-T)F^(-1))
void ExaModel::CalcEulerianStrain(DenseMatrix& E){

   DenseMatrix F, Finv, Binv;
   
   int dim = Ttr->GetDimension();
   
   double half = 1.0/2.0;
   
   Finv.SetSize(dim);
   Binv.SetSize(dim);
   
   Mult(Jpt1, Jpt0, F);
   
   CalcInverse(F, Finv);
   
   MultAtB(Finv, Finv, Binv);
   
   E = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         E(i, j) -= half * Binv(i, j);
      }
      E(j, j) += 1.0;
   }
   
   return;
}
   
//This method calculates the Lagrangian strain which is given as:
//E = 1/2 (C - I) = 1/2 (F^(T)F - I)
void ExaModel::CalcLagrangianStrain(DenseMatrix& E){

   DenseMatrix F, C;
   
   int dim = Ttr->GetDimension();
   
   double half = 1.0/2.0;
   
   C.SetSize(dim);
   
   Mult(Jpt1, Jpt0, F);
   
   MultAtB(F, F, C);
   
   E = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         E(i, j) += half * C(i, j);
      }
      E(j, j) -= 1.0;
   }
   
   return;
}
   
//This method calculates the Biot strain which is given as:
//E = (U - I)
void ExaModel::CalcBiotStrain(DenseMatrix& E){
   
   DenseMatrix rmat, umat, vmat;
   
   int dim = Ttr->GetDimension();
   
   rmat.SetSize(dim);
   umat.SetSize(dim);
   vmat.SetSize(dim);
   
   CalcPolarDecompDefGrad(rmat, umat, vmat);
   
   E = umat;
   E(0, 0) -= 1.0;
   E(1, 1) -= 1.0;
   E(2, 2) -= 1.0;
   
   return;
}

void ExaModel::CalcLogStrain(DenseMatrix& E)
{
   // calculate current end step logorithmic strain (Hencky Strain) 
   // which is taken to be E = ln(U) = 1/2 ln(C), where C = (F_T)F. 
   // We have incremental F from MFEM, and store F0 (Jpt0) so 
   // F = F_hat*F0. With F, use a spectral decomposition on C to obtain a 
   // form where we only have to take the natural log of the 
   // eigenvalues

   DenseMatrix C, F;

   int dim = Ttr->GetDimension();

   C.SetSize(dim); 

   Mult(Jpt1, Jpt0, F);

   MultAtB(F, F, C);

   // compute eigenvalue decomposition of C
   double lambda[dim];
   double vec[dim];
   C.CalcEigenvalues(lambda, vec);

   // compute ln(C) using spectral representation
   E = 0.0;
   for (int i=0; i<dim; ++i) // outer loop for every eigenvalue/vector
   {
      for (int j=0; j<dim; ++j) // inner loops for diadic product of eigenvectors
      {
         for (int k=0; k<dim; ++k)
         {
            //Dense matrices are col. maj. representation, so the indices were
            //reversed for it to be more cache friendly.
            E(k,j) += 0.5 * log(lambda[i]) * vec[i*dim+j] * vec[i*dim+k];
         }
      }
   }

   return;
}

void AbaqusUmatModel::CalcLogStrainIncrement(DenseMatrix& dE)
{
   // calculate incremental logorithmic strain (Hencky Strain) 
   // which is taken to be E = ln(U_hat) = 1/2 ln(C_hat), where 
   // C_hat = (F_hat_T)F_hat, where F_hat = Jpt1 on the model 
   // (available from MFEM element transformation computations). 
   // We can compute F_hat, so use a spectral decomposition on C_hat to 
   // obtain a form where we only have to take the natural log of the 
   // eigenvalues

   DenseMatrix F_hat, C_hat;

   int dim = Ttr->GetDimension();

   F_hat.SetSize(dim);
   C_hat.SetSize(dim); 

   F_hat = Jpt1;

   MultAtB(F_hat, F_hat, C_hat);

   // compute eigenvalue decomposition of C
   double lambda[dim];
   double vec[dim];
   C_hat.CalcEigenvalues(lambda, vec);

   // compute ln(C) using spectral representation
   dE = 0.0;
   for (int i=0; i<dim; ++i) // outer loop for every eigenvalue/vector
   {
      for (int j=0; j<dim; ++j) // inner loops for diadic product of eigenvectors
      {
         for (int k=0; k<dim; ++k)
         {
            //Dense matrices are col. maj. representation, so the indices were
            //reversed for it to be more cache friendly.
            dE(k,j) += 0.5 * log(lambda[i]) * vec[i*dim+j] * vec[i*dim+k];
         }
      }
   }

   return;
}
   
//This method calculates the Eulerian strain which is given as:
//e = 1/2 (I - B^(-1)) = 1/2 (I - F(^-T)F^(-1))
void AbaqusUmatModel::CalcEulerianStrainIncr(DenseMatrix& dE){
   
   DenseMatrix Finv, Binv;
   
   int dim = Ttr->GetDimension();
   
   double half = 1.0/2.0;
   
   Finv.SetSize(dim);
   Binv.SetSize(dim);
   
   CalcInverse(Jpt1, Finv);
   
   MultAtB(Finv, Finv, Binv);
   
   dE = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         dE(i, j) -= half * Binv(i, j);
      }
      dE(j, j) += 1.0;
   }
}

//This method calculates the Lagrangian strain which is given as:
//E = 1/2 (C - I) = 1/2 (F^(T)F - I)
void AbaqusUmatModel::CalcLagrangianStrainIncr(DenseMatrix& dE){
   
   DenseMatrix C;
   
   int dim = Ttr->GetDimension();
   
   double half = 1.0/2.0;
   
   C.SetSize(dim);
   
   MultAtB(Jpt1, Jpt1, C);
   
   dE = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         dE(i, j) += half * C(i, j);
      }
      dE(j, j) -= 1.0;
   }
   return;
}

// NOTE: this UMAT interface is for use only in ExaConstit and considers 
// only mechanical analysis. There are no thermal effects. Any thermal or 
// thermo-mechanical coupling variables for UMAT input are null.
void AbaqusUmatModel::EvalModel(const DenseMatrix &J, const DenseMatrix &DS,
                                const double weight)
{
   //======================================================
   // Set UMAT input arguments 
   //======================================================

   // initialize Umat variables
   int ndi   = 3; // number of direct stress components
   int nshr  = 3; // number of shear stress components
   int ntens = ndi + nshr;
   int noel  = elemID; // element id
   int npt   = ipID; // integration point number 
   int layer = 0; 
   int kspt  = 0;
   int kstep = 0;
   int kinc  = 0;
   
   // set properties and state variables length (hard code for now);
   int nprops = numProps;
   int nstatv = numStateVars;

   double pnewdt = 10.0; // revisit this
   double props[nprops];  // populate from the mat props vector wrapped by matProps on the base class
   double statev[nstatv]; // populate from the state variables associated with this element/ip

   double rpl        = 0.0;   // volumetric heat generation per unit time, not considered
   double drpldt     = 0.0;   // variation of rpl wrt temperature set to 0.0
   double tempk       = 0.0;   // no thermal considered at this point
   double dtemp      = 0.0;   // no increment in thermal considered at this point
   double predef  = 0.0; // no interpolated values of predefined field variables at ip point
   double dpred   = 0.0; // no array of increments of predefined field variables
   double sse        = 0.0;   // specific elastic strain energy, mainly for output
   double spd        = 0.0;   // specific plastic dissipation, mainly for output
   double scd        = 0.0;   // specific creep dissipation, mainly for output
   double cmname     = 0.0;   // user defined UMAT name
   double celent     = 0.0;   // set element length 

   // compute characteristic element length
   CalcElemLength();
   celent = elemLength;

   // integration point coordinates
   double coords[3];
   coords[0] = ipx;
   coords[1] = ipy;
   coords[2] = ipz;

   // set the time step
   double deltaTime = dt; // set on the ExaModel base class

   // set time. Abaqus has odd increment definition. time[1] is the value of total 
   // time at the beginning of the current increment. Since we are iterating from 
   // tn to tn+1, this is just tn. time[0] is value of step time at the beginning 
   // of the current increment. What is step time if not tn? It seems as though 
   // they sub-increment between tn->tn+1, where there is a Newton Raphson loop 
   // advancing the sub-increment. For now, set time[0] to 0.0 and time[1] to the 
   // actual beginning step total time.
   double time[2];
   time[0] = 0.0;
   time[1] = t; 

   double stress[6]; // Cauchy stress at ip 
   double ddsdt[6]; // variation of the stress increments wrt to temperature, set to 0.0
   double drplde[6]; // variation of rpl wrt strain increments, set to 0.0
   double stran[6];  // array containing total strains at beginning of the increment
   double dstran[6]; // array of strain increments

   // initialize 1d arrays
   for (int i=0; i<6; ++i) {
      stress[i] = 0.0;
      ddsdt[i] = 0.0;
      drplde[i] = 0.0;
      stran[i]  = 0.0;
      dstran[i] = 0.0;
   } 

   double ddsdde[36]; // output Jacobian matrix of the constitutive model.
                        // ddsdde(i,j) defines the change in the ith stress component 
                        // due to an incremental perturbation in the jth strain increment

   // initialize 6x6 2d arrays
   for (int i=0; i<6; ++i) {
      for (int j=0; j<6; ++j) {
         ddsdde[(i * 6) + j] = 0.0;
      }
   }

   double drot[9];   // rotation matrix for finite deformations
   double dfgrd0[9]; // deformation gradient at beginning of increment
   double dfgrd1[9]; // defomration gradient at the end of the increment.
                        // set to zero if nonlinear geometric effects are not 
                        // included in the step as is the case for ExaConstit
   
   // initialize 3x3 2d arrays to identity
   //drot is the only one that we need to do this for the
   //other 3x3 are set a little bit further below.
   for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
         drot[(i * 3) + j]   = 0.0;
         if (i == j)
         {
            drot[(i * 3) + j]   = 1.0;
         }
      }
   }

   // populate the beginning step and end step (or best guess to end step 
   // within the Newton iterations) of the deformation gradients
   for (int i=0; i<ndi; ++i)
   {
      for (int j=0; j<ndi; ++j)
      {
         //Dense matrices have column major layout so the below is fine.
         dfgrd0[(i * 3) + j] = Jpt0(i,j);
         dfgrd1[(i * 3) + j] = Jpt1(i,j);
      }
   }

   // get state variables and material properties
   GetElementStateVars(elemID, ipID, true, statev, nstatv);
   GetMatProps(props);

   // get element stress and make sure ordering is ok
   double stressTemp[6];
   GetElementStress(elemID, ipID, true, stressTemp, 6);

   // ensure proper ordering of the stress array. ExaConstit uses 
   // Voigt notation (11, 22, 33, 23, 13, 12), while 
   //------------------------------------------------------------------
   // We use Voigt notation: (11, 22, 33, 23, 13, 12)
   //
   // ABAQUS USES: 
   // (11, 22, 33, 12, 13, 23)
   //------------------------------------------------------------------
   stress[0] = stressTemp[0];
   stress[1] = stressTemp[1];
   stress[2] = stressTemp[2];
   stress[3] = stressTemp[5];
   stress[4] = stressTemp[4];
   stress[5] = stressTemp[3];

   //Abaqus does mention wanting to use a log strain for large strains
   //It's also based on an updated lagrangian formulation so as long as
   //we aren't generating any crazy strains do we really need to use the
   //log strain?
   DenseMatrix LogStrain;
   LogStrain.SetSize(ndi); // ndi x ndi

   CalcEulerianStrain(LogStrain);

   // populate STRAN (symmetric) 
   //------------------------------------------------------------------
   // We use Voigt notation: (11, 22, 33, 23, 13, 12)
   //
   // ABAQUS USES: 
   // (11, 22, 33, 12, 13, 23)
   //------------------------------------------------------------------
   stran[0] = LogStrain(0,0);
   stran[1] = LogStrain(1,1);
   stran[2] = LogStrain(2,2);
   stran[3] = LogStrain(0,1);
   stran[4] = LogStrain(0,2);
   stran[5] = LogStrain(1,2);

   // compute incremental strain, DSTRAN
   DenseMatrix dLogStrain;
   dLogStrain.SetSize(ndi);

   CalcLogStrainIncrement(dLogStrain);

   // populate DSTRAN (symmetric)
   //------------------------------------------------------------------
   // We use Voigt notation: (11, 22, 33, 23, 13, 12)
   //
   // ABAQUS USES: 
   // (11, 22, 33, 12, 13, 23)
   //------------------------------------------------------------------
   dstran[0] = dLogStrain(0,0);
   dstran[1] = dLogStrain(1,1);
   dstran[2] = dLogStrain(2,2);
   dstran[3] = dLogStrain(0,1);
   dstran[4] = dLogStrain(0,2);
   dstran[5] = dLogStrain(1,2);
   
   

   // call c++ wrapper of umat routine
            umat(stress, statev, ddsdde, &sse, &spd, &scd, &rpl,
                    ddsdt, drplde, &drpldt, stran, dstran, time,
                    &deltaTime, &tempk, &dtemp, &predef,
                    &dpred, &cmname, &ndi, &nshr, &ntens,
                    &nstatv, props, &nprops, coords, drot, &pnewdt, &celent,
                    dfgrd0, dfgrd1, &noel, &npt, &layer, &kspt, &kstep, &kinc);

   // restore the material Jacobian in a 1D array
   double mGrad[36];

   for (int i=0; i<6; ++i)
   {
     for (int j=0; j<6; ++j)
     {
       // row-wise ordering of material Jacobian
       mGrad[(6 * i) + j] = ddsdde[(i * 6) + j];
     }
     //      mGrad[(6 * i) + j] = 1.0;
   }

   // set the material stiffness on the model
   SetElementMatGrad(elemID, ipID, mGrad, ntens*ntens);

   // set the updated stress on the model. Have to convert from Abaqus 
   // ordering to Voigt notation ordering
   //------------------------------------------------------------------
   // We use Voigt notation: (11, 22, 33, 23, 13, 12)
   //
   // ABAQUS USES: 
   // (11, 22, 33, 12, 13, 23)
   //------------------------------------------------------------------
   stressTemp[0] = stress[0];
   stressTemp[1] = stress[1];
   stressTemp[2] = stress[2];
   stressTemp[3] = stress[5];
   stressTemp[4] = stress[4];
   stressTemp[5] = stress[3];

   SetElementStress(elemID, ipID, false, stressTemp, ntens);

   // set the updated statevars
   SetElementStateVars(elemID, ipID, false, statev, nstatv);

   CauchyToPK1();

   return;
}

void AbaqusUmatModel::AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                                const double weight, DenseMatrix &A)
{  
   // TODO get the material gradient off the quadrature vector function coeff.
   // Note: the Abaqus UMAT passes back 36 components in a 2D array of 
   // the symmetric fourth order tangent stiffness (of the Cauchy stress).
   // Figure out how to handle this in the easiest way.
   //
   int offset = 36;
   double matGrad[offset];

   GetElementMatGrad(elemID, ipID, matGrad, offset);

   // get the TOTAL beginning step deformation gradient stored on the model
   // as Jpt0
   GetElemDefGrad0(); 

   // get the TOTAL end-step (at a given NR iteration) deformation 
   // gradient (stored on the model as Jpt1)
   CalcElemDefGrad1(J); 

   // TODO finish implementing this routine
   // if (model->cauchy) then do chain rule. Neohookean give the PK1 
   // stress and the correct material tangent so no chain rule should 
   // be needed

   return;
}

void AbaqusUmatModel::CalcElemLength()
{
   // unclear how important a characteristic element length is to 
   // a UMAT for implicit mechanics. Just compute the largest 
   // euclidean distance between the first node and any other node
   int dim = Ttr->GetDimension();

   double len[dim]; 
   double maxLen = 0.0;
   double mag = 0.0;
   for (int i=1; i<dim; ++i)
   {
      len[0] = currElemCoords(i,0) - currElemCoords(0,0);
      len[1] = currElemCoords(i,1) - currElemCoords(0,1);
      len[2] = currElemCoords(i,2) - currElemCoords(0,2);

      mag = sqrt(len[0]*len[0] + len[1]*len[1] + len[2]*len[2]);

      if (mag > maxLen) maxLen = mag; 
   }

   elemLength = mag;

   return;
}

inline void NeoHookean::EvalCoeffs() const
{
   mu = c_mu->Eval(*Ttr, Ttr->GetIntPoint());
   K = c_K->Eval(*Ttr, Ttr->GetIntPoint());
   if (c_g)
   {
      g = c_g->Eval(*Ttr, Ttr->GetIntPoint());
   }
}

void NeoHookean::EvalModel(const DenseMatrix &J, const DenseMatrix &DS,
                           const double weight){
   EvalModel(J);
}
   
void NeoHookean::EvalModel(const DenseMatrix &J)
{
   // note: the J passed in is associated with the incremental 
   // nodal solution, so this is the incremental deformation 
   // gradient. 

   //printf("inside EvalModel \n");

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   // compute full end step PK1, old EvalP using the FULL end step 
   // deformation gradient, NOT the incremental one computed from the 
   // element transformation
   
   int dim = J.Width();

   Z.SetSize(dim);
   CalcAdjugateTranspose(J, Z);

   double dJ = J.Det();
   double a  = mu*pow(dJ, -2.0/dim);
   double b  = K*(dJ/g - 1.0)/g - a*(J*J)/(dim*dJ);

   //printf("dJ: %f \n", dJ);

   P.SetSize(dim);
   P = 0.0;
   P.Add(a, J);
   P.Add(b, Z);

   // debug print
//   for (int i=0; i<3; ++i)
//   {
//      for (int j=0; j<3; ++j)
//      {
//         printf("PK1: %f \n", P(i,j));
//      }
//   }
//   for (int i=0; i<3; ++i)
//   {
//      for (int j=0; j<3; ++j)
//      {
//         printf("J: %f \n", J(i,j));
//      }
//   }

   // convert the incremental PK1 stress to Cauchy stress. Note, we only store 
   // the 6 unique components of the symmetric Cauchy stress tensor.
   // 
   // RC, uncomment to transform Cauchy to PK1 (SRW)
   double sigma[6];
   for (int i=0; i<6; ++i) sigma[i] = 0.0;
   //printf("NeoHookean::EvalModel before PK1ToCauchy \n");
   PK1ToCauchy(P, J, sigma);

   // update total stress; We store the Cauchy stress, so we have to update 
   // that, and then we want to also carry around a local full PK1 stress 
   // that is used in the integrator, so we will have to convert back.
   //
   // RC, uncomment to access end step stress at current element/IP 
   // in order to update stress. Getting stress here, if printed to 
   // screen, gives screwy values. (SRW)
   //double sigma1[6];
   //for (int i=0; i<6; ++i) sigma1[i] = 0.0;   
   //printf("NeoHookean::EvalModel before GetElementStress \n");
   //   GetElementStress(elemID, ipID, false, sigma1, 6);
 
   // update stress
   //
   // RC, uncomment to print the stress from the previous call to 
   // GetElementStress (SRW)
   //printf("NeoHookean::EvalModel before stress update \n");
   //for (int i=0; i<6; ++i)
   //{
     //printf("sigma1 prior to resetting to new stress %f with %f\n", sigma1[i], sigma[i]);
   //   sigma1[i] = sigma[i];
   //}

   // set full updated stress on the quadrature function
   //printf("NeoHookean::EvalModel before SetElementStress \n");
//
//   RC, since the Getter didn't produce the correct initialized values of 
//   stress, then there is no point in testing the setter yet (SRW)
   SetElementStress(elemID, ipID, false, sigma, 6);

   /////////////////////////////////////////////////////////////////////////////
   // DEBUG CODE: remove later
   // place some dummy code here to test populating the material gradient 
   // quadrature function
//   printf("NeoHookean::EvalModel before dummy matGrad code \n");
//printf("elemID and ipID %d %d \n", elemID, ipID);
//   {
//      // set the material stiffness on the model. 
//      CMat.Clear();
//      CMat.SetSize(9);
//      CMat = 1.0;
//      double matGrd[12];
//
//      for (int i=0; i<12; ++i) matGrd[i] = 0.0;
//
////      for (int i=0; i<9; ++i)
////      {
////         for (int j=0; j<9; ++j)
////         {
////            matGrd[i*9+j] = CMat(i,j);
////         }
////      }
//
//      // TODO for 3D dof = 8 and dim = 3. This is assembling the full element 
//      // stiffness contribution, not just some material tangent modulus. Make 
//      // sure this is being used correctly with the new ExaModel.
//      printf("NeoHookean::EvalModel before SetElementMatGrad (elemID, ipID) %d %d \n", elemID, ipID);
////      printf("matGrad stride (dof*dim)*(dof*dim) %f\n", (dof*dim)*(dof*dim));
//      SetElementMatGrad(elemID, ipID, matGrd, 12);
//      printf("NeoHookean::EvalModel after SetElementMatGrad \n");
//
//   }
//   printf("NeoHookean::EvalModel after dummy matGrad code \n");

   return;
}

void NeoHookean::AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A)
{  
  //printf("NeoHookean::AssembleH before old AssembleH code \n");
   
   int dof = DS.Height(), dim = DS.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   double dJ = J.Det();
   double sJ = dJ/g;
   double a  = mu*pow(dJ, -2.0/dim);
   double bc = a*(J*J)/dim;
   double b  = bc - K*sJ*(sJ - 1.0);
   double c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);

   CalcAdjugateTranspose(J, Z);
   Z *= (1.0/dJ); // Z = J^{-t}

   MultABt(DS, J, C); // C = DS J^t
   MultABt(DS, Z, G); // G = DS J^{-1}

   a *= weight;
   b *= weight;
   c *= weight;

   // 1.
   for (int i = 0; i < dof; i++)
      for (int k = 0; k <= i; k++)
      {
         double s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += DS(i,d)*DS(k,d);
         }
         s *= a;

         for (int d = 0; d < dim; d++)
         {
            A(i+d*dof,k+d*dof) += s;
         }

         if (k != i)
            for (int d = 0; d < dim; d++)
            {
               A(k+d*dof,i+d*dof) += s;
            }
      }

   a *= (-2.0/dim);

   // 2.
   for (int i = 0; i < dof; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         for (int k = 0; k < dof; k++)
	 {
            for (int l = 0; l < dim; l++)
            {
               A(i+j*dof,k+l*dof) +=
                  a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                  b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
            }
         }
      }
   }// end all loops

   // debug print
//   for (int i=0; i<24; ++i)
//   {
//      for (int j=0; j<24; ++j)
//      {
//         printf("stf(i,j) %d %d %f \n", i, j, A(i,j));
//      }
//   }

   return;
}

// member functions for the ExaNLFIntegrator
double ExaNLFIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun)
{
   // we are not interested in the element energy at this time
   (void)el;
   (void)Ttr;
   (void)elfun;

   return 0.0;
}

void ExaNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   // this subroutine is modeled after HyperelasticNLFIntegretor version
   //printf("inside ExaNLFIntegrator::AssembleElementVector. \n");

   int dof = el.GetDof(), dim = el.GetDim(); 

   // these were previously stored on the hyperelastic NLF integrator;
   DenseMatrix DSh, DS;
   DenseMatrix Jrt, Jpr, Jpt; 
   DenseMatrix PMatI, PMatO;
   DenseMatrix *PMat;

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 1)); // must match quadrature space
   }

   elvect = 0.0;
   model->SetTransformation(Ttr);

   // set element id and attribute on model
   model->SetElementID(Ttr.ElementNo, Ttr.Attribute);

   // set the incremental nodal displacements on the model for 
   // the current element
   model->SetCoords(PMatI);

   //Get access to Pmat stored in the model
   PMat = model->GetPK1Stress();

   // get the timestep off the boundary condition manager. This isn't 
   // ideal, but in main(), the time step is just a local variable. 
   // Consider adding a simulation control class
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc = bcManager.CreateBCs(1);

   // set the time step on the model
   model->SetModelDt(bc.dt);

   // TODO fix a mismatch in integration points. This print statement 
   // shows a 3x3x3 Gauss rule, not a 2x2x2. Check where these are set
   // loop over integration points for current element
   //printf("AssembleElementVector GetNPoints() %d \n", ir->GetNPoints());
   for (int i=0; i<ir->GetNPoints(); i++)
   {
      // set integration point number on model
      model->SetIpID(i);

      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);

      // get integration point coordinates
      model->SetIPCoords(ip.x, ip.y, ip.z);

      // compute Jacobian of the transformation
      CalcInverse(Ttr.Jacobian(), Jrt); // Jrt = dxi / dX

      // compute Jpt, which is the incremental deformation gradient
      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX
      MultAtB(PMatI, DS, Jpt); // Jpt = F = dx/dX, PMatI = current config. coords.

      // get the beginning step deformation gradient. 
      model->GetElemDefGrad0(); 

      // get the TOTAL end-step (at a given NR iteration) deformation 
      // gradient
      model->CalcElemDefGrad1(Jpt); 

      // debug print
      /*for (int i=0; i<3; ++i)
      {
         for (int j=0; j<3; ++j)
         {
            printf("PMatI: %f \n", PMatI(i,j));
         }          
      }*/
      // get the stress update, set on the model (model->P)
      //printf("inside ExaNLFIntegrator::AssembleElementVector BEFORE EvalModel. \n");
      model->EvalModel(model->CopyOfJpt1(), DS, ip.weight * Ttr.Weight());
      //printf("inside ExaNLFIntegrator::AssembleElementVector AFTER EvalModel. \n");

      // multiply PK1 stress by integration point weight and 
      // determinant of the Jacobian of the transformation. Note that EvalModel
      // takes the weight input argument to conform to the old 
      // AssembleH where the weight was used in the NeoHookean model
      *PMat *= ip.weight * Ttr.Weight(); // Ttr.Weight is det() of Jacob. of 
                                            // transformation from ref. config. to 
                                            // parent space.
      AddMultABt(DS, *PMat, PMatO);

      // debug prints
      /*for (int i=0; i<3; ++i)
      {
         for (int j=0; j<3; ++j)
         {
	   printf("P after evalModel %f \n", (*PMatO)(i,j));
         }
      }*/
   }

   PMat = NULL;
   return;
}

void ExaNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, DenseMatrix &elmat)
{

   // printf("inside ExaNLFIntegrator::AssembleElementGrad. \n");
  
   // if debug is true on the model, then we are using the finite difference 
   // approximation to the element tangent stiffness contribution
   if (model->debug)
   {
     //  printf("inside ExaNLFIntegrator::AssembleElementGrad BEFORE FD routine. \n");
      AssembleElementGradFD(el, Ttr, elfun, elmat);
     //  printf("inside ExaNLFIntegrator::AssembleElementGrad AFTER FD routine. \n");
      return;
   }
   
   int dof = el.GetDof(), dim = el.GetDim();

   DenseMatrix DSh, DS, Jrt, Jpt, PMatI;
   
   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 1)); // <--- must match quadrature space
   }
  
   elmat = 0.0;
   model->SetTransformation(Ttr);
   model->SetElementID(Ttr.ElementNo, Ttr.Attribute); 

   // set the incremental nodal displacements on the model for 
   // the current element
   model->SetCoords(PMatI);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      model->SetIpID(i);

      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      // call the assembly routine. This may perform chain rule as necessary 
      // for a UMAT model
      //printf("inside ExaNLFIntegrator::AssembleElementGrad BEFORE AssembleH. \n");
      model->CalcElemDefGrad1(Jpt); 
      model->AssembleH(model->CopyOfJpt1(), DS, ip.weight * Ttr.Weight(), elmat);
      //      printf("inside ExaNLFIntegrator::AssembleElementGrad AFTER AssembleH. \n");
   }
   return;
}

void ExaNLFIntegrator::AssembleElementGradFD(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, DenseMatrix &elmat)
{

  //printf("inside ExaNLFIntegrator::AssembleElementGradFD. \n");

   double diff_step = 1.0e-8;
   Vector* temps;
   Vector* temp_out_1;
   Vector* temp_out_2;
   int dofs;

   temps = new Vector(elfun.GetData(), elfun.Size());
   temp_out_1 = new Vector();
   temp_out_2 = new Vector();
   dofs = elfun.Size();

   elmat.SetSize(dofs);

   for (int i=0; i<dofs; ++i)
   {
      temps[i] += diff_step;
      AssembleElementVector(el, Ttr, *temps, *temp_out_1);
      temps[i] -= 2.0*diff_step;
      AssembleElementVector(el, Ttr, *temps, *temp_out_2);
      for (int j=0; j<dofs; ++j)
      {
         elmat(j,i) = (temp_out_1[j] - temp_out_2[j]) / (2.0*diff_step);
      }
      temps[i] = elfun[i];
   }

   temps = NULL;
   temp_out_1 = NULL;
   temp_out_2 = NULL;

   return;
}

}
