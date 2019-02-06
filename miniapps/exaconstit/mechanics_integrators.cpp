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
#include <algorithm>
#include <iostream> // cerr
#include "userumat.h"

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

//We're assumming a 3D vector for this
void fixVectOrder(const Vector& v1, Vector &v2){

  int s1 = v1.Size()/3;
  int s2 = v2.Size()/3;
  //As noted in the original documentation this goes from a
  //[x0..xn, y0..yn] type ordering to [x0,y0, ..., xn,yn] type ordering
  for(int i = 0; i < s1; i++){
    int i1 = i * 3;
    v2[i1] = v1[i];
    v2[i1 + 1] = v1[i + s1];
    v2[i1 + 2] = v1[i + (s1 * 2)];
  }
  
}

void reorderVectOrder(const Vector &v1, Vector &v2){
  int s1 = v1.Size()/3;
  int s2 = v2.Size()/3;
  //As noted in the original documentation this goes from a
  //[x0,y0, ..., xn,yn] type ordering to [x0..xn, y0..yn] type ordering
  for(int i = 0; i < s1; i++){
    int i1 = i * 3;
    v2[i] = v1[i1];
    v2[i + s1] = v1[i1 + 1];
    v2[i + (2 * s1)] = v1[i1 + 2];
  }
}

void reorderMatrix(const DenseMatrix& a1, DenseMatrix& a2){

  int r = a1.Height()/3, c = a1.Width()/3;

  for(int i = 0; i < c * 3; i++){
    for(int j = 0; j < r; j++){
      int j1 = j * 3;
      a2(j, i) = a1(j1, i);
      a2(j + r, i) = a1(j1+1, i);
      a2(j + (2*r), i) = a1(j1+2, i);
    }
  }

  DenseMatrix a3(a2);
  
  for(int i = 0; i < c; i++){
    int i1 = i * 3;
    for(int j = 0; j < r * 3; j++){
      a2(j, i) = a3(j, i1);
      a2(j, i + c) = a3(j, i1+1);
      a2(j, i + (2*c)) = a3(j, i1+2);
    }
  }

  
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
   
   // check offset to input number of components
   if (qf_offset != numComps)
   {
      cerr << "\nGetElementStress: number of components does not match quad func offset" 
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<numComps; ++i)
   {
     stress[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
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

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<qf_offset; ++i)
   {
     int k = elID * elem_offset + ipNum * qf_offset + i;
     qf_data[k] = stress[i];
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
  
   double* mpdata = matProps->GetData();
   for(int i = 0; i < matProps->Size(); i++){
     props[i] = mpdata[i];
   }
   return;
}

void ExaModel::SetMatProps(double* props, int size)
{
   matProps->NewDataAndSize(props, size);
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

void ExaModel::UpdateEndCoords(const Vector& vel){

  int size;

  size = vel.Size();

  Vector end_crds(size);

  end_crds = 0.0;

  //tdofs sounds like it should hold the data points of interest, since the GetTrueDofs()
  //points to the underlying data in the GridFunction if all the TDofs lie on a processor
  Vector bcrds;
  bcrds.SetSize(size);
  //beg_coords is the beginning time step coordinates
  beg_coords -> GetTrueDofs(bcrds);
  int size2 = bcrds.Size();
  
  if (size != size2){
      mfem_error("TrueDofs and Vel Solution vector sizes are different");
  }
  
   //Perform a simple time integration to get our new end time step coordinates
  for (int i = 0; i < size; ++i){
    end_crds[i] = vel[i] * dt + bcrds[i];
  }

  //Now make sure the update gets sent to all the other processors that have ghost copies
  //of our data.
  end_coords->Distribute(end_crds);

  return;
}

void ExaModel::SwapMeshNodes(){

  int owns_nodes = 0;
  GridFunction *nodes;
  //We check to see if the end coordinates are currently in use.
  //If they are we swap the mesh nodes with the beggining time step nodes.
  //Then we update our EndCoordMesh flag to false to let us know this is the case.
  //If the EndCoordMesh flag was false we do the reverse of the above.
  if(EndCoordsMesh){
    nodes = beg_coords;
    pmesh->SwapNodes(nodes, owns_nodes);
    EndCoordsMesh = false;
  }else{
    nodes = end_coords;
    pmesh->SwapNodes(nodes, owns_nodes);
    EndCoordsMesh = true;
  }
  nodes = NULL;
  
}

void AbaqusUmatModel::UpdateModelVars( ParFiniteElementSpace *fes,
                              const Vector &x)
{
   // update the beginning step deformation gradient
   QuadratureFunction* defGrad = defGrad0.GetQuadFunction();
   if(EndCoordsMesh){
     SwapMeshNodes();
   }
   computeDefGrad(defGrad, fes, x);
   defGrad = NULL;
   //We just want to be super careful here in case the functions we call above
   //swap the mesh nodes without us realizing it. So, we want to swap them back to the
   //end nodes.
   if(!EndCoordsMesh){
     SwapMeshNodes();
   }
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

   double term4 = istress[3] * istress[3] + istress[4] * istress[4]
                  + istress[5] * istress[5];
   term4 *= 6.0;
                  
   double vm = sqrt(0.5 * (term1 + term2 + term3 + term4));

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
//rmat should be constant here...
void ExaModel::RMat2Quat(const DenseMatrix& rmat, Vector& quat){
   double inv2 = 1.0/2.0;
   double phi = 0.0;
   static const double eps = numeric_limits<double>::epsilon();
   double tr_r = 0.0;
   double inv_sin = 0.0;
   double s = 0.0;

   
   quat = 0.0;
   
   tr_r = rmat(0, 0) + rmat(1, 1) + rmat(2, 2);
   phi = inv2 * (tr_r - 1.0);
   phi = min(phi, 1.0);
   phi = max(phi, -1.0);
   phi = acos(phi);
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
void ExaModel::Quat2RMat(const Vector& quat, DenseMatrix& rmat){
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

   DenseMatrix omega_mat, temp;
   DenseMatrix def_grad(R, 3);
   
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

   quat = 0.0;
   
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
   
   //Now that we have the rotation portion of our deformation gradient
   //the left and right stretch tensors are easy to find.
   MultAtB(R, def_grad, U);
   MultABt(def_grad, R, V);
   
   return;
}

//This method calculates the Eulerian strain which is given as:
//e = 1/2 (I - B^(-1)) = 1/2 (I - F(^-T)F^(-1))
void ExaModel::CalcEulerianStrain(DenseMatrix& E){
   
   int dim = Ttr->GetDimension();

   DenseMatrix Finv(dim), Binv(dim);
   DenseMatrix F(Jpt1, dim);
   
   double half = 1.0/2.0;
   
   CalcInverse(F, Finv);
   
   MultAtB(Finv, Finv, Binv);
   
   E = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         E(i, j) -= half * Binv(i, j);
      }
      E(j, j) += half;
   }
   
   return;
}
   
//This method calculates the Lagrangian strain which is given as:
//E = 1/2 (C - I) = 1/2 (F^(T)F - I)
void ExaModel::CalcLagrangianStrain(DenseMatrix& E){
   
   int dim = Ttr->GetDimension();

   DenseMatrix F(Jpt1, dim);
   DenseMatrix C(dim);
   
   double half = 1.0/2.0;
   
   MultAtB(F, F, C);
   
   E = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         E(i, j) += half * C(i, j);
      }
      E(j, j) -= half;
   }
   
   return;
}
   
//This method calculates the Biot strain which is given as:
//E = (U - I) or sometimes seen as E = (V - I) if R = I
void ExaModel::CalcBiotStrain(DenseMatrix& E){

   int dim = Ttr->GetDimension();
  
   DenseMatrix rmat(Jpt1, dim);
   DenseMatrix umat, vmat;
   
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
   // UMAT uses the E = ln(V) approach instead

   DenseMatrix B, F;

   int dim = Ttr->GetDimension();

   B.SetSize(dim);
   F.SetSize(dim);

   F = Jpt1;

   MultABt(F, F, B);

   // compute eigenvalue decomposition of B
   double lambda[dim];
   double vec[dim*dim];
   //fix_me: was failing
   B.CalcEigenvalues(&lambda[0], &vec[0]);

   // compute ln(V) using spectral representation
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

void AbaqusUmatModel::CalcLogStrainIncrement(DenseMatrix& dE, const DenseMatrix &Jpt)
{
   // calculate incremental logorithmic strain (Hencky Strain) 
   // which is taken to be E = ln(U_hat) = 1/2 ln(C_hat), where 
   // C_hat = (F_hat_T)F_hat, where F_hat = Jpt1 on the model 
   // (available from MFEM element transformation computations). 
   // We can compute F_hat, so use a spectral decomposition on C_hat to 
   // obtain a form where we only have to take the natural log of the 
   // eigenvalues
   // UMAT uses the E = ln(V) approach instead

   DenseMatrix F_hat, B_hat;

   int dim = Ttr->GetDimension();

   F_hat.SetSize(dim);
   B_hat.SetSize(dim); 

   F_hat = Jpt;

   MultABt(F_hat, F_hat, B_hat);

   // compute eigenvalue decomposition of B
   double lambda[dim];
   double vec[dim*dim];
   B_hat.CalcEigenvalues(&lambda[0], &vec[0]);

   // compute ln(B) using spectral representation
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
void AbaqusUmatModel::CalcEulerianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt){

   int dim = Ttr->GetDimension();
   DenseMatrix Fincr(Jpt, dim);
   DenseMatrix Finv(dim), Binv(dim);
   
   double half = 1.0/2.0;
   
   CalcInverse(Fincr, Finv);
   
   MultAtB(Finv, Finv, Binv);
   
   dE = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         dE(i, j) -= half * Binv(i, j);
      }
      dE(j, j) += half;
   }
}

//This method calculates the Lagrangian strain which is given as:
//E = 1/2 (C - I) = 1/2 (F^(T)F - I)
void AbaqusUmatModel::CalcLagrangianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt){
   
   DenseMatrix C;
   
   int dim = Ttr->GetDimension();
   
   double half = 1.0/2.0;
   
   C.SetSize(dim);
   
   MultAtB(Jpt, Jpt, C);
   
   dE = 0.0;
   
   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         dE(i, j) += half * C(i, j);
      }
      dE(j, j) -= half;
   }
   return;
}

//This is based on Hughes Winget 1980 paper https://doi.org/10.1002/nme.1620151210
//more specifically equations 26 for the rotation. The strain increment is just
//dE = D * dt where D is the deformation rate
void AbaqusUmatModel::CalcIncrStrainRot(double* dE, DenseMatrix& dRot){

  DenseMatrix hskw_v(3);
  DenseMatrix a(3);
  DenseMatrix b(3);
  DenseMatrix Vgt(3);

  Vgt.Transpose(Vgrad);
  
  double qdt = 0.25 * dt;
  
  dE[0] = Vgrad(0,0) * dt;
  dE[1] = Vgrad(1,1) * dt;
  dE[2] = Vgrad(2,2) * dt;
  dE[3] = (Vgrad(0,1) + Vgrad(1,0)) * dt;
  dE[4] = (Vgrad(0,2) + Vgrad(2,0)) * dt;
  dE[5] = (Vgrad(1,2) + Vgrad(2,1)) * dt;

  Add(qdt, Vgrad.GetData(), -qdt, Vgt.GetData(), hskw_v);

  a = 0.0;
  b = 0.0;
  a(0,0) = 1.0;
  a(1,1) = 1.0;
  a(2,2) = 1.0;
  Add(1.0, a.GetData(), -1.0, hskw_v.GetData(), a);

  a.Invert();
  
  b(0,0) = 1.0;
  b(1,1) = 1.0;
  b(2,2) = 1.0;

  Add(b, hskw_v, 1.0, b);

  Mult(a, b, dRot);
  
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
   double tempk       = 300.0;   // no thermal considered at this point
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
   // advancing the sub-increment. For now, set time[0] is set to t - dt/
   double time[2];
   time[0] = t - dt;
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

   double *drot;   // rotation matrix for finite deformations
   double dfgrd0[9]; // deformation gradient at beginning of increment
   double dfgrd1[9]; // defomration gradient at the end of the increment.
                        // set to zero if nonlinear geometric effects are not 
                        // included in the step as is the case for ExaConstit
   
   DenseMatrix Uincr(3), Vincr(3);
   DenseMatrix Rincr(J, 3);
   CalcPolarDecompDefGrad(Rincr, Uincr, Vincr);
   
   drot = Rincr.GetData();

   // populate the beginning step and end step (or best guess to end step 
   // within the Newton iterations) of the deformation gradients
   for (int i=0; i<ndi; ++i)
   {
      for (int j=0; j<ndi; ++j)
      {
         //Dense matrices have column major layout so the below is fine.
         dfgrd0[(i * 3) + j] = Jpt0(j,i);
         dfgrd1[(i * 3) + j] = Jpt1(j,i);
      }
   }

   // get state variables and material properties
   GetElementStateVars(elemID, ipID, true, statev, nstatv);
   GetMatProps(props);

   // get element stress and make sure ordering is ok
   double stressTemp[6];
   double stressTemp2[6];
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
   stran[3] = 2 * LogStrain(0,1);
   stran[4] = 2 * LogStrain(0,2);
   stran[5] = 2 * LogStrain(1,2);

   // compute incremental strain, DSTRAN
   DenseMatrix dLogStrain;
   dLogStrain.SetSize(ndi);
   CalcEulerianStrainIncr(dLogStrain, J);

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
   dstran[3] = 2 * dLogStrain(0,1);
   dstran[4] = 2 * dLogStrain(0,2);
   dstran[5] = 2 * dLogStrain(1,2);
   
   
   // call c++ wrapper of umat routine
   umat(&stress[0], &statev[0], &ddsdde[0], &sse, &spd, &scd, &rpl,
        ddsdt, drplde, &drpldt, &stran[0], &dstran[0], time,
        &deltaTime, &tempk, &dtemp, &predef,&dpred, &cmname,
        &ndi, &nshr, &ntens, &nstatv, &props[0], &nprops, &coords[0],
        drot, &pnewdt, &celent, &dfgrd0[0], &dfgrd1[0], &noel, &npt,
        &layer, &kspt, &kstep, &kinc);

   // restore the material Jacobian in a 1D array
   double mGrad[36];

   for (int i=0; i<6; ++i)
   {
     for (int j=0; j<6; ++j)
     {
       // column-wise ordering of material Jacobian
       mGrad[(6 * i) + j] = ddsdde[(i * 6) + j];
     }
   }
   
   //Due to how Abaqus has things ordered we need to swap the 4th and 6th columns
   //and rows with one another for our C_stiffness matrix.
   double temp=0;
   int j = 3;
   for(int i=0; i<6; i++){
     temp = mGrad[(6*i) + j];
     mGrad[(6*i) + j] = mGrad[(6*i) + 5];
     mGrad[(6*i) + 5] = temp;
   }
   for(int i=0; i<6; i++){
     temp = mGrad[(6*j) + i];
     mGrad[(6*j) + i] = mGrad[(6*5) + i];
     mGrad[(6*5) + i] = temp;
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
   stressTemp2[0] = stress[0];
   stressTemp2[1] = stress[1];
   stressTemp2[2] = stress[2];
   stressTemp2[3] = stress[5];
   stressTemp2[4] = stress[4];
   stressTemp2[5] = stress[3];
   
   SetElementStress(elemID, ipID, false, stressTemp2, ntens);

   // set the updated statevars
   SetElementStateVars(elemID, ipID, false, statev, nstatv);
   //Could probably later have this only set once...
   //Would reduce the number mallocs that we're doing and
   //should potentially provide a small speed boost.
   P.SetSize(3);
   P(0,0) = stressTemp2[0];
   P(1,1) = stressTemp2[1];
   P(2,2) = stressTemp2[2];
   P(1,2) = stressTemp2[3];
   P(0,2) = stressTemp2[4];
   P(0,1) = stressTemp2[5];

   P(2,1) = P(1,2);
   P(2,0) = P(0,2);
   P(1,0) = P(0,1);

   int dof = DS.Height(), dim = DS.Width();

   //The below is letting us just do: Int_{body} B^t sigma dV
   DenseMatrix PMatO, DSt(DS);
   DSt *= weight;

   PMatO.UseExternalData(elresid->GetData(), dof, dim);
   
   AddMult(DSt,P,PMatO);
   
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

   int dof = DS.Height(), dim = DS.Width();
   
   GetElementMatGrad(elemID, ipID, matGrad, offset);

   DenseMatrix Cstiff(matGrad, 6, 6);

   //Now time to start assembling stuff
   
   DenseMatrix temp1, kgeom;
   DenseMatrix sbar;
   DenseMatrix BtsigB;
   
   //We'll first be using the above variable for our geomtric contribution
   //
   //The geometric contribution is currently commented out after not really
   //seeing it being used in a few other libraries. It can just as easily be added
   //back if it is deemed desirable in later simulation cases.
   /*
   temp1.SetSize(dim*dim, dof*dim);
   
   kgeom.SetSize(dof*dim, dim*dim);

   sbar.SetSize(dim*dim);

   BtsigB.SetSize(dof*dim);
   
   for(int i = 0; i < dim; i++){
     int i1 = i * dim;
     int j1 = i * dim;
     sbar(i1, j1) = P(0, 0);
     sbar(i1, j1 + 1) = P(0, 1);
     sbar(i1, j1 + 2) = P(0, 2);

     sbar(i1 + 1, j1) = P(1, 0);
     sbar(i1 + 1, j1 + 1) = P(1, 1);
     sbar(i1 + 1, j1 + 2) = P(1, 2);

     sbar(i1 + 2, j1) = P(2, 0);
     sbar(i1 + 2, j1 + 1) = P(2, 1);
     sbar(i1 + 2, j1 + 2) = P(2, 2);
   }

   sbar *= weight;
   
   GenerateGradGeomMatrix(DS, kgeom);
   
   MultABt(sbar, kgeom, temp1);
   AddMult(kgeom, temp1, A);
   */
   
   //temp1 is now going to become the transpose Bmatrix as seen in
   //[B^t][Cstiff][B]
   temp1.SetSize(dof*dim, 6);
   //We need a temp matrix to store our first matrix results as seen in here
   kgeom.SetSize(6, dof*dim);
   //temp1 is B^t as seen above
   GenerateGradMatrix(DS, temp1);
   //We multiple our quadrature wts here to our Cstiff matrix
   Cstiff *= dt * weight;
   //We use kgeom as a temporary matrix
   //kgeom = [Cstiff][B]
   MultABt(Cstiff, temp1, kgeom);
   //We now add our [B^t][kgeom] product to our tangent stiffness matrix that
   //we want to output to our material tangent stiffness matrix
   AddMult(temp1, kgeom, A);

   //A.Symmetrize();
   //A *= dt;

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

//This function is used in generating the B matrix commonly seen in the formation of 
//the material tangent stiffness matrix in mechanics [B^t][Cstiff][B]
//Although we're goint to return really B^t here since it better matches up
//with how DS is set up memory wise
//The B matrix should have dimensions equal to (dof*dim, 6).
//We assume it hasn't been initialized ahead of time or it's already
//been written in, so we rewrite over everything in the below.
void ExaModel::GenerateGradMatrix(const DenseMatrix& DS, DenseMatrix& B){

   int dof = DS.Height(), dim = DS.Width();
   

   //The B matrix generally has the following structure that is
   //repeated for the number of dofs if we're dealing with something
   //that results in a symmetric Cstiff. If we aren't then it's a different
   //structure
   //[DS(i,0) 0 0]
   //[0 DS(i, 1) 0]
   //[0 0 DS(i, 2)]
   //[0 DS(i,2) DS(i,1)]
   //[DS(i,2) 0 DS(i,0)]
   //[DS(i,1) DS(i,0) 0]
   
   //Just going to go ahead and make the assumption that
   //this is for a 3D space. Should put either an assert
   //or an error here if it isn't
   //We should also put an assert if B doesn't have dimensions of
   //(dim*dof, 6)
   //fix_me
   int i1;
   //We've rolled out the above B matrix in the comments
   //This is definitely not the most efficient way of doing this memory wise.
   //However, it might be fine for our needs.
   //The ordering has now changed such that B matches up with mfem's internal
   //ordering of vectors such that it's [x0...xn, y0...yn, z0...zn] ordering
   
   //The previous single loop has been split into 3 so the B matrix
   //is constructed in chunks now instead of performing multiple striding
   //operations in a single loop.
   //x dofs
   for (int i = 0; i < dof; i++){
     B(i, 0) = DS(i, 0);
     B(i, 1) = 0.0;
     B(i, 2) = 0.0;
     B(i, 3) = 0.0;
     B(i, 4) = DS(i, 2);
     B(i, 5) = DS(i, 1);
   }
   //y dofs
   for (int i = 0; i < dof; i++){
     B(i + dof, 0) = 0.0;
     B(i + dof, 1) = DS(i, 1);
     B(i + dof, 2) = 0.0;
     B(i + dof, 3) = DS(i, 2);
     B(i + dof, 4) = 0.0;
     B(i + dof, 5) = DS(i, 0);
   }
   //z dofs
   for (int i = 0; i < dof; i++){
     B(i + 2*dof, 0) = 0.0;
     B(i + 2*dof, 1) = 0.0;
     B(i + 2*dof, 2) = DS(i, 2);
     B(i + 2*dof, 3) = DS(i, 1);
     B(i + 2*dof, 4) = DS(i, 0);
     B(i + 2*dof, 5) = 0.0;
   }

   return;
}

void ExaModel::GenerateGradGeomMatrix(const DenseMatrix& DS, DenseMatrix& Bgeom){

  int dof = DS.Height(), dim = DS.Width();
  //For a 3D mesh Bgeom has the following shape:
  //[DS(i, 0), 0, 0]
  //[DS(i, 0), 0, 0]
  //[DS(i, 0), 0, 0]
  //[0, DS(i, 1), 0]
  //[0, DS(i, 1), 0]
  //[0, DS(i, 1), 0]
  //[0, 0, DS(i, 2)]
  //[0, 0, DS(i, 2)]
  //[0, 0, DS(i, 2)]
  //We'll be returning the transpose of this.
  //It turns out the Bilinear operator can't have this created using
  //the dense gradient matrix, DS.
  //It can be used in the following: Bgeom^T Sigma_bar Bgeom
  //where Sigma_bar is a block diagonal version of sigma repeated 3 times in 3D.

  //I'm assumming we're in 3D and have just unrolled the loop
  //The ordering has now changed such that Bgeom matches up with mfem's internal
  //ordering of vectors such that it's [x0...xn, y0...yn, z0...zn] ordering
  
  //The previous single loop has been split into 3 so the B matrix
  //is constructed in chunks now instead of performing multiple striding
  //operations in a single loop.
  
  //x dofs
  for (int i = 0; i < dof; i++){
    Bgeom(i, 0) = DS(i, 0);
    Bgeom(i, 1) = DS(i, 1);
    Bgeom(i, 2) = DS(i, 2);
    Bgeom(i, 3) = 0.0;
    Bgeom(i, 4) = 0.0;
    Bgeom(i, 5) = 0.0;
    Bgeom(i, 6) = 0.0;
    Bgeom(i, 7) = 0.0;
    Bgeom(i, 8) = 0.0;
  }
  //y dofs
  for (int i = 0; i < dof; i++){
    Bgeom(i + dof, 0) = 0.0;
    Bgeom(i + dof, 1) = 0.0;
    Bgeom(i + dof, 2) = 0.0;
    Bgeom(i + dof, 3) = DS(i, 0);
    Bgeom(i + dof, 4) = DS(i, 1);
    Bgeom(i + dof, 5) = DS(i, 2);
    Bgeom(i + dof, 6) = 0.0;
    Bgeom(i + dof, 7) = 0.0;
    Bgeom(i + dof, 8) = 0.0;
  }
  //z dofs
  for (int i = 0; i < dof; i++){
    Bgeom(i + 2*dof, 0) = 0.0;
    Bgeom(i + 2*dof, 1) = 0.0;
    Bgeom(i + 2*dof, 2) = 0.0;
    Bgeom(i + 2*dof, 3) = 0.0;
    Bgeom(i + 2*dof, 4) = 0.0;
    Bgeom(i + 2*dof, 5) = 0.0;
    Bgeom(i + 2*dof, 6) = DS(i, 0);
    Bgeom(i + 2*dof, 7) = DS(i, 1);
    Bgeom(i + 2*dof, 8) = DS(i, 2);
  }
  
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

//Outside of the UMAT function calls this should be the function called
//to assemble our residual vectors.
void ExaNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim(); 

   DenseMatrix DSh, DS;
   DenseMatrix Jrt, Jpr, Jpt; 
   DenseMatrix PMatI, PMatO;
   DenseMatrix *PMat;

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   //PMatI would be our velocity in this case
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   //PMatO would be our residual vector
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

   //We want to use the Cauchy stress here
   //Get access to Pmat stored in the model
   PMat = model->GetCauchyStress();

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

      // Jpt here would be our velocity gradient
      // get the stress update, set on the model (model->P)
      model->EvalModel(Jpt, DS, ip.weight * Ttr.Weight());

      // multiply Cauchy stress by integration point weight and
      // determinant of the Jacobian of the transformation. Note that EvalModel
      // takes the weight input argument to conform to the old 
      // AssembleH where the weight was used in the NeoHookean model
      *PMat *= ip.weight * Ttr.Weight(); // Ttr.Weight is det() of Jacob. of 
                                            // transformation from ref. config. to 
                                            // parent space.
      AddMult(DS, *PMat, PMatO);
   }

   PMat = NULL;
   return;
}

// This function should pretty much be used for our UMAT interface
// Ttr_beg allows us to get the incremental deformation gradient
// so our shape functions are based on our beginning time step mesh
// Ttr_end allows us to get the velocity gradient
// so our shape functions are based on our end time step mesh
// elfun is our end time step mesh coordinates
// elvect is our residual that we're trying to drive to 0
// elvel is our end time step velocity field
void ExaNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Ttr_beg,
   ElementTransformation &Ttr_end,
   const Vector &elfun, Vector &elvect,
   const Vector &elvel)
{

   int dof = el.GetDof(), dim = el.GetDim(); 

   int elnum = Ttr_beg.ElementNo;
   
   // these were previously stored on the hyperelastic NLF integrator;
   DenseMatrix DSh, DS, DS_diff;
   DenseMatrix Jrt, Jpr, Jpt; 
   DenseMatrix PMatI, PMatO;
   DenseMatrix *PMat;
   
   Vector elv(dim*dof);
   
   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   DS_diff.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   /*printf("\n Coordinates: \n");
   for(int i = 0; i<dof; i++){
     for(int j = 0; j<dim; j++){
       printf("%lf ", PMatI(i,j));
     }
     printf("\n");
   }*/
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dim, dof);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 1)); // must match quadrature space
   }

   elvect = 0.0;
   
   //set element id and attribute on model
   model->SetElementID(Ttr_beg.ElementNo, Ttr_beg.Attribute);
   model->SetTransformation(Ttr_beg);
   // set the incremental nodal displacements on the model for 
   // the current element
   model->SetCoords(PMatI);
   //Our velocity, residual, and coords are passed as reference here
   model->SetVel(&elvel);
   model->SetResid(&elvect);
   model->SetCrds(&elfun);

   //Get access to Pmat stored in the model
   PMat = model->GetCauchyStress();
   //Vgrad = model->GetVGrad();

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
   for (int i=0; i<ir->GetNPoints(); i++)
   {
      // set integration point number on model
      model->SetIpID(i);
      const IntegrationPoint &ip = ir->IntPoint(i);

      //We first need to obtain our current time steps deformation gradient
      Ttr_beg.SetIntPoint(&ip);
      model->SetIPCoords(ip.x, ip.y, ip.z);
      
      // compute Jacobian of the transformation
      CalcInverse(Ttr_beg.Jacobian(), Jrt); // Jrt = dxi / dX
      // compute Jpt, which is the incremental deformation gradient
      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX
      //Ds here is in terms of our beginning time step coordinates
      MultAtB(PMatI, DS, Jpt); // Jpt = F = dx/dX, PMatI = current config. coords.
      // get the beginning step deformation gradient. 
      model->GetElemDefGrad0();
      // get the TOTAL end-step (at a given NR iteration) deformation 
      // gradient
      model->CalcElemDefGrad1(Jpt);
      Ttr_beg.SetIntPoint(NULL);
      
      //We now need to compute our shape functions in terms of our current configuration
      Ttr_end.SetIntPoint(&ip);
      // compute Jacobian of the transformation using the current configuration information
      CalcInverse(Ttr_end.Jacobian(), Jrt); // Jrt = dxi / dX
      // compute Jpt, which is the incremental deformation gradient
      el.CalcDShape(ip, DSh);
      //Ds is now in terms of the current configuration
      Mult(DSh, Jrt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX
      
      //Inside of this function our residual is updated already
      model->EvalModel(Jpt, DS, ip.weight * Ttr_end.Weight());

      Ttr_end.SetIntPoint(NULL);
   }

   PMat = NULL;
   return;
}

void ExaNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, DenseMatrix &elmat)
{
   // Pass in an extra vector called crdfun it will be used to calculate the
   // deformation gradient. We'll need to update the nonlinear operator
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
   Jpt = 0.0;
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

      // call the assembly routine. This may perform chain rule as necessary 
      // for a UMAT model
      model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
   }
   /*printf("\n Stiffness Matrix \n");
   for(int i = 0; i<dof*dim; i++){
     for(int j = 0; j<dof*dim; j++){
       printf("%lf ", elmat(i, j));
     }
     printf("\n");
     }*/
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
