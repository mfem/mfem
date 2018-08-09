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

namespace mfem
{

// TODO have to call GetQuadFunction on the QuadVectorFunctionCoefficient

// member functions for the Abaqus Umat base class
int ExaModel::GetStressOffset()
{
   QuadratureFunction* qf = stress0.GetQuadFunction();
   int qf_offset = qf->GetVDim();
   return qf_offset;
}

int ExaModel::GetMatGradOffset()
{
   QuadratureFunction* qf = matGrad.GetQuadFunction();
   int qf_offset = qf->GetVDim();
   return qf_offset;
}

int ExaModel::GetMatVarsOffset()
{
   QuadratureFunction* qf = matVars0.GetQuadFunction();
   int qf_offset = qf->GetVDim();
   return qf_offset;
}

// the getter simply returns the beginning step stress
void ExaModel::GetElementStress(const int elID, const int ipNum,
                                bool beginStep, double* stress, int numComps)
{
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
      cerr << "\nGetElementStress: number of components does not match quad func offset" 
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i=0; i<numComps; ++i)
   {
     stress[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
   }

   return;
}

void ExaModel::SetElementStress(const int elID, const int ipNum, 
                                bool beginStep, double* stress, int numComps)
{
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
     qf_data[elID * elem_offset + ipNum * qf_offset + i] = stress[i];
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

   return;
}

void ExaModel::SetElementMatGrad(const int elID, const int ipNum, double* grad)
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
     qf_data[elID * elem_offset + ipNum * qf_offset + i] = grad[i];
   }
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

   // get finite element object off the element transformation 
   // on the ExaModel base class to get the dimension 
   // of that element.
   FiniteElement* fe =  Ttr->GetFE();
   int dim = fe->GetDim();

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
         Jpt0(m,n) = 
           qf_data[elemID * elem_offset + ipID * qf_offset + k];
         ++k;
      }
   }

   return;
}
void ExaModel::SetElemDefGrad1(DenseMatrix& Jpt)
{
   FiniteElement* fe = Ttr->GetFE();
   int dim = fe->GetDim();
   
   Jpt1.Clear();
   Jpt1.SetSize(dim);
   Jpt1 = 0.0;

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
   return;
}

void ExaModel::UpdateModelVars(ParFiniteElementSpace *fes, 
                                       const Vector &x)
{
// update the beginning step deformation gradient, which is set to the current, 
// converged endstep deformation gradient
   QuadratureFunction* defGrad = defGrad0.GetQuadFunction();
   computeDefGrad(defGrad, fes, x);
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

void AbaqusUmatModel::CalcLogStrain(DenseMatrix& E)
{
   // calculate current end step logorithmic strain (Hencky Strain) 
   // which is taken to be E = ln(U) = 1/2 ln(C), where C = (F_T)F. 
   // We have incremental F from MFEM, and store F0 (Jpt0) so 
   // F = F_hat*F0. With F, use a spectral decomposition on C to obtain a 
   // form where we only have to take the natural log of the 
   // eigenvalues

   DenseMatrix C, F;

   FiniteElement* fe = Ttr->GetFE();
   int dim = fe->GetDim();

   C.SetSize(dim); 

   Mult(Jpt1, Jpt0, F);

   MultAtB(F, F, C);

   // compute eigenvalue decomposition of C
   double lambda[dim];
   double vec[dim];
   C.CalcEigenValues(lambda, vec);

   // compute ln(C) using spectral representation
   E = 0.0;
   for (int i=0; i<dim; ++i) // outer loop for every eigenvalue/vector
   {
      for (int j=0; j<dim; ++j) // inner loops for diadic product of eigenvectors
      {
         for (int k=0; k<dim; ++k)
         {
            E(j,k) += 0.5 * log(lambda[i]) * vec[i*dim+j] * vec[i*dim+k];
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

   FiniteElement* fe = Ttr->GetFE();
   int dim = fe->GetDim();

   F_hat.SetSize(dim);
   C_hat.SetSize(dim); 

   F_hat = Jpt1;

   MultAtB(F_hat, F_hat, C_hat);

   // compute eigenvalue decomposition of C
   double lambda[dim];
   double vec[dim];
   C_hat.CalcEigenValues(lambda, vec);

   // compute ln(C) using spectral representation
   dE = 0.0;
   for (int i=0; i<dim; ++i) // outer loop for every eigenvalue/vector
   {
      for (int j=0; j<dim; ++j) // inner loops for diadic product of eigenvectors
      {
         for (int k=0; k<dim; ++k)
         {
            dE(j,k) += 0.5 * log(lambda[i]) * vec[i*dim+j] * vec[i*dim+k];
         }
      }
   }

   return;
}

// NOTE: this UMAT interface is for use only in ExaConstit and considers 
// only mechanical analysis. There are no thermal effects. Any thermal or 
// thermo-mechanical coupling variables for UMAT input are null.
void AbaqusUmatModel::EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                                const double weight)
{
   // get the beginning step deformation gradient. Note: even though 
   // F0 = I in the incremental form, a UMAT expects beginning step 
   // TOTAL deformation gradient.
   GetElemDefGrad0(); 

   // get the TOTAL end-step (at a given NR iteration) deformation 
   // gradient
   SetElemDefGrad1(Jpt); 

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
   double temp       = 0.0;   // no thermal considered at this point
   double dtemp      = 0.0;   // no increment in thermal considered at this point
   double predef[1]  = {0.0}; // no interpolated values of predefined field variables at ip point
   double dpred[1]   = {0.0}; // no array of increments of predefined field variables
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
   time[0] = 0.0;
   time[1] = time; 

   double stress[6]; // Cauchy stress at ip 
   double ddsddt[6]; // variation of the stress increments wrt to temperature, set to 0.0
   double drplde[6]; // variation of rpl wrt strain increments, set to 0.0
   double stran[6];  // array containing total strains at beginning of the increment
   double dstran[6]; // array of strain increments

   // initialize 1d arrays
   for (int i=0; i<6; ++i) {
      stress[i] = 0.0;
      ddsddt[i] = 0.0;
      drplde[i] = 0.0;
      stran[i]  = 0.0;
      dstran[i] = 0.0;
   } 

   double ddsdde[6][6]; // output Jacobian matrix of the constitutive model.
                        // ddsdde(i,j) defines the change in the ith stress component 
                        // due to an incremental perturbation in the jth strain increment

   // initialize 6x6 2d arrays
   for (int i=0; i<6; ++i) {
      for (int j=0; j<6; ++j) {
         ddsdde[i][j] = 0.0;
      }
   }

   double drot[3][3];   // rotation matrix for finite deformations
   double dfgrd0[3][3]; // deformation gradient at beginning of increment
   double dfgrd1[3][3]; // defomration gradient at the end of the increment.
                        // set to zero if nonlinear geometric effects are not 
                        // included in the step as is the case for ExaConstit
   
   // initialize 3x3 2d arrays to identity
   for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
         drot[i][j]   = 0.0;
         dfgrd0[i][j] = 0.0;
         dfgrd1[i][j] = 0.0;
   
         if (i == j)
         {
            drot[i][j]   = 1.0;
            dfgrd0[i][j] = 1.0;
            dfgrd1[i][j] = 1.0;
         }
      }
   }

   // populate the beginning step and end step (or best guess to end step 
   // within the Newton iterations) of the deformation gradients
   for (int i=0; i<ndi; ++i)
   {
      for (int j=0; j<ndi; ++j)
      {
         dfgrd0[i][j] = Jpt0(i,j);
         dfgrd1[i][j] = Jpt1(i,j);
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

   // compute the logorithmic strain
   DenseMatrix LogStrain;
   LogStrain.SetSize(ndi); // ndi x ndi

   CalcLogStrain(LogStrain);

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

   // call fortran umat routine
   umat(stress, statev, ddsdde, sse, spd, scd, rpl, 
        ddsddt, drplde, drpldt, stran, dstran, time, deltaTime,
        temp, dtemp, predef, dpred, cmname, &ndi, &nshr, &ntens,
        &nstatv, props, &nprops, coords, drot, &pnewdt, celent,
        dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc);

   // restore the material Jacobian in a 1D array
   double mGrad[36];

   for (int i=0; i<6; ++i)
   {
     for (int j=0; j<6; ++j)
     {
       // row-wise ordering of material Jacobian
       mGrad[6*i+j] = ddsdde[i][j];
     }
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

void AbaqusUmatModel::CalcElemLength()
{
   // unclear how important a characteristic element length is to 
   // a UMAT for implicit mechanics. Just compute the largest 
   // euclidean distance between the first node and any other node
   FiniteElement* fe =  Ttr->GetFE();
   int dof = fe->GetDof();
   int dim = fe->GetDim();

   double len[dim]; 
   double maxLen = 0.0;
   double mag = 0.0;
   for (int i=1; i<dof; ++i)
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

void AbaqusUmatModel::CauchyToPK1()
{
   double det;
   DenseMatrix FinvT;;

   // compute the determinant of the end step deformation gradient
   det = Jpt1.Det();

   int size = Jpt1.Size();

   FinvT.SetSize(size);
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
   Mult(Cauchy, FinvT, P)
   P *= det;

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
                           const double weight);
{
   // note: the J passed in is associated with the incremental 
   // nodal solution, so this is the incremental deformation 
   // gradient. We have to convert this to the full end step 
   // deformation gradient

   // get the beginning step deformation gradient. Note: even though 
   // F0 = I in the incremental form, a UMAT expects beginning step 
   // TOTAL deformation gradient.
   GetElemDefGrad0(); 

   // get the TOTAL end-step (at a given NR iteration) deformation 
   // gradient
   SetElemDefGrad1(J); 

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   // compute PK1, old EvalP
   {
      int dim = Jpt1.Width();

      Z.SetSize(dim);
      CalcAdjugateTranspose(Jpt1, Z);

      double dJ = Jpt1.Det();
      double a  = mu*pow(dJ, -2.0/dim);
      double b  = K*(dJ/g - 1.0)/g - a*(Jpt1*Jpt1)/(dim*dJ);

      P = 0.0;
      P.Add(a, Jpt1);
      P.Add(b, Z);
   }

   // get material Jacobian, old AssembleH
   {
      int dof = DS.Height(), dim = DS.Width();

      Z.Clear();

      CMat.SetSize(dof*dim); // stored on the model
      Z.SetSize(dim);
      G.SetSize(dof, dim);
      C.SetSize(dof, dim);

      CMat = 0.0;

      double dJ = Jpt1.Det();
      double sJ = dJ/g;
      double a  = mu*pow(dJ, -2.0/dim);
      double bc = a*(Jpt1*Jpt1)/dim;
      double b  = bc - K*sJ*(sJ - 1.0);
      double c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);

      CalcAdjugateTranspose(Jpt1, Z);
      Z *= (1.0/dJ); // Z = Jpt1^{-t}

      MultABt(DS, Jpt1, C); // C = DS Jpt1^t
      MultABt(DS, Z, G); // G = DS Jpt1^{-1}

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
               CMat(i+d*dof,k+d*dof) += s;
            }

            if (k != i)
               for (int d = 0; d < dim; d++)
               {
                  CMat(k+d*dof,i+d*dof) += s;
               }
         }

      a *= (-2.0/dim);

      // 2.
      for (int i = 0; i < dof; i++)
         for (int j = 0; j < dim; j++)
            for (int k = 0; k < dof; k++)
               for (int l = 0; l < dim; l++)
               {
                  CMat(i+j*dof,k+l*dof) +=
                     a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                     b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
               }

      // TODO store Cmat on the quadrature function, matGrad 
      // on the model
   }

   return;
}

double NeoHookean::EvalW(const DenseMatrix &J) const
{
   // get the beginning step deformation gradient. Note: even though 
   // F0 = I in the incremental form, a UMAT expects beginning step 
   // TOTAL deformation gradient.
   GetElemDefGrad0(); 

   // get the TOTAL end-step (at a given NR iteration) deformation 
   // gradient
   SetElemDefGrad1(J); 
   
   int dim = Jpt1.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   double dJ = Jpt1.Det();
   double sJ = dJ/g;
   double bI1 = pow(dJ, -2.0/dim)*(Jpt1*Jpt1); // \bar{I}_1

   return 0.5*(mu*(bI1 - dim) + K*(sJ - 1.0)*(sJ - 1.0));
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

   int dof = el.GetDof(), dim = el.GetDim(); 

   // these were stored on the hyperelastic NLF integrator;
   DenseMatrix DSh, DS;
   DenseMatrix Jrt, Jpr, Jpt; 
   DenseMatrix PMatI, PMatO;

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
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3));
   }

   elvect = 0.0;
   model->SetTransformation(Ttr);

   // set element id and attribute on model
   model->SetElementID(Ttr.ElementNo, Ttr.Attribute);

   // set the incremental nodal displacements on the model for 
   // the current element
   model->SetCoords(PMatI);

   // get the timestep off the boundary condition manager. This isn't 
   // ideal, but in main(), the time step is just a local variable. 
   // Consider adding a simulation control class
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc = bcManager.CreateBCs(1);

   // set the time step on the model
   dt = bc.dt;

   // loop over integration points for current element
   for (int i=0; i<ir->GetNPoints(); i++)
   {
      // set integration point number on model
      model->SetIpID(i);

      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);

      // get integration point coordinates
      model->ipx = ip.x;
      model->ipy = ip.y;
      model->ipz = ip.z;

      // compute Jacobian of the transformation
      CalcInverse(Ttr.Jacobian(), Jrt);

      // compute Jpt, which is the incremental deformation gradient
      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      // get the stress update, set on the model (model->P)
      model->EvalModel(Jpt, DS, ip.weight * Ttr.Weight());

      model->P *= ip.weight * Ttr.Weight();
      AddMultABt(DS, model->P, PMatO);
   }

   return;
}

void ExaNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, DenseMatrix &elmat)
{

   // TODO write this routine so that it does 
   // not call anything to compute the material 
   // stiffness, but rather assembles it from the 
   // matGrad quadrature function on the model
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
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }
  
   elmat = 0.0;
   model->SetTransformation(Ttr);
   model->SetElementID(Ttr.ElementNo, Ttr.Attribute); 

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      model->SetIpID(i);

      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      // call the AssembleH routine on the model to play nice with the 
      // nonlinear model class, even though the material stiffness MAY 
      // have been evaluated and returned by the user material subroutine 
      // evaluated in EvalP
      model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
   }
   return;
}

}
