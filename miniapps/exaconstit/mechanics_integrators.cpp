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

namespace mfem
{

// member functions for the Abaqus Umat base class
void AbaqusUmatModel::InitializeModel() const
{
   // local variables used for converting input
   int orient, tempVar, enerVar, enerRPL, failVar, kinType;
   double rotTmp[3][3];

   // initialize Umat member variables
   ntens = 6;
   noel  = NULL;
   npt   = NULL;
   layer = NULL;
   kspt  = NULL;
   kstep = NULL;
   kinc  = NULL;

   pnewdt = 10.0; // revisit this
   rpl    = NULL;
   drpldt = NULL;
   dtemp  = NULL; 
   predef = NULL;
   dpred  = NULL;

   for (int i=0; i<3; ++i) {
      coords[i] = 0.0;
   }

   for (int i=0; i<2; ++i) {
      time[i] = 0.0;
   }
   
   // initialize 1d arrays
   for (int i=0; i<6; ++i) {
      stress[i] = 0.0;
      ddsdt[i]  = 0.0;
      drplde[i] = 0.0;
      stran[i]  = 0.0;
      dstran[i] = 0.0;
   } 

   // initialize 6x6 2d arrays
   for (int i=0; i<6; ++i) {
      for (int j=0; j<6; ++j) {
         ddsdde[i][j] = 0.0;
      }
   }

   // initialize 3x3 2d arrays
   for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
         drot[i][j]   = 0.0;
         dfgrd0[i][j] = 0.0;
         dfgrd1[i][j] = 0.0;
         rotTmp[i][j] = 0.0;
      }
   }

   // TODO: populate the necessary data and take into account Fortran array
   // ordering.

   return;
}

void AbaqusUmatModel::Update() const
{

   // initialize the model
   InitializeModel();

   // call fortran umat routine
   umat(stress, statev, ddsdde, sse, spd, scd, rpl, 
        ddsdt, drplde, drpldt, stran, dstran, time, deltaTime,
        tempk, dtemp, predef, dpred, cmname, &ndi, &nshr, &ntens,
        &nstatv, props, &nprops, coords, drot, &pnewdt, celent,
        dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc);

   return;
}

double AbaqusUmatModel::EvalW(const DenseMatrix &J) const
{
   (void) J;
   return 0.0;
}

void AbaqusUmatModel::EvalP(const DenseMatrix &J,
                            DenseMatrix &P) const
{
   (void) J;
   (void) P;

   // note: this routine is called on a per integration point basis
   //
   // Call the Update routine, which initializes the Umat data per 
   // the given integration point and then calls the Umat. Then a 
   // QuadratureFunction to store the tangent stiffness matrix is 
   // populated.

   Update();

   return;
}

void AbaqusUmatModel::AssembleH(const DenseMatrix &J,
                                const DenseMatrix &DS,
                                const double weight,
                                DenseMatrix &A) const
{
   return;
}


// member functions for the UserDefinedNLFIntegrator
double UserDefinedNLFIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun)
{
   return 0.0;
}

void UserDefinedNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   return;
}

void UserDefinedNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, DenseMatrix &elmat)
{
   return;
}

}
