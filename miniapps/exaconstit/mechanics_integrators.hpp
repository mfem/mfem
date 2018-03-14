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

#ifndef MECHANICS_INTEG
#define MECHANICS_INTEG

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"


//#include "../config/config.hpp"
//#include "fe.hpp"
//#include "coefficient.hpp"

namespace mfem
{

// define user defined material model base class
class UserDefinedModel
{
protected:
   ElementTransformation *Ttr; /**< Reference-element to target-element
                                    transformation. */

   // quadrature vector function coefficient for the beginning step stress and the end step 
   // (or incrementally upated) stress
   QuadratureVectorFunctionCoefficient stress0;
   QuadratureVectorFunctionCoefficient stress1;

   // quadrature vector function coefficient for the updated material tangent stiffness 
   // matrix, which will need to be stored after an EvalP call and used in a 
   // later AssembleH call
   QuadratureVectorFunctionCoefficient matGrad; 

   // quadrature vector function coefficients for any history variables at the beginning 
   // of the step and end (or incrementally updated) step.
   QuadratureVectorFunctionCoefficient matVars0;
   QuadratureVectorFunctionCoefficient matVars1;

public:
   UserDefinedModel(QuadratureFunction *q_stress0, QuadratureFunction *q_stress1,
                    QuadratureFunction *q_matGrad, QuadratureFunction *q_matVars0,
                    QuadratureFunction *q_matVars1) : Ttr(NULL), stress0(q_stress0), 
                        stress1(q_stress1), matGrad(q_matGrad), 
                        matVars0(q_matVars0), matVars1(q_matVars1) { }
   virtual ~UserDefinedModel() { }

   /// A reference-element to target-element transformation that can be used to
   /// evaluate Coefficient%s.
   /** @note It is assumed that _Ttr.SetIntPoint() is already called for the
       point of interest. */
   void SetTransformation(ElementTransformation &_Ttr) { Ttr = &_Ttr; }

   /** @brief Perform constitutive update by evaluating the user defined 
       material model */
   virtual void Update() const = 0;

   /** @brief Evaluate the strain energy density function, W = W(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix. */
   virtual double EvalW(const DenseMatrix &Jpt) const = 0;

   /** @brief Evaluate the 1st Piola-Kirchhoff stress tensor, P = P(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix.
       @param[out]  P  The evaluated 1st Piola-Kirchhoff stress tensor. */
   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const = 0;

   /** @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
       and assemble its contribution to the local gradient matrix 'A'.
       @param[in] Jpt     Represents the target->physical transformation
                          Jacobian matrix.
       @param[in] DS      Gradient of the basis matrix (dof x dim).
       @param[in] weight  Quadrature weight coefficient for the point.
       @param[in,out]  A  Local gradient matrix where the contribution from this
                          point will be added.

       Computes weight * d(dW_dxi)_d(xj) at the current point, for all i and j,
       where x1 ... xn are the FE dofs. This function is usually defined using
       the matrix invariants and their derivatives.
   */
   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const = 0;

};

// Abaqus Umat class
class AbaqusUmatModel : public UserDefinedModel
{
protected:

   // add member variables. 
   mutable double stress[6]; 
   mutable double *statev; // 1d array, size not specified
   mutable double ddsdde[6][6]; 
   mutable double *sse;
   mutable double *spd;
   mutable double *scd;
   mutable double *rpl;
   mutable double ddsdt[6]; 
   mutable double drplde[6]; 
   mutable double *drpldt;
   mutable double stran[6]; 
   mutable double dstran[6]; 
   mutable double time[2]; 
   mutable double *deltaTime;
   mutable double *tempk;
   mutable double *dtemp;
   mutable double *predef;
   mutable double *dpred;
   mutable double *cmname;
   mutable int ndi;
   mutable int nshr;
   mutable int ntens;
   mutable int nstatv;
   mutable double *props; // 1d array, size not specified
   mutable int nprops;
   mutable double coords[3]; 
   mutable double drot[3][3]; 
   mutable double pnewdt;
   mutable double *celent;
   mutable double dfgrd0[3][3]; 
   mutable double dfgrd1[3][3]; 
   mutable int *noel;
   mutable int *npt;
   mutable int *layer;
   mutable int *kspt;
   mutable int *kstep;
   mutable int *kinc; 

   // pointer to umat function
   void (*umat)(double[6], double[], double[6][6], 
                double*, double*, double*, double*,
                double[6], double[6], double*,
                double[6], double[6], double[2],
                double*, double*, double*, double*,
                double*, double*, int*, int*, int*,
                int *, double[], int*, double[3],
                double[3][3], double*, double*,
                double[3][3], double[3][3], int*, int*, 
                int*, int*, int*, int*);
                 

public:
   AbaqusUmatModel(QuadratureFunction *q_stress0, QuadratureFunction *q_stress1,
                   QuadratureFunction *q_matGrad, QuadratureFunction *q_matVars0,
                   QuadratureFunction *q_matVars1) : UserDefinedModel(q_stress0, 
                      q_stress1, q_matGrad, q_matVars0, q_matVars1)  
	                 { statev = NULL;  
                           props = NULL; 
                           umat = NULL; }
   virtual ~AbaqusUmatModel() { }

   void InitializeModel() const;

   virtual void Update() const;

   virtual double EvalW(const DenseMatrix &J) const;
   
   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};


class UserDefinedNLFIntegrator : public NonlinearFormIntegrator
{
private:
   UserDefinedModel *model;

public:
   UserDefinedNLFIntegrator(UserDefinedModel *m) : model(m) { }

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Ttr,
                                    const Vector &elfun, DenseMatrix &elmat);
};

}

#endif
