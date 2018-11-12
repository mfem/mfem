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

namespace mfem
{

// free function to compute the beginning step deformation gradient to store 
// on a quadrature function
void computeDefGrad(const QuadratureFunction *qf, const ParFiniteElementSpace *fes, 
                    const Vector &x0);
void computeDefGradTest(const QuadratureFunction *qf,
                        ParFiniteElementSpace *fes, const Vector &x0);
   
class ExaModel 
{
public:
   int numProps;
   int numStateVars;
   bool cauchy;
   bool debug;
   DenseMatrix currElemCoords; // local variable to store current configuration 
                               // element coordinates 
  // DenseMatrix P; // temporary PK1 stress for NLF integrator to populate/access 
protected:
   ElementTransformation *Ttr; /**< Reference-element to target-element
                                    transformation. */

   // local variables for an ip evaluation of the constitutive model
   int elemID, ipID, elemAttribute;
   double dt, t;
   double ipx, ipy, ipz;

   //---------------------------------------------------------------------------
   // STATE VARIABLES and PROPS common to all user defined models

   // quadrature vector function coefficient for the beginning step stress and 
   // the end step (or incrementally upated) stress
   QuadratureVectorFunctionCoefficient stress0;
   QuadratureVectorFunctionCoefficient stress1;

   // quadrature vector function coefficient for the updated material tangent 
   // stiffness matrix, which will need to be stored after an EvalP call and 
   // used in a later AssembleH call
   QuadratureVectorFunctionCoefficient matGrad; 

   // quadrature vector function coefficients for any history variables at the 
   // beginning of the step and end (or incrementally updated) step.
   QuadratureVectorFunctionCoefficient matVars0;
   QuadratureVectorFunctionCoefficient matVars1;

   // add QuadratureVectorFunctionCoefficient to store the beginning step 
   // Note you can compute the end step def grad from the incremental def 
   // grad (from the solution: Jpt) and the beginning step def grad
   QuadratureVectorFunctionCoefficient defGrad0; 

   // add QuadratureVectorFunctionCoefficient to store von Mises 
   // scalar stress measure
   QuadratureVectorFunctionCoefficient vonMises;
  
   // add vector for material properties, which will be populated based on the 
   // requirements of the user defined model. The properties are expected to be 
   // the same at all quadrature points. That is, the material properties are 
   // constant and not dependent on space
   Vector matProps;

   // beginning and end (at NR iterate) step deformation gradient for current
   // element (full deformation gradients) and material tangent.
   // All matrices are local to an integration point used 
   // for a given element level computation
   DenseMatrix Jpt0, Jpt1, CMat; // note: these local copies are for convenience
   DenseMatrix P; // temporary PK1 stress for NLF integrator to populate/access

   //---------------------------------------------------------------------------

public:
   ExaModel(QuadratureFunction *q_stress0, QuadratureFunction *q_stress1,
             QuadratureFunction *q_matGrad, QuadratureFunction *q_matVars0,
             QuadratureFunction *q_matVars1, QuadratureFunction *q_defGrad0, 
             Vector &props, int nProps, int nStateVars, bool _cauchy) : 
               numProps(nProps), numStateVars(nStateVars),
               cauchy(_cauchy), Ttr(NULL), 
               stress0(q_stress0),
               stress1(q_stress1), 
               matGrad(q_matGrad), 
               matVars0(q_matVars0), 
               matVars1(q_matVars1), 
               defGrad0(q_defGrad0), 
               matProps(props) { }

   virtual ~ExaModel() { }

   /// A reference-element to target-element transformation that can be used to
   /// evaluate Coefficient%s.
   /** @note It is assumed that _Ttr.SetIntPoint() is already called for the
       point of interest. */
   void SetTransformation(ElementTransformation &_Ttr) { Ttr = &_Ttr; }

   // routine to call constitutive update. Note that this routine takes 
   // the weight input argument to conform to the old AssembleH where the 
   // weight was used in the NeoHookean model. Consider refactoring this
   virtual void EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight) = 0;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) = 0;

   // routine to update the beginning step deformation gradient. This must
   // be written by a model class extension to update whatever else
   // may be required for that particular model
   virtual void UpdateModelVars(const ParFiniteElementSpace *fes, 
                                const Vector &x) = 0;

   // routine to set the element id and integration point number for 
   // the current element
   void SetElementID(const int elementNum, 
                     const int attr) { elemID = elementNum;
                                       elemAttribute = attr; }

   // routine to set the integration point ID on the model
   void SetIpID(const int ipNum) { ipID = ipNum; }

   void SetIPCoords(const int x, const int y, const int z) 
      { ipx = x; ipy = y; ipz = z; }

   // set time on the base model class
   void SetModelTime(const double time) { t = time; }

   // set timestep on the base model class
   void SetModelDt(const double dtime) { dt = dtime; }

   // return a pointer to beginning step stress. This is used for output visualization
   QuadratureVectorFunctionCoefficient *GetStress0() { return &stress0; }

   // return a pointer to beginning step stress. This is used for output visualization
   QuadratureVectorFunctionCoefficient *GetStress1() { return &stress1; }

   // return a pointer to the deformation gradient.
   QuadratureVectorFunctionCoefficient *GetDefGrad0() { return &defGrad0; }
  
   // return a pointer to von Mises stress quadrature vector function coefficient for visualization
   QuadratureVectorFunctionCoefficient *GetVonMises() { return &vonMises; }

   // return a pointer to the matVars0 quadrature vector function coefficient 
   QuadratureVectorFunctionCoefficient *GetMatVars0() { return &matVars0; }

   // return a pointer to the matProps vector
   Vector *GetMatProps() { return &matProps; }

   //return a pointer to the PK1 stress densematrix
   DenseMatrix *GetPK1Stress() {return &P;}

  DenseMatrix CopyOfJpt1() {DenseMatrix temp(Jpt1); return temp;}
  
   // routine to get element stress at ip point. These are the six components of 
   // the symmetric Cauchy stress
   void GetElementStress(const int elID, const int ipNum, bool beginStep, 
                         double* stress, int numComps);

   // set the components of the member function end stress quadrature function with 
   // the updated stress
   void SetElementStress(const int elID, const int ipNum, bool beginStep, 
                         double* stress, int numComps);

   // routine to get the element statevars at ip point.
   void GetElementStateVars(const int elID, const int ipNum, bool beginStep, 
                            double* stateVars, int numComps);

   // routine to set the element statevars at ip point
   void SetElementStateVars(const int elID, const int ipNum, bool beginStep, 
                            double* stateVars, int numComps);

   // routine to get the material properties data from the decorated mfem vector
   void GetMatProps(double* props);

   // setter for the material properties data on the user defined model object
   void SetMatProps(double* props, int size);

   // routine to set the material Jacobian for this element and integration point.
   void SetElementMatGrad(const int elID, const int ipNum, double* grad, int numComps);

   // routine to get the material Jacobian for this element and integration point
   void GetElementMatGrad(const int elId, const int ipNum, double* grad, int numComps); 

   int GetStressOffset();
  
   int GetMatGradOffset();

   int GetMatVarsOffset();

   // get the beginning step deformation gradient
   void GetElemDefGrad0(); 

   // calc the full end step deformation gradient using stored beginning step 
   // deformation gradient and current incremental deformation gradient 
   // (calculated from the element transformation)
   void CalcElemDefGrad1(const DenseMatrix &Jpt);

   // routine to update beginning step stress with end step values
   void UpdateStress(int elID, int ipNum);

   // routine to update beginning step state variables with end step values
   void UpdateStateVars(int elID, int ipNum);
 
   void SymVoigtToDenseMat(const double* const A, DenseMatrix &B);

   void SetCoords(const DenseMatrix &coords) { currElemCoords = coords; }

   //Should we move this over to a protected function?
   void CalcPolarDecompDefGrad(DenseMatrix& R, DenseMatrix& U,
                               DenseMatrix& V, double err = 1e-12);
   //Various Strain measures we can use
   //Same as above should these be a protected function?
   void CalcLagrangianStrain(DenseMatrix& E);
   void CalcEulerianStrain(DenseMatrix& E);
   void CalcBiotStrain(DenseMatrix& E);
   void CalcLogStrain(DenseMatrix& E);
   
   //Some useful rotation functions that we can use
   //Do we want to have these exposed publically or should they
   //be protected?
   //Also, do we want to think about moving these type of orientation
   //conversions to their own class?
   void Quat2RMat(Vector& quat, DenseMatrix& rmat);
   void RMat2Quat(DenseMatrix& rmat, Vector& quat);
   
   void CauchyToPK1();

   void PK1ToCauchy(const DenseMatrix &P, const DenseMatrix& J, double* sigma);

   void ComputeVonMises(const int elemID, const int ipID);
   
   void test_def_grad_func(ParFiniteElementSpace *fes, const Vector &x0){
      computeDefGradTest(defGrad0.GetQuadFunction(), fes, x0);
   }

};

// Abaqus Umat class. This has NOT been tested
class AbaqusUmatModel : public ExaModel
{
protected:

   // add member variables. 
   double elemLength;

   // pointer to umat function
   void (*umat)(double[6], double[], double[36], 
                double*, double*, double*, double*,
                double[6], double[6], double*,
                double[6], double[6], double[2],
                double*, double*, double*, double*,
                double*, double*, int*, int*, int*,
                int *, double[], int*, double[3],
                double[9], double*, double*,
                double[9], double[9], int*, int*, 
                int*, int*, int*, int*);
                 

public:
   AbaqusUmatModel(QuadratureFunction *_q_stress0, QuadratureFunction *_q_stress1,
                   QuadratureFunction *_q_matGrad, QuadratureFunction *_q_matVars0,
                   QuadratureFunction *_q_matVars1, QuadratureFunction *_q_defGrad0, 
                   Vector _props, int _nProps, 
                   int _nStateVars) : ExaModel(_q_stress0, 
                      _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1, _q_defGrad0, 
                      _props, _nProps, _nStateVars, true) { }
   virtual ~AbaqusUmatModel() { }

   virtual void EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight);

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A);
   
   virtual void UpdateModelVars(const ParFiniteElementSpace *fes,
                                const Vector &x);

   void CalcLogStrainIncrement(DenseMatrix &dE);
   void CalcEulerianStrainIncr(DenseMatrix& dE);
   void CalcLagrangianStrainIncr(DenseMatrix& dE);

   void CalcElemLength();

};

class NeoHookean : public ExaModel
{
protected:
   // from fem/nonlininteg.hpp
   mutable double mu, K, g;
   Coefficient *c_mu, *c_K, *c_g;
   bool have_coeffs;

   mutable DenseMatrix Z;    // dim x dim
   mutable DenseMatrix G, C; // dof x dim

   inline void EvalCoeffs() const;
public:
   NeoHookean(QuadratureFunction *_q_stress0, QuadratureFunction *_q_stress1,
              QuadratureFunction *_q_matGrad, QuadratureFunction *_q_matVars0,
              QuadratureFunction *_q_matVars1, QuadratureFunction *_q_defGrad0, 
              Vector _props, int _nProps, int _nStateVars, double _mu, double _K, 
              double _g = 1.0) : 
              ExaModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1, 
              _q_defGrad0, _props, _nProps, _nStateVars, false), mu(_mu), K(_K), g(_g), 
              have_coeffs(false) 
              { c_mu = c_K = c_g = NULL; }

   NeoHookean(QuadratureFunction *_q_stress0, QuadratureFunction *_q_stress1,
              QuadratureFunction *_q_matGrad, QuadratureFunction *_q_matVars0,
              QuadratureFunction *_q_matVars1, QuadratureFunction *_q_defGrad0, 
              Vector _props, int _nProps, int _nStateVars, Coefficient &_mu, 
              Coefficient &_K, Coefficient *_g = NULL) : 
              ExaModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1, 
              _q_defGrad0, _props, _nProps, _nStateVars, false), 
              mu(0.0), K(0.0), g(1.0), c_mu(&_mu), 
              c_K(&_K), c_g(_g), have_coeffs(false) { }

   virtual ~NeoHookean() { }

   // place original EvalP and AssembleH into Eval()
   virtual void EvalModel(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight);

  virtual void EvalModel(const DenseMatrix &J);
  
   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A);
   
   virtual void UpdateModelVars(const ParFiniteElementSpace *fes,
                                const Vector &x);
};

class ExaNLFIntegrator : public NonlinearFormIntegrator
{
private:
   ExaModel *model;

public:
   ExaNLFIntegrator(ExaModel *m) : model(m) { }

   virtual ~ExaNLFIntegrator() { }
  
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Ttr,
                                    const Vector &elfun, DenseMatrix &elmat);

   // debug routine for finite difference approximation of 
   // the element tangent stiffness contribution 
   void AssembleElementGradFD(const FiniteElement &el,
                              ElementTransformation &Ttr,
                              const Vector &elfun, DenseMatrix &elmat);

};

}

#endif
