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
#include "userumat.h"

namespace mfem
{

// free function to compute the beginning step deformation gradient to store 
// on a quadrature function
void computeDefGrad(const QuadratureFunction *qf, ParFiniteElementSpace *fes, 
                    const Vector &x0);
// A function that can be used to check and see what deformation gradient you're
// calculating
void computeDefGradTest(const QuadratureFunction *qf,
                        ParFiniteElementSpace *fes, const Vector &x0);
//mfem traditionally has things layed out [x0...xn, y0...yn, z0...zn]
//this function takes that vector and reorders so x,y, and z are interleaved
//with each other so [x0,y0,z0 ... xn, yn, zn]
void fixVectOrder(const Vector& v1, Vector &v2);

//This function does the reverse of the above and takes a vector that has
//data ordered as [x0,y0,z0 ... xn, yn, zn] and returns it as [x0...xn, y0...yn, z0...zn]
void reorderVectOrder(const Vector &v1, Vector &v2);
   
//One might typical stiffness matrices as being created to be applied to a vector
//that has an [x0,y0,z0 ... xn, yn, zn] ordering. This function takes a matrix ordered
//as such and reorders it such that it can now be applied to a vector that has an
//[x0...xn, y0...yn, z0...zn] ordering
void reorderMatrix(const DenseMatrix& a1, DenseMatrix& a2);
  
class ExaModel 
{
public:
   int numProps;
   int numStateVars;
   bool cauchy;
   bool debug;
   DenseMatrix currElemCoords; // local variable to store current configuration 
                               // element coordinates
protected:
   ElementTransformation *Ttr; /**< Reference-element to target-element
                                    transformation. */

   // local variables for an ip evaluation of the constitutive model
   int elemID, ipID, elemAttribute;
   double dt, t;
   double ipx, ipy, ipz;                                                            
   // If this variable is true then currently the mesh nodes correspond to the EndCoords. However,
   // if it is set to false then the mesh nodes correspond to the BegCoords.  
   bool EndCoordsMesh;

   //--------------------------------------------------------------------------
   // The velocity method requires us to retain both the beggining and end time step
   // coordinates of the mesh. We need these to be able to compute the correct
   // incremental deformation gradient (using the beg. time step coords) and the
   // velocity gradient (uses the end time step coords).
   // A pointer to the parallel mesh is also needed so that when needed we need
   // to swap between the reference and current configuration for our gradients
   // we can easily do so.

   ParGridFunction* beg_coords;
   ParGridFunction* end_coords;
   ParMesh* pmesh;
  
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
   QuadratureFunctionCoefficient vonMises;
  
   // add vector for material properties, which will be populated based on the 
   // requirements of the user defined model. The properties are expected to be 
   // the same at all quadrature points. That is, the material properties are 
   // constant and not dependent on space
   Vector *matProps;

   // beginning and end (at NR iterate) step deformation gradient for current
   // element (full deformation gradients) and material tangent.
   // All matrices are local to an integration point used 
   // for a given element level computation
   DenseMatrix Jpt0, Jpt1, CMat, Vgrad; // note: these local copies are for convenience
   DenseMatrix P; // temporary PK1 stress for NLF integrator to populate/access
   
   //These pointers point to the current elemental velocity, coord, and residual
   //fields for a given element. They should be used to allow one to pass information
   //between the ExaNLF class to the necessary EvalModel classes that lay in the ExaModel
   //and subsequent children classes. The EvalModel should update it's portion of the residual
   //internally. 
   const Vector *elvel;
   Vector *elresid;
   const Vector *elcrds;
  
   //---------------------------------------------------------------------------

public:
   ExaModel(QuadratureFunction *q_stress0, QuadratureFunction *q_stress1,
             QuadratureFunction *q_matGrad, QuadratureFunction *q_matVars0,
             QuadratureFunction *q_matVars1, QuadratureFunction *q_defGrad0,
	     ParGridFunction* _beg_coords, ParGridFunction* _end_coords, ParMesh* _pmesh,  
	    Vector *props, int nProps, int nStateVars, bool _cauchy, bool _endcrdm) : 
               numProps(nProps), numStateVars(nStateVars),
               cauchy(_cauchy), Ttr(NULL), 
               stress0(q_stress0),
               stress1(q_stress1), 
               matGrad(q_matGrad), 
               matVars0(q_matVars0), 
               matVars1(q_matVars1), 
               defGrad0(q_defGrad0),
	       beg_coords(_beg_coords),
	       end_coords(_end_coords),
	       pmesh(_pmesh),
               matProps(props),
	       EndCoordsMesh(_endcrdm) {}

   virtual ~ExaModel() { }

   /// A reference-element to target-element transformation that can be used to
   /// evaluate Coefficient%s.
   /** @note It is assumed that _Ttr.SetIntPoint() is already called for the
       point of interest. */
   void SetTransformation(ElementTransformation &_Ttr) { Ttr = &_Ttr; }

   //This function is used in generating the B matrix commonly seen in the formation of
   //the material tangent stiffness matrix in mechanics [B^t][Cstiff][B]
   virtual void GenerateGradMatrix(const DenseMatrix& DS, DenseMatrix& B);
   
   //This function is used in generating the B matrix that's used in the formation
   //of the geometric stiffness contribution of the stiffness matrix seen in mechanics
   //as [B^t][sigma][B]
   virtual void GenerateGradGeomMatrix(const DenseMatrix& DS, DenseMatrix& Bgeom);
   
   // routine to call constitutive update. Note that this routine takes
   // the weight input argument to conform to the old AssembleH where the 
   // weight was used in the NeoHookean model. Consider refactoring this
   virtual void EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight) = 0;

   //This function assembles the necessary stiffness matrix to be used in the
   //linearization of our nonlinear system of equations
   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) = 0;

   // routine to update the beginning step deformation gradient. This must
   // be written by a model class extension to update whatever else
   // may be required for that particular model
   virtual void UpdateModelVars(ParFiniteElementSpace *fes, 
                                const Vector &x) = 0;

   // routine to set the element id and integration point number for 
   // the current element
   void SetElementID(const int elementNum, 
                     const int attr) { elemID = elementNum;
                                       elemAttribute = attr; }

   // routine to set the integration point ID on the model
   void SetIpID(const int ipNum) { ipID = ipNum; }

   // function to set the integration coords for the model
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

   // function to set the internal von Mises QuadratureFuntion pointer to some
   // outside source
   void setVonMisesPtr(QuadratureFunction* vm_ptr) {vonMises = vm_ptr;}

   //Returns the value of the end step mesh boolean
   bool GetEndCoordsMesh() {return EndCoordsMesh; }
  
   // return a pointer to von Mises stress quadrature vector function coefficient for visualization
   QuadratureFunctionCoefficient *GetVonMises() { return &vonMises; }

   // return a pointer to the matVars0 quadrature vector function coefficient 
   QuadratureVectorFunctionCoefficient *GetMatVars0() { return &matVars0; }

   // return a pointer to the end coordinates
   // this should probably only be used within the solver itself
   // if it's touched outside of that who knows whether or not the data
   // might be tampered with and thus we could end up with weird results
   // It's currently only being exposed due to the requirements UMATS place
   // on how things are solved outside of this class
   // fix_me
   ParGridFunction *GetEndCoords(){return end_coords;}

   // return a pointer to the beggining coordinates
   // this should probably only be used within the solver itself
   // if it's touched outside of that who knows whether or not the data
   // might be tampered with and thus we could end up with weird results    
   // It's currently only being exposed due to the requirements UMATS place
   // on how things are solved outside of this class
   // fix_me
   ParGridFunction *GetBegCoords(){return beg_coords;}
   
   // This just seems bad to be doing this, but it's in many requirement of the UMAT
   // I might move these to within the UMAT class itself.
   // fix_me
   ParMesh *GetPMesh(){return pmesh;}
  
   // return a pointer to the matProps vector
   Vector *GetMatProps() { return matProps; }

   //return a pointer to the PK1 stress densematrix
   DenseMatrix *GetPK1Stress() {return &P;}

   //return a pointer to the Cauchy stress densematrix
   //We use the same dense matrix as for the PK1 stress
   DenseMatrix *GetCauchyStress() {return &P;}
   
  //return a pointer to the Velocity gradient densematrix
   DenseMatrix *GetVGrad() {return &Vgrad;}
  
   // Returns a copy of the Jpt1 DenseMatrix
   DenseMatrix CopyOfJpt1() {DenseMatrix temp(Jpt1); return temp;}
  
   // routine to get element stress at ip point. These are the six components of 
   // the symmetric Cauchy stress where standard Voigt notation is being used
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

   // Update the End Coordinates using a simple Forward Euler Integration scheme
   // The beggining time step coordinates should be updated outside of the model routines
   void UpdateEndCoords(const Vector& vel);

   // A simple routine to update the mesh nodes to either the end or beggining time step
   // coordinates. The method will depend on a protected bool called EndCoordsMesh. If this
   // variable is true then currently the mesh nodes correspond to the EndCoords. However,
   // if it is set to false then the mesh nodes correspond to the BegCoords.
   void SwapMeshNodes();
  
   void SymVoigtToDenseMat(const double* const A, DenseMatrix &B);

   void SetCoords(const DenseMatrix &coords) { currElemCoords = coords; }
   
   //The below functions are used to set the velocity, coords, and residual field variables to
   //an element to an outside Vector that should come from the ExaNLF class.
   void SetVel(const Vector *vel) {elvel = vel;}
   void SetResid(Vector *resid) {elresid = resid;}
   void SetCrds(const Vector *crds) {elcrds = crds;}
   
   //This method performs a fast approximate polar decomposition for 3x3 matrices
   //The deformation gradient or 3x3 matrix of interest to be decomposed is passed
   //in as the initial R matrix. The error on the solution can be set by the user.
   void CalcPolarDecompDefGrad(DenseMatrix& R, DenseMatrix& U,
                               DenseMatrix& V, double err = 1e-12);
   
   //Various Strain measures we can use
   //Same as above should these be a protected function?
   
   //Lagrangian is simply E = 1/2(F^tF - I)
   void CalcLagrangianStrain(DenseMatrix& E);
   //Eulerian is simply e = 1/2(I - F^(-t)F^(-1))
   void CalcEulerianStrain(DenseMatrix& E);
   //Biot strain is simply B = U - I
   void CalcBiotStrain(DenseMatrix& E);
   //Log strain is equal to e = 1/2 * ln(C) or for UMATs its e = 1/2 * ln(B)
   void CalcLogStrain(DenseMatrix& E);
   
   //Some useful rotation functions that we can use
   //Do we want to have these exposed publically or should they
   //be protected?
   //Also, do we want to think about moving these type of orientation
   //conversions to their own class?
   void Quat2RMat(const Vector& quat, DenseMatrix& rmat);
   void RMat2Quat(const DenseMatrix& rmat, Vector& quat);
   
   //Converts from Cauchy stress to PK1 stress
   void CauchyToPK1();

   //Converts from PK1 stress to Cauchy stress
   void PK1ToCauchy(const DenseMatrix &P, const DenseMatrix& J, double* sigma);

   //Computes the von Mises stress from the Cauchy stress
   void ComputeVonMises(const int elemID, const int ipID);
   
   //A test function that allows us to see what deformation gradient we're
   //getting out
   void test_def_grad_func(ParFiniteElementSpace *fes, const Vector &x0){
      computeDefGradTest(defGrad0.GetQuadFunction(), fes, x0);
   }

};

// Abaqus Umat class.
class AbaqusUmatModel : public ExaModel
{
protected:

   // add member variables. 
   double elemLength;
  
   // pointer to umat function
   // we really don't use this in the code
   void (*umatp)(double[6], double[], double[36], 
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
		   ParGridFunction* _beg_coords, ParGridFunction* _end_coords, ParMesh *_pmesh,
                   Vector *_props, int _nProps, 
                   int _nStateVars) : ExaModel(_q_stress0, 
                      _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1, _q_defGrad0,
		      _beg_coords, _end_coords,_pmesh,
		      _props, _nProps, _nStateVars, true, false) { }
   virtual ~AbaqusUmatModel() { }

   virtual void EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight);

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A);
   
   virtual void UpdateModelVars(ParFiniteElementSpace *fes,
                                const Vector &x);
   
   //Calculates the incremental versions of the strain measures that we're given
   //above
   void CalcLogStrainIncrement(DenseMatrix &dE, const DenseMatrix &Jpt);
   void CalcEulerianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt);
   void CalcLagrangianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt);
   
   //Returns the incremental rotation and strain based on the velocity gradient
   void CalcIncrStrainRot(double* dE, DenseMatrix& dRot);
   
   //calculates the element length
   void CalcElemLength();

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
   //This function really only be used whenever UMATs are being used.
   //The other models should be using the one above this
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr_beg,
				      ElementTransformation &Ttr_end,
                                      const Vector &elfun, Vector &elvect,
				      const Vector &elvel);
  
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
