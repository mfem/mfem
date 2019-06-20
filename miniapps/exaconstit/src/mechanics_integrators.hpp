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
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
//#include "exacmech.hpp" //Will need to export all of the various header files into here as well

//namespace mfem
//{
using namespace mfem;
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
               EndCoordsMesh(_endcrdm),
               beg_coords(_beg_coords),
               end_coords(_end_coords),
               pmesh(_pmesh),
               stress0(q_stress0),
               stress1(q_stress1), 
               matGrad(q_matGrad), 
               matVars0(q_matVars0), 
               matVars1(q_matVars1), 
               defGrad0(q_defGrad0),
               matProps(props){}

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
   
   //This function is needed in the UMAT child class to drive parts of the
   //solution in the mechanics_operator file. It should just be set as a no-op
   //in other children class if they aren't using it.
   virtual void calc_incr_end_def_grad(ParFiniteElementSpace *fes,
                                       const Vector &x0) = 0;
   
   //This function is needed in the UMAT child class to drive parts of the
   //solution in the mechanics_operator file.
   //It should just be set as a no-op
   //in other children class if they aren't using it.
   //For when the ParFinitieElementSpace is stored on the class...
   virtual void calc_incr_end_def_grad(const Vector &x0) = 0;

   // routine to update the beginning step deformation gradient. This must
   // be written by a model class extension to update whatever else
   // may be required for that particular model
   virtual void UpdateModelVars() = 0;

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
   void CalcLagrangianStrain(DenseMatrix& E, const DenseMatrix &F);
   //Eulerian is simply e = 1/2(I - F^(-t)F^(-1))
   void CalcEulerianStrain(DenseMatrix& E, const DenseMatrix &F);
   //Biot strain is simply B = U - I
   void CalcBiotStrain(DenseMatrix& E, const DenseMatrix &F);
   //Log strain is equal to e = 1/2 * ln(C) or for UMATs its e = 1/2 * ln(B)
   void CalcLogStrain(DenseMatrix& E, const DenseMatrix &F);
   
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
   
   //The initial local shape function gradients.
   QuadratureFunction loc0_sf_grad;
   
   //The incremental deformation gradients.
   QuadratureFunction incr_def_grad;
   
   //The end step deformation gradients.
   QuadratureFunction end_def_grad;
   ParFiniteElementSpace* loc_fes;
  
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
                   int _nStateVars, ParFiniteElementSpace* fes) : ExaModel(_q_stress0,
                      _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1, _q_defGrad0,
		      _beg_coords, _end_coords,_pmesh,
		      _props, _nProps, _nStateVars, true, false), loc_fes(fes)
   {
      init_loc_sf_grads(fes);
      init_incr_end_def_grad();
   }
   
   virtual ~AbaqusUmatModel() { }

   virtual void EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight);

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A);
   virtual void calc_incr_end_def_grad(ParFiniteElementSpace *fes,
                                       const Vector &x0) {}
   
   //For when the ParFinitieElementSpace is stored on the class...
   virtual void calc_incr_end_def_grad(const Vector &x0);
   virtual void UpdateModelVars();
   
   //Calculates the incremental versions of the strain measures that we're given
   //above
   void CalcLogStrainIncrement(DenseMatrix &dE, const DenseMatrix &Jpt);
   void CalcEulerianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt);
   void CalcLagrangianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt);
   
   //Returns the incremental rotation and strain based on the velocity gradient
   void CalcIncrStrainRot(double* dE, DenseMatrix& dRot);
   
   //calculates the element length
   void CalcElemLength();
   
   void init_loc_sf_grads(ParFiniteElementSpace *fes);
   void init_incr_end_def_grad();
   

};

using namespace ecmech;
// Base class for all of our ExaCMechModels.
class ExaCMechModel : public ExaModel
{
protected:

   //Current temperature in Kelvin degrees
   double temp_k;
   //The indices of our history variables which will be quite useful for
   //post processing of the results.
   //These will be set during the initialization stage of things
   //Initial values for these are passed in through the state variable file.
   //This file will include everything associated with the history variables
   //along with the relative volume value and the initial internal energy value.
   //For most purposes the relative volume should be set to 1 and the initial internal
   //energy should be set to 0.
   int ind_dp_eff, ind_eql_pl_strain, ind_num_evals, ind_dev_elas_strain;
   int ind_quats, ind_hardness, ind_gdot, ind_vols, ind_int_eng;
   //number of hardness variables and number of slip systems
   //these are available in the mat_model_base class as well but I thought
   //it might not hurt to have these explicitly declared.
   int num_hardness, num_slip, num_vols, num_int_eng;
   //Our total number of state variables for the FCC Voce and KinKMBalDD model should be equal to
   //3+5+1+12+2 = 27 with 4 supplied from quaternion sets so 23 should be in the state variable file.
   virtual void class_instantiation() = 0;
   //A pointer to our actual material model class that ExaCMech uses.
   //The childern classes to this class will have also have another variable
   //that actually contains the real material model that is then dynamically casted
   //to this base class during the instantiation of the class. 
   matModelBase* mat_model_base;
   //We might also want a QuadratureFunctionCoefficientVector class for our accumulated gammadots called gamma
   //We would update this using a pretty simple integration scheme so gamma += delta_t * gammadot
   //QuadratureVectorFunctionCoefficient gamma;                 

public:
   ExaCMechModel(QuadratureFunction *_q_stress0, QuadratureFunction *_q_stress1,
             QuadratureFunction *_q_matGrad, QuadratureFunction *_q_matVars0,
             QuadratureFunction *_q_matVars1, QuadratureFunction *_q_defGrad0,
	          ParGridFunction* _beg_coords, ParGridFunction* _end_coords, ParMesh* _pmesh,  
	          Vector *_props, int _nProps, int _nStateVars, double _temp_k) : 
             ExaModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
               _q_defGrad0, _beg_coords, _end_coords, _pmesh, _props, _nProps, _nStateVars,
               true, true), temp_k(_temp_k) { }

   virtual ~ExaCMechModel() { }
   //The interface for this will look the same across all of the other functions
   virtual void EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight);

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A);

   virtual void calc_incr_end_def_grad(ParFiniteElementSpace *fes,
                                       const Vector &x0) {}
   
   //For when the ParFinitieElementSpace is stored on the class...
   virtual void calc_incr_end_def_grad(const Vector &x0){}
   virtual void UpdateModelVars();

};

// A linear isotropic Voce hardening model with a power law formulation for the slip kinetics
// This model generally can do a decent job of capturing the material behavior in strain rates
// that are a bit lower where thermally activated slip is a more appropriate approximation.
// Generally, you'll find that if fitted to capture the elastic plastic transition it will miss the later
// plastic behavior of the material. However, if it is fitted to the general macroscopic stress-strain
// curve and the d\sigma / d \epsilon_e vs epsilon it will miss the elastic-plastic regime.
// Based on far-field high energy x-ray diffraction (ff-HEXD) data, this model is capable of capture 1st
// order behaviors of the distribution of elastic intragrain heterogeneity. However, it fails to capture 
// transient behaviors of these distributions as seen in  http://doi.org/10.7298/X4JM27SD and 
// http://doi.org/10.1088/1361-651x/aa6dc5 for fatigue applications.

// A good reference for the Voce implementation can be found in:
// section 2.1 https://doi.org/10.1016/S0045-7825(98)00034-6
// section 2.1 https://doi.org/10.1016/j.ijplas.2007.03.004
// Basics for how to fit such a model can be found here:
// https://doi.org/10.1016/S0921-5093(01)01174-1 . Although, it should be noted
// that this is more for the MTS model it can be adapted to the Voce model by taking into
// account that the m parameter determines the rate sensitivity. So, the more rate insensitive
// the material is the closer this will be to 0. It is also largely responsible for how sharp
// the knee of the macroscopic stress-strain curve is. You'll often see OFHC copper around 0.01 - 0.02.
// The exponent to the Voce hardening law can be determined by what ordered function best
// fits the d\sigma / d \epsilon_e vs epsilon curve. The initial CRSS term best determines
// when the material starts to plastically deform. The saturation CRSS term determines pretty
// much how much the material is able to harden. The hardening coeff. for CRSS best determines
// the rate at which the material hardens so larger values lead to a quicker hardening of the material.
// The saturation CRSS isn't seen as constant in several papers involving this model.
// Since, they are often only examining things for a specific strain rate and temperature.
//I don't have a lot of personal experience with fitting this, so I can't provide
//good guidance on how to properly fit this parameter. 
//
class VoceFCCModel : public ExaCMechModel
{
protected:

   //We can define our class instantiation using the following
   virtual void class_instantiation(){

      mat_model_base = dynamic_cast<matModelBase*>(&mat_model);

      ind_dp_eff = ecmech::evptn::ind_histA_shrateEff;
      ind_eql_pl_strain = ecmech::evptn::ind_histA_shrEff;
      ind_num_evals = ecmech::evptn::ind_histA_nFEval;
      ind_dev_elas_strain = ecmech::evptn::ind_hist_LbE;
      ind_quats = ecmech::evptn::ind_hist_LbQ;
      ind_hardness = ecmech::evptn::ind_hist_LbH;
      
      ind_gdot = mat_model.ind_hist_Lb_gdot;
      //This will always be 1 for this class
      num_hardness = mat_model.nH;
      //This will always be 12 for this class
      num_slip = mat_model.nslip;
      //The number of vols -> we actually only need to save the previous time step value
      //instead of all 4 values used in the evalModel. The rest can be calculated from
      //this value.
      num_vols = 1;
      ind_vols = ind_gdot + num_slip;
      //The number of internal energy variables -> currently 1
      num_int_eng = 1;
      ind_int_eng = ind_vols + num_vols;
      //Params start off with:
      //initial density, heat capacity at constant volume, and a tolerance param
      //Params then include Elastic constants:
      // c11, c12, c44 for Cubic crystals
      //Params then include the following: 
      //shear modulus, m parameter seen in slip kinetics, gdot_0 term found in slip kinetic eqn,
      //hardening coeff. defined for g_crss evolution eqn, initial CRSS value,
      //initial CRSS saturation strength, CRSS saturation strength scaling exponent,
      //CRSS saturation strength rate scaling coeff, tausi -> hdn_init (not used)
      //Params then include the following:
      //the Grüneisen parameter, reference internal energy

      //Opts and strs are just empty vectors of int and strings
      std::vector<double> params;
      std::vector<int> opts;
      std::vector<std::string> strs;
      //9 terms come from hardening law, 3 terms come from elastic modulus, 4 terms are related to EOS params,
      //1 terms is related to solving tolerances == 17 total params
      MFEM_ASSERT(matProps->Size() == 17, "Properties did not contain 17 parameters for Voce FCC model.");

      for(int i = 0; i < matProps->Size(); i++)
      {
         params.push_back(matProps->Elem(i));
      }
      //We really shouldn't see this change over time at least for our applications.
      mat_model_base->initFromParams(opts, params, strs);
      //
      mat_model_base->complete();

      std::vector<double> histInit_vec;
      {
         std::vector<std::string> names;
         std::vector<bool>        plot;
         std::vector<bool>        state;
         mat_model_base->getHistInfo(names, histInit_vec, plot, state );
      }
      
      QuadratureFunction* _state_vars = matVars0.GetQuadFunction();
      double* state_vars = _state_vars->GetData();

      int qf_size = (_state_vars->Size()) / (_state_vars->GetVDim());
      
      int vdim = _state_vars->GetVDim();
      int ind = 0;
      
      for(int i = 0; i < qf_size; i++)
      {
         ind = i * vdim;

         state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
         state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
         state_vars[ind + ind_num_evals] = histInit_vec[ind_num_evals];
         state_vars[ind + ind_hardness] = histInit_vec[ind_hardness];
         state_vars[ind + ind_vols] = 1.0;

         for(int j = 0; j < ecmech::ne; j++)
         {
            state_vars[ind + ind_int_eng] = 0.0;
         }

         for(int j = 0; j < 5; j++)
         {
            state_vars[ind + ind_dev_elas_strain + j] = histInit_vec[ind_dev_elas_strain + j];
         }

         for(int j = 0; j < num_slip; j++)
         {
            state_vars[ind + ind_gdot + j] = histInit_vec[ind_gdot + j];
         }

      }

   }

   //This is a type alias for:
   //evptn::matModel< SlipGeomFCC, KineticsVocePL, evptn::ThermoElastNCubic, EosModelConst<false> >
   //We also need to store the full class instantiation of the class on our own class.
   matModelEvptn_FCC_A mat_model; 

   //We might also want a QuadratureFunctionCoefficientVector class for our accumulated gammadots called gamma
   //We would update this using a pretty simple integration scheme so gamma += delta_t * gammadot
   //QuadratureVectorFunctionCoefficient gamma;                 

public:
   VoceFCCModel(QuadratureFunction *_q_stress0, QuadratureFunction *_q_stress1,
             QuadratureFunction *_q_matGrad, QuadratureFunction *_q_matVars0,
             QuadratureFunction *_q_matVars1, QuadratureFunction *_q_defGrad0,
	          ParGridFunction* _beg_coords, ParGridFunction* _end_coords, ParMesh* _pmesh,  
	          Vector *_props, int _nProps, int _nStateVars, double _temp_k) : 
               ExaCMechModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
			     _q_defGrad0, _beg_coords, _end_coords,_pmesh, _props, _nProps, _nStateVars, _temp_k)
               {
                  class_instantiation();
               }

   virtual ~VoceFCCModel() { }

};

// A class with slip and hardening kinetics based on a single Kocks-Mecking dislocation density
// balanced thermally activated MTS-like slip kinetics with phonon drag effects.
// See papers https://doi.org/10.1088/0965-0393/17/3/035003 (Section 2 - 2.3) 
// and https://doi.org/10.1063/1.4792227  (Section 3 up to the intro of the twinning kinetics ~ eq 8)
// for more info on this particular style of models.
// This model includes a combination of the above two see the actual implementation of
// ExaCMech ECMech_kinetics_KMBalD.h file for the actual specifics.
//
// This model is much more complicated than the simple Voce hardening model and power law slip kinetics
// seen above. However, it is capable of capturing the behavior of the material over a wide range of
// not only strain rates but also temperature ranges. The thermal activated slip kinetics is more or less
// what the slip kinetic power law used with the Voce hardening model approximates as seen in: 
// https://doi.org/10.1016/0079-6425(75)90007-9 and more specifically the Emperical Law section eqns:
// 34h - 34s. It should be noted though that this was based on work for FCC materials.
// The classical MTS model can be seen here towards its application towards copper for historical context: 
// https://doi.org/10.1016/0001-6160(88)90030-2
// An incredibly detailed overview of the thermally activated slip mechanisms can
// be found in https://doi.org/10.1016/S0079-6425(02)00003-8 . The conclusions provide a nice
// overview for how several of the parameters can be fitted for this model. Sections 2.3 - 3.4
// also go a bit more in-depth into the basis for why the fits are done the way they are
// conducted.
// The phonon drag contribution has shown to really start to play a role at strain rates
// 10^3 and above. A bit of a review on this topic can be found in https://doi.org/10.1016/0001-6160(87)90285-9 .
// It should be noted that the model implemented here does not follow the same formulation
// provided in that paper. This model can be thought of as a simplified version. 
class KinKMBalDDFCCModel : public ExaCMechModel
{
protected:

   //We can define our class instantiation using the following
   virtual void class_instantiation(){

      mat_model_base = dynamic_cast<matModelBase*>(&mat_model);

      ind_dp_eff = ecmech::evptn::ind_histA_shrateEff;
      ind_eql_pl_strain = ecmech::evptn::ind_histA_shrEff;
      ind_num_evals = ecmech::evptn::ind_histA_nFEval;
      ind_dev_elas_strain = ecmech::evptn::ind_hist_LbE;
      ind_quats = ecmech::evptn::ind_hist_LbQ;
      ind_hardness = ecmech::evptn::ind_hist_LbH;
      
      ind_gdot = mat_model.ind_hist_Lb_gdot;
      //This will always be 1 for this class
      num_hardness = mat_model.nH;
      //This will always be 12 for this class
      num_slip = mat_model.nslip;
      //The number of vols -> we actually only need to save the previous time step value
      //instead of all 4 values used in the evalModel. The rest can be calculated from
      //this value.
      num_vols = 1;
      ind_vols = ind_gdot + num_slip;
      //The number of internal energy variables -> currently 1
      num_int_eng = ecmech::ne;
      ind_int_eng = ind_vols + num_vols;

      //Params start off with:
      //initial density, heat capacity at constant volume, and a tolerance param
      //Params then include Elastic constants:
      // c11, c12, c44 for Cubic crystals
      //Params then include the following: 
      //reference shear modulus, reference temperature, g_0 * b^3 / \kappa where b is the 
      //magnitude of the burger's vector and \kappa is Boltzmann's constant, Peierls barrier,
      //MTS curve shape parameter (p), MTS curve shape parameter (q), reference thermally activated
      //slip rate, reference drag limited slip rate, drag reference stress, slip resistance const (g_0),
      //slip resistance const (s), dislocation density production constant (k_1), 
      //dislocation density production constant (k_{2_0}), dislocation density exponential constant,
      //reference net slip rate constant, reference relative dislocation density
      //Params then include the following:
      //the Grüneisen parameter, reference internal energy

      //Opts and strs are just empty vectors of int and strings
      std::vector<double> params;
      std::vector<int> opts;
      std::vector<std::string> strs;
      //15 terms come from hardening law, 3 terms come from elastic modulus, 4 terms are related to EOS params,
      //1 terms is related to solving tolerances == 23 total params
      MFEM_ASSERT(matProps->Size() == 23, "Properties need 23 parameters for FCC MTS like hardening model");

      for(int i = 0; i < matProps->Size(); i++)
      {
         params.push_back(matProps->Elem(i));
      }
      //We really shouldn't see this change over time at least for our applications.
      mat_model_base->initFromParams(opts, params, strs);
      //
      mat_model_base->complete();


      std::vector<double> histInit_vec;
      {
         std::vector<std::string> names;
         std::vector<bool>        plot;
         std::vector<bool>        state;
         mat_model_base->getHistInfo(names, histInit_vec, plot, state );
      }
      
      QuadratureFunction* _state_vars = matVars0.GetQuadFunction();
      double* state_vars = _state_vars->GetData();

      int qf_size = (_state_vars->Size()) / (_state_vars->GetVDim());
      
      int vdim = _state_vars->GetVDim();
      int ind = 0;
      
      for(int i = 0; i < qf_size; i++)
      {
         ind = i * vdim;

         state_vars[ind + ind_dp_eff] = histInit_vec[ind_dp_eff];
         state_vars[ind + ind_eql_pl_strain] = histInit_vec[ind_eql_pl_strain];
         state_vars[ind + ind_num_evals] = histInit_vec[ind_num_evals];
         state_vars[ind + ind_hardness] = histInit_vec[ind_hardness];
         state_vars[ind + ind_vols] = 1.0;

         for(int j = 0; j < ecmech::ne; j++)
         {
            state_vars[ind + ind_int_eng] = 0.0;
         }

         for(int j = 0; j < ecmech::ntvec; j++)
         {
            state_vars[ind + ind_dev_elas_strain + j] = histInit_vec[ind_dev_elas_strain + j];
         }

         for(int j = 0; j < num_slip; j++)
         {
            state_vars[ind + ind_gdot + j] = histInit_vec[ind_gdot + j];
         }
         
      }
   }

   //This is a type alias for:
   //evptn::matModel< SlipGeomFCC, Kin_KMBalD_FFF, evptn::ThermoElastNCubic, EosModelConst<false> >
   //where Kin_KMBalD_FFF is further a type alias for:
   //KineticsKMBalD< false, false, false > Kin_KMBalD_FFF; 
   //We also need to store the full class instantiation of the class on our own class.
   matModelEvptn_FCC_B mat_model; 

   //We might also want a QuadratureFunctionCoefficientVector class for our accumulated gammadots called gamma
   //We would update this using a pretty simple integration scheme so gamma += delta_t * gammadot
   //QuadratureVectorFunctionCoefficient gamma;                 

public:
   KinKMBalDDFCCModel(QuadratureFunction *_q_stress0, QuadratureFunction *_q_stress1,
             QuadratureFunction *_q_matGrad, QuadratureFunction *_q_matVars0,
             QuadratureFunction *_q_matVars1, QuadratureFunction *_q_defGrad0,
	          ParGridFunction* _beg_coords, ParGridFunction* _end_coords, ParMesh* _pmesh,  
	          Vector *_props, int _nProps, int _nStateVars, double _temp_k) : 
               ExaCMechModel(_q_stress0, _q_stress1, _q_matGrad, _q_matVars0, _q_matVars1,
               _q_defGrad0, _beg_coords, _end_coords, _pmesh, _props, _nProps, _nStateVars, _temp_k)
               {
                  class_instantiation();
               }

   virtual ~KinKMBalDDFCCModel() { }

};

//End the need for the ecmech namespace
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

//}

#endif
