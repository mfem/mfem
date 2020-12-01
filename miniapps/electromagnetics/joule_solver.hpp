// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_JOULE_SOLVER
#define MFEM_JOULE_SOLVER

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace electromagnetics
{

// Some global variable for convenience
const double       SOLVER_TOL = 1.0e-9;
const int       SOLVER_MAX_IT = 1000;
// Initialized in joule.cpp and used in joule_solver.cpp:
extern int SOLVER_PRINT_LEVEL;
extern int        STATIC_COND;

// These are defined in joule.cpp
void edot_bc(const Vector &x, Vector &E);
void e_exact(const Vector &x, double t, Vector &E);
void b_exact(const Vector &x, double t, Vector &B);
double p_bc(const Vector &x, double t);
double t_exact(const Vector &x);

// A Coefficient is an object with a function Eval that returns a double.  A
// MeshDependentCoefficient returns a different value depending upon the given
// mesh attribute, i.e. a "material property".
// Somewhat inefficiently, this is achieved using a GridFunction.
class MeshDependentCoefficient: public Coefficient
{
private:
   std::map<int, double> *materialMap;
   double scaleFactor;
public:
   MeshDependentCoefficient(const std::map<int, double> &inputMap,
                            double scale = 1.0);
   MeshDependentCoefficient(const MeshDependentCoefficient &cloneMe);
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   void SetScaleFactor(const double &scale) { scaleFactor = scale; }
   virtual ~MeshDependentCoefficient()
   {
      if (materialMap != NULL) { delete materialMap; }
   }
};

// This Coefficient is a product of a GridFunction and MeshDependentCoefficient
// for example if T (temperature) is a GridFunction and c (heat capacity) is a
// MeshDependentCoefficient, this function can compute c*T.
class ScaledGFCoefficient: public GridFunctionCoefficient
{
private:
   MeshDependentCoefficient mdc;
public:
   ScaledGFCoefficient(GridFunction *gf, MeshDependentCoefficient &input_mdc);
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   void SetMDC(const MeshDependentCoefficient &input_mdc) { mdc = input_mdc; }
   virtual ~ScaledGFCoefficient() {}
};

/**
   After spatial discretization, the magnetic diffusion equation can be written
   as a system of ODEs:

      S0(sigma) P = 0
      dE/dt       = - (M1(sigma) + dt S1(1/mu))^{-1}*(S1(1/mu)*E + sigma Grad P)
      dB/dt       = - Curl(E)
      dF/dt       = (M2(c/k) + dt S2(1))^{-1} (-S2(1) F + Div J)
      dcT/dt      = -Div F + J

   where

     P is the 0-form electrostatic potential,
     E is the 1-form electric field,
     B is the 2-form magnetic flux,
     F is the 2-form thermal flux,
     T is the 3-form temperature.
     M is the mass matrix,
     S is the stiffness matrix,
     Curl is the curl matrix,
     Div is the divergence matrix,
     sigma is the electrical conductivity,
     k is the thermal conductivity,
     c is the heat capacity,
     J is a function of the Joule heating sigma (E dot E).

   Class MagneticDiffusionEOperator represents the right-hand side of the above
   system of ODEs.
*/
class MagneticDiffusionEOperator : public TimeDependentOperator
{
protected:
   // These ParFiniteElementSpace objects provide degree-of-freedom mappings.
   // To create these you must provide the mesh and the definition of the FE
   // space. These objects are used to create hypre vectors to store the DOFs,
   // they are used to create grid functions to perform FEM interpolation, and
   // they are used by bilinear forms.
   ParFiniteElementSpace &L2FESpace;
   ParFiniteElementSpace &HCurlFESpace;
   ParFiniteElementSpace &HDivFESpace;
   ParFiniteElementSpace &HGradFESpace;

   // ParBilinearForms are used to create sparse matrices representing discrete
   // linear operators.
   ParBilinearForm *a0, *a1, *a2, *m1, *m2, *m3, *s1, *s2;
   ParDiscreteLinearOperator *grad, *curl;
   ParMixedBilinearForm *weakDiv, *weakDivC, *weakCurl;

   // Hypre matrices and vectors for 1-form systems A1 X1 = B1 and 2-form
   // systems A2 = X2 = B2
   HypreParMatrix *A0, *A1, *A2, *M1, *M2, *M3;
   Vector *X0, *X1, *X2, *B0, *B1, *B2, *B3;

   // temporary work vectors
   ParGridFunction *v0, *v1, *v2;

   // HypreSolver is derived from Solver, which is derived from Operator. So a
   // HypreSolver object has a Mult() operator, which is actually the solver
   // operation y = A^-1 x i.e. multiplication by A^-1.
   // HyprePCG is a wrapper for hypre's preconditioned conjugate gradient.
   mutable HypreSolver * amg_a0;
   mutable HyprePCG    * pcg_a0;
   mutable HypreSolver * ads_a2;
   mutable HyprePCG    * pcg_a2;
   mutable HypreSolver * ams_a1;
   mutable HyprePCG    * pcg_a1;
   mutable HypreSolver * dsp_m3;
   mutable HyprePCG    * pcg_m3;
   mutable HypreSolver * dsp_m1;
   mutable HyprePCG    * pcg_m1;
   mutable HypreSolver * dsp_m2;
   mutable HyprePCG    * pcg_m2;

   mutable Array<int> ess_bdr;
   mutable Array<int> ess_bdr_vdofs;
   mutable Array<int> thermal_ess_bdr;
   mutable Array<int> thermal_ess_bdr_vdofs;
   mutable Array<int> poisson_ess_bdr;
   mutable Array<int> poisson_ess_bdr_vdofs;

   MeshDependentCoefficient *sigma, *Tcapacity, *InvTcap, *InvTcond;
   double mu, dt_A1, dt_A2;

   // The method builA2 creates the ParBilinearForm a2, the HypreParMatrix A2,
   // and the solver and preconditioner pcg_a2 and amg_a2. The other build
   // functions do similar things.
   void buildA0(MeshDependentCoefficient &sigma);
   void buildA1(double muInv, MeshDependentCoefficient &sigma, double dt);
   void buildA2(MeshDependentCoefficient &InvTcond,
                MeshDependentCoefficient &InvTcap, double dt);
   void buildM1(MeshDependentCoefficient &sigma);
   void buildM2(MeshDependentCoefficient &alpha);
   void buildM3(MeshDependentCoefficient &Tcap);
   void buildS1(double muInv);
   void buildS2(MeshDependentCoefficient &alpha);
   void buildGrad();
   void buildCurl(double muInv);
   void buildDiv( MeshDependentCoefficient &InvTcap);

public:
   MagneticDiffusionEOperator(int len,
                              ParFiniteElementSpace &L2FES,
                              ParFiniteElementSpace &HCurlFES,
                              ParFiniteElementSpace &HDivFES,
                              ParFiniteElementSpace &HGradFES,
                              Array<int> &ess_bdr,
                              Array<int> &thermal_ess_bdr,
                              Array<int> &poisson_ess_bdr,
                              double mu,
                              std::map<int, double> sigmaAttMap,
                              std::map<int, double> TcapacityAttMap,
                              std::map<int, double> InvTcapAttMap,
                              std::map<int, double> InvTcondAttMap
                             );

   // Initialize the fields. This is where restart would go to.
   void Init(Vector &vx);

   // class TimeDependentOperator is derived from Operator, and class Operator
   // has the virtual function Mult(x,y) which computes y = A x for some matrix
   // A, or more generally a nonlinear operator y = A(x).
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   // Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown
   // slope k. This is the only requirement for high-order SDIRK implicit
   // integration. This is a virtual function of class TimeDependentOperator.
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   // Compute B^T M2 B, where M2 is the HDiv mass matrix with permeability
   // coefficient.
   // double MagneticEnergy(ParGridFunction &B_gf) const;

   // Compute E^T M1 E, where M1 is the HCurl mass matrix with conductivity
   // coefficient.
   double ElectricLosses(ParGridFunction &E_gf) const;

   // E is the input, w is the output which is L2 heating.
   void GetJouleHeating(ParGridFunction &E_gf, ParGridFunction &w_gf) const;

   void SetTime(const double _t);

   // Write all the hypre matrices and vectors to disk.
   void Debug(const char *basefilename, double time);

   virtual ~MagneticDiffusionEOperator();
};

// A Coefficient is an object with a function Eval that returns a double. The
// JouleHeatingCoefficient object will contain a reference to the electric field
// grid function, and the conductivity sigma, and returns sigma E dot E at a
// point.
class JouleHeatingCoefficient: public Coefficient
{
private:
   ParGridFunction &E_gf;
   MeshDependentCoefficient sigma;
public:
   JouleHeatingCoefficient(const MeshDependentCoefficient &sigma_,
                           ParGridFunction &E_gf_)
      : E_gf(E_gf_), sigma(sigma_) {}
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~JouleHeatingCoefficient() {}
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_JOULE_SOLVER
