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

#ifndef MFEM_TRANSPORT_SOLVER
#define MFEM_TRANSPORT_SOLVER

#include "../common/pfem_extras.hpp"
#include "plasma.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

/** Multispecies Electron-Ion Collision Time in seconds
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
double tau_e(double Te, int ns, double * ni, int * zi, double lnLambda);

/** Multispecies Ion-Ion Collision Time in seconds
   ma is the ion mass in a.m.u
   Ta is the ion temperature in eV
   ion is the ion species index for the desired collision time
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
double tau_i(double ma, double Ta, int ion, int ns, double * ni, int * zi,
             double lnLambda);

/**
  Thermal diffusion coefficient along B field for electrons
  Return value is in m^2/s.
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
inline double chi_e_para(double Te, int ns, double * ni, int * zi)
{
   // The factor of q_ is included to convert Te from eV to Joules
   return 3.16 * (q_ * Te / me_kg_) * tau_e(Te, ns, ni, zi, 17.0);
}

/**
  Thermal diffusion coefficient perpendicular to B field for electrons
  Return value is in m^2/s.
*/
inline double chi_e_perp()
{
   // The factor of q_ is included to convert Te from eV to Joules
   return 1.0;
}

/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for electrons.
  Return value is in m^2/s.
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   z is the charge number of the ion species
*/
inline double chi_e_cross()
{
   // The factor of q_ is included to convert Te from eV to Joules
   return 0.0;
}

/**
  Thermal diffusion coefficient along B field for ions
  Return value is in m^2/s.
   ma is the ion mass in a.m.u.
   Ta is the ion temperature in eV
   ion is the ion species index for the desired coefficient
   ns is the number of ion species
   nb is the density of ions in particles per meter^3
   zb is the charge number of the ion species
*/
inline double chi_i_para(double ma, double Ta,
                         int ion, int ns, double * nb, int * zb)
{
   // The factor of q_ is included to convert Ta from eV to Joules
   // The factor of u_ is included to convert ma from a.m.u to kg
   return 3.9 * (q_ * Ta / (ma * amu_ ) ) *
          tau_i(ma, Ta, ion, ns, nb, zb, 17.0);
}

/**
  Thermal diffusion coefficient perpendicular to B field for ions
  Return value is in m^2/s.
*/
inline double chi_i_perp()
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of u_ is included to convert mi from a.m.u to kg
   return 1.0;
}

/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for ions
  Return value is in m^2/s.
*/
inline double chi_i_cross()
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of u_ is included to convert mi from a.m.u to kg
   return 0.0;
}

/**
  Viscosity coefficient along B field for electrons
  Return value is in (a.m.u)*m^2/s.
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
inline double eta_e_para(double ne, double Te, int ns, double * ni, int * zi)
{
   // The factor of q_ is included to convert Te from eV to Joules
   // The factor of u_ is included to convert from kg to a.m.u
   return 0.73 * ne * (q_ * Te / amu_) * tau_e(Te, ns, ni, zi, 17.0);
}

/**
  Viscosity coefficient along B field for ions
  Return value is in (a.m.u)*m^2/s.
   ma is the ion mass in a.m.u.
   Ta is the ion temperature in eV
   ion is the ion species index for the desired coefficient
   ns is the number of ion species
   nb is the density of ions in particles per meter^3
   zb is the charge number of the ion species
*/
inline double eta_i_para(double ma, double Ta,
                         int ion, int ns, double * nb, int * zb)
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of u_ is included to convert from kg to a.m.u
   return 0.96 * nb[ion] * (q_ * Ta / amu_) *
          tau_i(ma, Ta, ion, ns, nb, zb, 17.0);
}

class MultiSpeciesDiffusion;
class MultiSpeciesAdvection;

class TransportSolver : public ODESolver
{
private:
   ODESolver * impSolver_;
   ODESolver * expSolver_;

   ParFiniteElementSpace & sfes_; // Scalar fields
   ParFiniteElementSpace & vfes_; // Vector fields
   ParFiniteElementSpace & ffes_; // Full system

   BlockVector & nBV_;

   ParGridFunction & B_;

   Array<int> & charges_;
   Vector & masses_;

   MultiSpeciesDiffusion * msDiff_;

   void initDiffusion();

public:
   TransportSolver(ODESolver * implicitSolver, ODESolver * explicitSolver,
                   ParFiniteElementSpace & sfes,
                   ParFiniteElementSpace & vfes,
                   ParFiniteElementSpace & ffes,
                   BlockVector & nBV,
                   ParGridFunction & B,
                   Array<int> & charges,
                   Vector & masses);
   ~TransportSolver();

   void Update();

   void Step(Vector &x, double &t, double &dt);
};

class ChiParaCoefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;

   int ion_;
   Array<int> & z_;
   Vector     * m_;
   Vector       n_;

public:
   ChiParaCoefficient(BlockVector & nBV, Array<int> & charges);
   ChiParaCoefficient(BlockVector & nBV, int ion_species,
                      Array<int> & charges, Vector & masses);
   void SetT(ParGridFunction & T);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiPerpCoefficient : public Coefficient
{
private:
   int ion_;

public:
   ChiPerpCoefficient(BlockVector & nBV, Array<int> & charges);
   ChiPerpCoefficient(BlockVector & nBV, int ion_species,
                      Array<int> & charges, Vector & masses);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCrossCoefficient : public Coefficient
{
private:
   int ion_;

public:
   ChiCrossCoefficient(BlockVector & nBV, Array<int> & charges);
   ChiCrossCoefficient(BlockVector & nBV, int ion_species,
                       Array<int> & charges, Vector & masses);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCoefficient : public MatrixCoefficient
{
private:
   ChiParaCoefficient  chiParaCoef_;
   ChiPerpCoefficient  chiPerpCoef_;
   ChiCrossCoefficient chiCrossCoef_;
   VectorGridFunctionCoefficient BCoef_;

   Vector bHat_;

public:
   ChiCoefficient(int dim, BlockVector & nBV, Array<int> & charges);
   ChiCoefficient(int dim, BlockVector & nBV, int ion_species,
                  Array<int> & charges, Vector & masses);

   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class EtaParaCoefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;

   int ion_;
   Array<int> & z_;
   Vector     * m_;
   Vector       n_;

public:
   EtaParaCoefficient(BlockVector & nBV, Array<int> & charges);
   EtaParaCoefficient(BlockVector & nBV, int ion_species,
                      Array<int> & charges, Vector & masses);

   void SetT(ParGridFunction & T);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class MultiSpeciesDiffusion : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &sfes_;
   ParFiniteElementSpace &vfes_;

   BlockVector & nBV_;

   Array<int> & charges_;
   Vector & masses_;

   void initCoefficients();
   void initBilinearForms();

public:
   MultiSpeciesDiffusion(ParFiniteElementSpace & sfes,
                         ParFiniteElementSpace & vfes,
                         BlockVector & nBV,
                         Array<int> & charges,
                         Vector & masses);

   ~MultiSpeciesDiffusion();

   void Assemble();

   void Update();

   void ImplicitSolve(const double dt, const Vector &x, Vector &y);
};


// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the diffusion term. (modified from ex14p)
class DiffusionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   double dt_;
   double dg_sigma_;
   double dg_kappa_;

   ParFiniteElementSpace &fes_;
   ParFiniteElementSpace &dfes_;
   ParFiniteElementSpace &vfes_;

   ParBilinearForm m_;
   ParBilinearForm d_;

   ParLinearForm rhs_;
   ParGridFunction x_;

   HypreParMatrix * M_;
   HypreParMatrix * D_;

   Vector RHS_;
   Vector X_;

   HypreSolver * solver_;
   HypreSolver * amg_;

   MatrixCoefficient &nuCoef_;
   ScalarMatrixProductCoefficient dtNuCoef_;

   void initSolver(double dt);

public:
   DiffusionTDO(ParFiniteElementSpace &fes,
                ParFiniteElementSpace &dfes,
                ParFiniteElementSpace &_vfes,
                MatrixCoefficient & nuCoef,
                double dg_sigma,
                double dg_kappa);

   // virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~DiffusionTDO() { }
};

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the advection term.
class AdvectionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   const int num_equation_;
   const double specific_heat_ratio_;

   mutable double max_char_speed_;

   ParFiniteElementSpace &vfes_;
   Operator &A_;
   SparseMatrix &Aflux_;
   DenseTensor Me_inv_;

   mutable Vector state_;
   mutable DenseMatrix f_;
   mutable DenseTensor flux_;
   mutable Vector z_;

   void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;

public:
   AdvectionTDO(ParFiniteElementSpace &_vfes,
                Operator &A, SparseMatrix &Aflux, int num_equation,
                double specific_heat_ratio);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~AdvectionTDO() { }
};

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
   int num_equation_;
   double specific_heat_ratio_;
   Vector flux1_;
   Vector flux2_;

public:
   RiemannSolver(int num_equation, double specific_heat_ratio);
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};


// Constant (in time) mixed bilinear form multiplying the flux grid function.
// The form is (vec(v), grad(w)) where the trial space = vector L2 space (mesh
// dim) and test space = scalar L2 space.
class DomainIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape_;
   DenseMatrix flux_;
   DenseMatrix dshapedr_;
   DenseMatrix dshapedx_;

public:
   DomainIntegrator(const int dim, const int num_equation);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   int num_equation_;
   double max_char_speed_;
   RiemannSolver rsolver_;
   Vector shape1_;
   Vector shape2_;
   Vector funval1_;
   Vector funval2_;
   Vector nor_;
   Vector fluxN_;
   IntegrationPoint eip1_;
   IntegrationPoint eip2_;

public:
   FaceIntegrator(RiemannSolver &rsolver_, const int dim,
                  const int num_equation);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_TRANSPORT_SOLVER
