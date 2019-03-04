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

#ifndef MFEM_BRAGINSKII_TRANSPORT_SOLVER
#define MFEM_BRAGINSKII_TRANSPORT_SOLVER

#include "../common/pfem_extras.hpp"
#include "plasma.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{
/**
   The coefficients and much of the notation used here are taken from
   "Transport Processes in a Plasma" by S.I. Braginskii 1965.  The main
   difference being that Braginskii uses the CGS-Gaussian system of units
   whereas we use the SI unit system.
*/

/**
   Returns the mean Electron-Ion mean collision time in seconds (see
   equation 2.5e)
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
inline double tau_e(double Te, double ni, int zi, double lnLambda)
{
   // The factor of q_^{3/2} is included to convert Te from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * pow(q_ * Te, 3) / M_PI) /
          (lnLambda * pow(q_, 4) * zi * zi * ni);
}

/**
   Returns the mean Ion-Ion mean collision time in seconds (see equation 2.5i)
   mi is the ion mass in a.m.u.
   Ti is the ion temperature in eV
   ni is the density of ions in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
inline double tau_i(double mi, double Ti, double ni, int zi, double lnLambda)
{
   // The factor of q_^{3/2} is included to convert Ti from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(mi * amu_ * pow(q_ * Ti, 3) / M_PI) /
          (lnLambda * pow(q_ * zi, 4) * ni);
}

/**
   Many of the coefficients depend upon the charge of the ion species.  These
   dependencies are tabulated in Table 2 (pg. 251) of Braginskii.  The
   functions given here interpolate the values in this table.

   For brevity the function names use latin letters in place of greek
   e.g. 'a' rather than 'alpha'.

   * Note that the tabulated values for alpha_0 are inconsistent with the
   declared values of 1-(alpha_0'/delta_0).  In this case we return the values
   from 1-(alpha_0'/delta_0) rather than the tabulated values because the curve
   fit of the tabulated values was inconsistent with the expected value
   as z->infinity.
*/
inline double t2_a0(double z)
{
   return 0.2945 + 0.1613 * exp(-0.1795 * z) + 0.2936 * exp(-1.258 * z);
}
inline double t2_b0(double z)
{
   return 1.521 - 0.5202 * exp(-0.9325 * z) - 0.6845 * exp(-0.1230 * z);
}
inline double t2_c0(double z)
{
   return 12.47 - 4.114 * exp(-0.6409 * z) - 7.922 * exp(-0.1036 * z);
}
inline double t2_d0(double z)
{
   return 0.0961 + 25.63 * exp(-2.250 * z) + 1.424 * exp(-0.3801 * z);
}
inline double t2_d1(double z)
{
   return 7.482 + 21.49 * exp(-1.690 * z) + 4.321 * exp(-0.2566 * z);
}
inline double t2_a1p(double z)
{
   return 4.630 + 4.248 * exp(-1.565 * z) + 1.140 * exp(-0.2387 * z);
}
inline double t2_a0p(double z)
{
   return 0.0678 + 10.10 * exp(-2.103 * z) + 0.7635 * exp(-0.3523 * z);
}
inline double t2_a1pp(double z)
{
   return 1.704;
}
inline double t2_a0pp(double z)
{
   return 0.0940 + 2.920 * exp(-1.909 * z) + 0.3441 * exp(-0.3082 * z);
}
inline double t2_b1p(double z)
{
   return 3.798 + 3.090 * exp(-1.564 * z) + 0.8344 * exp(-0.2395 * z);
}
inline double t2_b0p(double z)
{
   return 0.1461 + 13.51 * exp(-2.060 * z) + 1.134 * exp(-0.3341 * z);
}
inline double t2_b1pp(double z)
{
   return 1.5;
}
inline double t2_b0pp(double z)
{
   return 0.877 + 7.590 * exp(-1.802 * z) + 1.220 * exp(-0.2773 * z);
}
inline double t2_c1p(double z)
{
   return 3.25 + 3.322 * exp(-1.533 * z) + 0.8789 * exp(-0.2325 * z);
}
inline double t2_c0p(double z)
{
   return 1.20 + 46.83 * exp(-1.937 * z) + 5.351 * exp(-0.2986 * z);
}
inline double t2_c1pp(double z)
{
   return 2.5;
}
inline double t2_c0pp(double z)
{
   return 10.23 + 34.75 * exp(-1.727 * z) + 6.844 * exp(-0.2635 * z);
}

/**
  Particle diffusion coefficient perpendicular to B field for ions
  Return value is in m^2/s.
*/
inline double diff_i_perp()
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of u_ is included to convert mi from a.m.u to kg
   return 1.0;
}

/**
  Particle diffusion coefficient perpendicular to both B field and
  particle gradient for ions
  Return value is in m^2/s.
*/
inline double diff_i_cross()
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of u_ is included to convert mi from a.m.u to kg
   return 0.0;
}

/**
  Thermal diffusion coefficient along B field for electrons
  Return value is in m^2/s.
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
inline double chi_e_para(double Te, int ns, double * ni, double * zi)
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
                         int ion, int ns, double * nb, double * zb)
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
inline double eta_e_para(double ne, double Te, int ns, double * ni, double * zi)
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
                         int ion, int ns, double * nb, double * zb)
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of u_ is included to convert from kg to a.m.u
   return 0.96 * nb[ion] * (q_ * Ta / amu_) *
          tau_i(ma, Ta, ion, ns, nb, zb, 17.0);
}

struct DGParams
{
   double sigma;
   double kappa;
};

class MultiSpeciesDiffusion;
class MultiSpeciesAdvection;

class ReducedTransportSolver : public ODESolver
{
private:
   ODESolver * impSolver_;
   ODESolver * expSolver_;

   DGParams & dg_;

   ParFiniteElementSpace & sfes_; // Scalar fields
   ParFiniteElementSpace & vfes_; // Vector fields
   ParFiniteElementSpace & ffes_; // Full system

   BlockVector & nBV_;
   BlockVector & uBV_;
   BlockVector & TBV_;

   ParGridFunction & B_;

   Vector & charges_;
   Vector & masses_;

   MultiSpeciesDiffusion * msDiff_;

   void initDiffusion();

public:
   ReducedTransportSolver(ODESolver * implicitSolver,
                          ODESolver * explicitSolver,
                          DGParams & dg,
                          ParFiniteElementSpace & sfes,
                          ParFiniteElementSpace & vfes,
                          ParFiniteElementSpace & ffes,
                          BlockVector & nBV,
                          BlockVector & uBV,
                          BlockVector & TBV,
                          ParGridFunction & B,
                          Vector & charges,
                          Vector & masses);
   ~ReducedTransportSolver();

   void Update();

   void Step(Vector &x, double &t, double &dt);
};

class DiffPerpCoefficient : public Coefficient
{
private:
   int ion_;

public:
   DiffPerpCoefficient(BlockVector & nBV, int ion_species,
                       Vector & charges, Vector & masses);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class DiffCrossCoefficient : public Coefficient
{
private:
   int ion_;

public:
   DiffCrossCoefficient(BlockVector & nBV, int ion_species,
                        Vector & charges, Vector & masses);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class DiffCoefficient : public MatrixCoefficient
{
private:
   DiffPerpCoefficient  diffPerpCoef_;
   DiffCrossCoefficient diffCrossCoef_;
   VectorGridFunctionCoefficient BCoef_;

   Vector bHat_;

public:
   DiffCoefficient(int dim, BlockVector & nBV, int ion_species,
                   Vector & charges, Vector & masses);

   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ChiParaCoefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;

   int ion_;
   Vector & z_;
   Vector * m_;
   Vector   n_;

public:
   ChiParaCoefficient(BlockVector & nBV, Vector & charges);
   ChiParaCoefficient(BlockVector & nBV, int ion_species,
                      Vector & charges, Vector & masses);
   void SetT(ParGridFunction & T);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiPerpCoefficient : public Coefficient
{
private:
   int ion_;

public:
   ChiPerpCoefficient(BlockVector & nBV, Vector & charges);
   ChiPerpCoefficient(BlockVector & nBV, int ion_species,
                      Vector & charges, Vector & masses);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCrossCoefficient : public Coefficient
{
private:
   int ion_;

public:
   ChiCrossCoefficient(BlockVector & nBV, Vector & charges);
   ChiCrossCoefficient(BlockVector & nBV, int ion_species,
                       Vector & charges, Vector & masses);

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
   ChiCoefficient(int dim, BlockVector & nBV, Vector & charges);
   ChiCoefficient(int dim, BlockVector & nBV, int ion_species,
                  Vector & charges, Vector & masses);

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
   Vector & z_;
   Vector * m_;
   Vector   n_;

public:
   EtaParaCoefficient(BlockVector & nBV, Vector & charges);
   EtaParaCoefficient(BlockVector & nBV, int ion_species,
                      Vector & charges, Vector & masses);

   void SetT(ParGridFunction & T);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dpdnCoefficient : public Coefficient
{
private:
   int c_;
   double m_;
   VectorCoefficient & uCoef_;
   mutable Vector u_;

public:
   dpdnCoefficient(int c, double m, VectorCoefficient & uCoef);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dpduCoefficient : public Coefficient
{
private:
   double m_;
   Coefficient & nCoef_;

public:
   dpduCoefficient(double m, Coefficient & nCoef);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dEdnCoefficient : public Coefficient
{
private:
   Coefficient & TCoef_;
   VectorCoefficient & uCoef_;
   double m_;

   mutable Vector u_;

public:
   dEdnCoefficient(Coefficient & TCoef, double m, VectorCoefficient & uCoef);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dEduCoefficient : public Coefficient
{
private:
   int c_;
   double m_;
   Coefficient & nCoef_;
   VectorCoefficient & uCoef_;
   mutable Vector u_;

public:
   dEduCoefficient(int c, double m, Coefficient & nCoef,
                   VectorCoefficient & uCoef);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

typedef ProductCoefficient dEdTCoefficient;

class MultiSpeciesDiffusion : public TimeDependentOperator
{
private:
   int dim_;

   DGParams & dg_;

   ParFiniteElementSpace &sfes_;
   ParFiniteElementSpace &vfes_;

   BlockVector & nBV_;
   BlockVector & uBV_;
   BlockVector & TBV_;

   Vector & charges_;
   Vector & masses_;

   std::vector<ParGridFunction> nGF_;
   std::vector<ParGridFunction> uGF_;
   std::vector<ParGridFunction> TGF_;

   std::vector<GridFunctionCoefficient>       nCoef_;
   std::vector<VectorGridFunctionCoefficient> uCoef_;
   std::vector<GridFunctionCoefficient>       TCoef_;

   std::vector<dpdnCoefficient *> dpdnCoef_;
   std::vector<dpduCoefficient *> dpduCoef_;

   std::vector<dEdnCoefficient *> dEdnCoef_;
   std::vector<dEduCoefficient *> dEduCoef_;
   std::vector<dEdTCoefficient *> dEdTCoef_;

   std::vector<DiffCoefficient *> diffCoef_;
   std::vector<ChiCoefficient *>  chiCoef_;

   std::vector<ScalarMatrixProductCoefficient *> nChiCoef_;

   std::vector<ScalarMatrixProductCoefficient *> dtnChiCoef_;
   std::vector<ScalarMatrixProductCoefficient *> dtDiffCoef_;

   // Bilinear Forms for energy equation
   std::vector<ParBilinearForm *> a_dEdn_;
   std::vector<ParBilinearForm *> a_dEdu_;
   std::vector<ParBilinearForm *> a_dEdT_;
   std::vector<ParBilinearForm *> stiff_nChi_;

   void initCoefficients();
   void initBilinearForms();

   void deleteCoefficients();
   void deleteBilinearForms();

public:
   MultiSpeciesDiffusion(DGParams & dg,
                         ParFiniteElementSpace & sfes,
                         ParFiniteElementSpace & vfes,
                         BlockVector & nBV,
                         BlockVector & uBV,
                         BlockVector & TBV,
                         Vector & charges,
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

#endif // MFEM_BRAGINSKII_TRANSPORT_SOLVER
