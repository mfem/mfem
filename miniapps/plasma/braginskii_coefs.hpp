// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BRAGINSKII_COEFFICIENTS
#define MFEM_BRAGINSKII_COEFFICIENTS

#include "../common/pfem_extras.hpp"
#include "plasma.hpp"

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
inline double tau_e(double Te, double zi, double ni, double lnLambda)
{
   // The factor of q_^{3/2} is included to convert Te from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * pow(q_ * Te, 3) / M_PI) /
          (lnLambda * pow(q_, 4) * zi * zi * ni);
}

/**
   Returns the mean Ion-Ion mean collision time in seconds (see equation 2.5i)
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
   lnLambda is the Coulomb Logarithm
*/
inline double tau_i(double mi, double zi, double ni, double Ti,
                    double lnLambda)
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
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   ni is the density of ions in particles per meter^3
   zi is the charge number of the ion species
*/
inline double chi_e_para(double ne, double Te, double zi, double ni)
{
   // The factor of q_ is included to convert Te from eV to Joules
   return t2_c0(zi) * ne * (q_ * Te / me_kg_) * tau_e(Te, zi, ni, 17.0);
}

/**
  Thermal diffusion coefficient perpendicular to B field for electrons
  Return value is in m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
*/
inline double chi_e_perp(double Bmag, double ne, double Te,
                         double zi, double ni)
{
   // The factor of q_ is included to convert Te from eV to Joules
   double tau = tau_e(Te, zi, ni, 17.0);
   double x = cyclotronFrequency(Bmag, me_u_, 1.0) * tau;
   double delta = x * x * (x * x + t2_d1(zi)) + t2_d0(zi);
   return ne * (q_ * Te / me_kg_) * tau *
          (t2_c1p(zi) * x * x + t2_c0p(zi)) / delta;
}

/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for electrons.
  Return value is in m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
inline double chi_e_cross(double Bmag, double ne, double Te,
                          double ni, double zi)
{
   // The factor of q_ is included to convert Te from eV to Joules
   double tau = tau_e(Te, zi, ni, 17.0);
   double x = cyclotronFrequency(Bmag, me_u_, 1.0) * tau;
   double delta = x * x * (x * x + t2_d1(zi)) + t2_d0(zi);
   return ne * (q_ * Te / me_kg_) * tau *
          x * (t2_c1pp(zi) * x * x + t2_c0pp(zi)) / delta;
}

/**
  Thermal diffusion coefficient along B field for ions
  Return value is in m^2/s.
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double chi_i_para(double mi, double zi, double ni, double Ti)
{
   // The factor of q_ is included to convert Ta from eV to Joules
   // The factor of u_ is included to convert ma from a.m.u to kg
   return 3.906 * ni * (q_ * Ti / (mi * amu_ ) ) *
          tau_i(mi, zi, ni, Ti, 17.0);
}

/**
  Thermal diffusion coefficient perpendicular to B field for ions
  Return value is in m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double chi_i_perp(double Bmag, double mi, double zi,
                         double ni, double Ti)
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of amu_ is included to convert mi from a.m.u to kg
   double tau = tau_i(mi, zi, ni, Ti, 17.0);
   double x = cyclotronFrequency(Bmag, mi, zi) * tau;
   double delta = x * x * (x * x + 2.70) + 0.677;
   return ni * (q_ * Ti / (mi * amu_)) * tau *
          (2.0 * x * x + 2.645) / delta;
}

/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for ions
  Return value is in m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double chi_i_cross(double Bmag, double mi, double zi,
                          double ni, double Ti)
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of amu_ is included to convert mi from a.m.u to kg
   double tau = tau_i(mi, zi, ni, Ti, 17.0);
   double x = cyclotronFrequency(Bmag, mi, zi) * tau;
   double delta = x * x * (x * x + 2.70) + 0.677;
   return ni * (q_ * Ti * tau/ (mi * amu_)) *
          (2.5 * x * x + 4.65) / delta;
}

/**
  Viscosity coefficient along B field for electrons
  Return value is in (a.m.u)*m^2/s.
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
*/
inline double eta0_e(double ne, double Te, double zi, double ni)
{
   // The factor of q_ is included to convert Te from eV to Joules
   // The factor of amu_ is included to convert from kg to a.m.u
   return 0.73 * ne * (q_ * Te / amu_) * tau_e(Te, zi, ni, 17.0);
}

/**
  First viscosity coefficient perpendicular to B field for electrons
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
*/
inline double eta1_e(double Bmag, double ne, double Te, double zi, double ni)
{
   // The factor of q_ is included to convert Te from eV to Joules
   // The factor of amu_ is included to convert from kg to a.m.u
   double omega = cyclotronFrequency(Bmag, me_u_, 1.0);
   return 0.51 * ne * (q_ * Te / amu_) /
          (omega * omega * tau_e(Te, zi, ni, 17.0));
}

/**
  Second viscosity coefficient perpendicular to B field for electrons
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
*/
inline double eta2_e(double Bmag, double ne, double Te, double zi, double ni)
{
   return 4.0 * eta1_e(Bmag, ne, Te, zi, ni);
}

/**
  Third viscosity coefficient perpendicular to B field for electrons
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
*/
inline double eta3_e(double Bmag, double ne, double Te)
{
   // The factor of q_ is included to convert Te from eV to Joules
   // The factor of amu_ is included to convert from kg to a.m.u
   double omega = cyclotronFrequency(Bmag, me_u_, 1.0);
   return -0.5 * ne * (q_ * Te / amu_) / omega;
}

/**
  Fourth viscosity coefficient perpendicular to B field for electrons
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
*/
inline double eta4_e(double Bmag, double ne, double Te)
{
   return 2.0 * eta3_e(Bmag, ne, Te);
}

/**
  Viscosity coefficient along B field for ions
  Return value is in (a.m.u)*m^2/s.
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double eta0_i(double mi, double zi, double ni, double Ti)
{
   // The factor of q_ is included to convert Ti from eV to Joules
   // The factor of u_ is included to convert from kg to a.m.u
   return 0.96 * ni * (q_ * Ti / amu_) * tau_i(mi, zi, ni, Ti, 17.0);
}

/**
  First viscosity coefficient perpendicular to B field for ions
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double eta1_i(double Bmag, double mi, double zi, double ni, double Ti)
{
   // The factor of q_ is included to convert Te from eV to Joules
   // The factor of amu_ is included to convert from kg to a.m.u
   double omega = cyclotronFrequency(Bmag, mi, zi);
   return 0.3 * ni * (q_ * Ti / amu_) /
          (omega * omega * tau_i(mi, zi, ni, Ti, 17.0));
}

/**
  Second viscosity coefficient perpendicular to B field for ions
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double eta2_i(double Bmag, double mi, double zi, double ni, double Ti)
{
   return 4.0 * eta1_i(Bmag, mi, zi, ni, Ti);
}

/**
  Third viscosity coefficient perpendicular to B field for ions
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double eta3_i(double Bmag, double mi, double zi, double ni, double Ti)
{
   // The factor of q_ is included to convert Te from eV to Joules
   // The factor of amu_ is included to convert from kg to a.m.u
   double omega = cyclotronFrequency(Bmag, mi, zi);
   return 0.5 * ni * (q_ * Ti / amu_) / omega;
}

/**
  Fourth viscosity coefficient perpendicular to B field for ions
  Return value is in (a.m.u)*m^2/s.
   Bmag is the magnitude of the magnetic field in tesla
   mi is the ion mass in a.m.u.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
*/
inline double eta4_i(double Bmag, double mi, double zi, double ni, double Ti)
{
   return 2.0 * eta3_i(Bmag, mi, zi, ni, Ti);
}

class DiffPerpCoefficient : public Coefficient
{
public:
   DiffPerpCoefficient();

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class DiffCrossCoefficient : public Coefficient
{
public:
   DiffCrossCoefficient();

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class DiffCoefficient : public MatrixCoefficient
{
private:
   VectorGridFunctionCoefficient BCoef_;

   // double ion_mass_;
   // double ion_charge_;

   mutable Vector bHat_;

public:
   DiffCoefficient(int dim, ParGridFunction & B);

   // void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ChiParaCoefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;

   bool   ion_;
   double zi_;
   double m_;
   double ne_;
   double ni_;

public:
   ChiParaCoefficient(BlockVector & nBV, double zi);
   ChiParaCoefficient(BlockVector & nBV, double mi, double zi);
   void SetT(ParGridFunction & T);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiPerpCoefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;
   VectorGridFunctionCoefficient BCoef_;

   bool ion_;
   double mi_;
   double zi_;
   double ne_;
   double ni_;

   mutable Vector B_;

public:
   ChiPerpCoefficient(BlockVector & nBV, double zi);
   ChiPerpCoefficient(BlockVector & nBV, double mi, double zi);
   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCrossCoefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;
   VectorGridFunctionCoefficient BCoef_;

   bool ion_;

   double mi_;
   double zi_;
   double ne_;
   double ni_;

   mutable Vector B_;

public:
   ChiCrossCoefficient(BlockVector & nBV, double zi);
   ChiCrossCoefficient(BlockVector & nBV, double mi, double zi);
   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCoefficient : public MatrixCoefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;
   VectorGridFunctionCoefficient BCoef_;

   bool   ion_;
   double zi_;
   double mi_;
   double ne_;
   double ni_;

   mutable Vector bHat_;

public:
   ChiCoefficient(int dim, BlockVector & nBV, ParGridFunction & B, double zi);
   ChiCoefficient(int dim, BlockVector & nBV, ParGridFunction & B,
                  double mi, double zi);

   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class Eta0Coefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;

   bool ion_;
   double zi_;
   double mi_;
   double ne_;
   double ni_;

public:
   Eta0Coefficient(BlockVector & nBV, double zi);
   Eta0Coefficient(BlockVector & nBV,
                   double mi, double zi);

   void SetT(ParGridFunction & T);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class Eta1Coefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;
   VectorGridFunctionCoefficient BCoef_;

   bool ion_;
   double zi_;
   double mi_;
   double ne_;
   double ni_;

   mutable Vector bHat_;

public:
   Eta1Coefficient(BlockVector & nBV, double zi);
   Eta1Coefficient(BlockVector & nBV,
                   double mi, double zi);

   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class Eta3Coefficient : public Coefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;
   VectorGridFunctionCoefficient BCoef_;

   bool ion_;
   double zi_;
   double mi_;
   double ne_;
   double ni_;

   mutable Vector bHat_;

public:
   Eta3Coefficient(BlockVector & nBV, double zi);
   Eta3Coefficient(BlockVector & nBV,
                   double mi, double zi);

   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class EtaCoefficient : public MatrixCoefficient
{
private:
   BlockVector & nBV_;
   ParFiniteElementSpace * sfes_;
   ParGridFunction nGF_;
   GridFunctionCoefficient nCoef_;
   GridFunctionCoefficient TCoef_;
   VectorGridFunctionCoefficient BCoef_;

   DenseMatrix del_;
   DenseMatrix eps2_;
   DenseTensor eps3_;

   int bi_;
   int bj_;

   bool ion_;
   double zi_;
   double mi_;
   double ne_;
   double ni_;

   mutable Vector bPara_;
   mutable Vector bPerp_;
   mutable DenseMatrix bx_;

   void initSymbols();

public:
   EtaCoefficient(int dim, int bi, int bj,
                  BlockVector & nBV, ParGridFunction & B, double zi);
   EtaCoefficient(int dim, int bi, int bj,
                  BlockVector & nBV, ParGridFunction & B,
                  double mi, double zi);

   void SetT(ParGridFunction & T);
   void SetB(ParGridFunction & B);

   void Eval(DenseMatrix & K, ElementTransformation &T,
             const IntegrationPoint &ip);
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

class pAdvectionCoefficient : public VectorCoefficient
{
private:
   double m_;
   Coefficient & nCoef_;
   VectorCoefficient & uCoef_;

public:
   pAdvectionCoefficient(double m, Coefficient & nCoef,
                         VectorCoefficient & uCoef);

   void Eval(Vector & K, ElementTransformation &T, const IntegrationPoint &ip);
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

class TAdvectionCoefficient : public VectorCoefficient
{
private:
   Coefficient & nCoef_;
   VectorCoefficient & uCoef_;

public:
   TAdvectionCoefficient(Coefficient & nCoef,
                         VectorCoefficient & uCoef);

   void Eval(Vector & K, ElementTransformation &T, const IntegrationPoint &ip);
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_BRAGINSKII_COEFFICIENTS

