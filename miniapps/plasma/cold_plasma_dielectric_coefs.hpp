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

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_COEFS
#define MFEM_COLD_PLASMA_DIELECTRIC_COEFS

#include "../common/pfem_extras.hpp"
#include "plasma.hpp"
#include <complex>

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

// Cyclotron frequency
inline double omega_c(double Bmag   /* Tesla */,
                      double charge /* electron charge */,
                      double mass   /* AMU */)
{
   return (charge * q_ * Bmag) / (mass * amu_);
}
inline std::complex<double> omega_c(double Bmag, double charge,
                                    std::complex<double> mass)
{
   return (charge * q_ * Bmag) / (mass * amu_);
}

// Plasma frequency in s^{-1}
inline double omega_p(double number /* particles / m^3 */,
                      double charge /* electron charge */,
                      double mass   /* AMU */)
{
   return fabs(charge * q_) * 1.0 * sqrt(number / (epsilon0_ * mass * amu_));
}
inline std::complex<double> omega_p(double number, double charge,
                                    std::complex<double> mass)
{
   return fabs(charge * q_) * 1.0 * sqrt(number / (epsilon0_ * mass * amu_));
}

// Coulomb logarithm
inline double CoulombLog(double n, double Te)
{
   return log((4.0 * M_PI * pow(epsilon0_ * Te, 1.5)) / (pow(q_, 3) * sqrt(n)));
}

// Collisional frequency between electrons and ions
inline double nu_ei(double charge, double coul_log, double mass,
                    double Te, double number)
{
   return (8.0 * number * M_PI * pow(charge * q_, 4) * coul_log) /
          (3.0 * sqrt(2.0 * M_PI * me_kg_) * pow(4.0 * M_PI * epsilon0_, 2)
           * pow(Te, 1.5));
}

// Collisional frequency profile from Kohno et al 2017:
inline double nu_art(double x) { return (3e11*exp(-x/0.1)); }

std::complex<double> R_cold_plasma(double omega, double Bmag,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp);

std::complex<double> L_cold_plasma(double omega, double Bmag,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp);

std::complex<double> P_cold_plasma(double omega,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp);

std::complex<double> S_cold_plasma(double omega, double Bmag,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp);

std::complex<double> D_cold_plasma(double omega, double Bmag,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp);
// """""""""""""""""""""""""""""""""""""""""""""""""""""""
// Jim's old sheath parameterization from Kohno et al 2017:
double gabsANY(double x);

double gargANY(double x);

std::complex<double> ficmplxANY1(double x);

double xafun(double x);

double maxfun(double x);

std::complex<double> ficmplxANY(double omega, double vpp);

double vrectfun(double x);

std::complex<double> fdcmplxANY(double omega, double vpp);

std::complex<double> fecmplxANY(double vpp);

std::complex<double> ftotcmplxANY(double omega, double vpp);
// """""""""""""""""""""""""""""""""""""""""""""""""""""""
double mu(double mass_e, double mass_i);

double ff(double x);

double gg(double w);

double phi0avg(double w, double xi);

double he(double x);

double phips(double bx, double wci,
             double mass_e,
             double mass_i);

double niw(double wci, double bx,
           double phi, double mass_e,
           double mass_i);

double ye(double bx, double xi);

double niww(double w, double wci,
            double bx, double xi,
            double mass_e,
            double mass_i);

std::complex<double> yd(double w, double wci,
                        double bx, std::complex<double> xi,
                        double mass_e,
                        double mass_i);

std::complex<double> yi(double w, double wci,
                        double bx, std::complex<double> xi,
                        double mass_e,
                        double mass_i);

std::complex<double> ytot(double w, double wci,
                          double bx, std::complex<double> xi,
                          double mass_e,
                          double mass_i);

double debye(double Te, double n0_cm);

class SheathBase: public Coefficient
{
public:
   SheathBase(const BlockVector & density,
              const BlockVector & temp,
              const ParFiniteElementSpace & L2FESpace,
              const ParFiniteElementSpace & H1FESpace,
              double omega,
              const Vector & charges,
              const Vector & masses,
              bool realPart = true);

   SheathBase(const SheathBase &sb, bool realPart = true);

   virtual void SetRealPart() { realPart_ = true; }
   virtual void SetImaginaryPart() { realPart_ = false; }

   virtual void SetPotential(ParComplexGridFunction & potential)
   { potential_ = &potential; }

   double               EvalIonDensity(ElementTransformation &T,
                                       const IntegrationPoint &ip);
   double               EvalElectronTemp(ElementTransformation &T,
                                         const IntegrationPoint &ip);
   std::complex<double> EvalSheathPotential(ElementTransformation &T,
                                            const IntegrationPoint &ip);

protected:

   const BlockVector & density_;
   const BlockVector & temp_;
   ParComplexGridFunction * potential_;
   const ParFiniteElementSpace & L2FESpace_;
   const ParFiniteElementSpace & H1FESpace_;

   double omega_;
   bool realPart_;

   ParGridFunction density_gf_;
   ParGridFunction temperature_gf_;

   const Vector & charges_;
   const Vector & masses_;
};

class RectifiedSheathPotential : public SheathBase
{
public:
   RectifiedSheathPotential(const BlockVector & density,
                            const BlockVector & temp,
                            const ParFiniteElementSpace & L2FESpace,
                            const ParFiniteElementSpace & H1FESpace,
                            double omega,
                            const Vector & charges,
                            const Vector & masses,
                            bool realPart = true);

   RectifiedSheathPotential(const SheathBase &sb,
                            bool realPart = true)
      : SheathBase(sb, realPart)
   {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
private:
};

class SheathImpedance: public SheathBase
{
public:
   SheathImpedance(const ParGridFunction & B,
                   const BlockVector & density,
                   const BlockVector & temp,
                   const ParFiniteElementSpace & L2FESpace,
                   const ParFiniteElementSpace & H1FESpace,
                   double omega,
                   const Vector & charges,
                   const Vector & masses,
                   bool realPart = true);

   SheathImpedance(const SheathBase &sb,
                   const ParGridFunction & B,
                   bool realPart = true)
      : SheathBase(sb, realPart), B_(B) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   const ParGridFunction & B_;
};

class StixCoefBase
{
public:
   StixCoefBase(const ParGridFunction & B,
                const BlockVector & density,
                const BlockVector & temp,
                const ParFiniteElementSpace & L2FESpace,
                const ParFiniteElementSpace & H1FESpace,
                double omega,
                const Vector & charges,
                const Vector & masses,
                bool realPart = true);

   // Copy constructor
   StixCoefBase(StixCoefBase & s);

   void SetRealPart() { realPart_ = true; }
   void SetImaginaryPart() { realPart_ = false; }
   bool GetRealPartFlag() const { return realPart_; }

   void SetOmega(double omega) { omega_ = omega; }
   double GetOmega() const { return omega_; }

   const ParGridFunction & GetBField() const { return B_; }
   const BlockVector & GetDensityFields() const { return density_; }
   const BlockVector & GetTemperatureFields() const { return temp_; }
   const ParFiniteElementSpace & GetDensityFESpace() const
   { return L2FESpace_; }
   const ParFiniteElementSpace & GetTemperatureFESpace() const
   { return H1FESpace_; }
   const Vector & GetCharges() const { return charges_; }
   const Vector & GetMasses() const { return masses_; }

protected:
   double getBMagnitude(ElementTransformation &T,
                        const IntegrationPoint &ip);
   void   fillDensityVals(ElementTransformation &T,
                          const IntegrationPoint &ip);
   void   fillTemperatureVals(ElementTransformation &T,
                              const IntegrationPoint &ip);

   const ParGridFunction & B_;
   const BlockVector & density_;
   const BlockVector & temp_;
   const ParFiniteElementSpace & L2FESpace_;
   const ParFiniteElementSpace & H1FESpace_;

   double omega_;
   bool realPart_;

   mutable Vector BVec_;
   ParGridFunction density_gf_;
   ParGridFunction temperature_gf_;

   Vector density_vals_;
   Vector temp_vals_;
   const Vector & charges_;
   const Vector & masses_;
};

class StixSCoef: public Coefficient, public StixCoefBase
{
public:
   StixSCoef(const ParGridFunction & B,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             bool realPart = true);

   StixSCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixSCoef() {}
};

class StixDCoef: public Coefficient, public StixCoefBase
{
public:
   StixDCoef(const ParGridFunction & B,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             bool realPart = true);

   StixDCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixDCoef() {}
};

class StixPCoef: public Coefficient, public StixCoefBase
{
public:
   StixPCoef(const ParGridFunction & B,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             bool realPart = true);

   StixPCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixPCoef() {}
};

class StixTensorBase: public StixCoefBase
{
public:
   StixTensorBase(const ParGridFunction & B,
                  const BlockVector & density,
                  const BlockVector & temp,
                  const ParFiniteElementSpace & L2FESpace,
                  const ParFiniteElementSpace & H1FESpace,
                  double omega,
                  const Vector & charges,
                  const Vector & masses,
                  bool realPart = true);

   StixTensorBase(StixCoefBase &s) : StixCoefBase(s) {}

   virtual ~StixTensorBase() {}

protected:
   void addParallelComp(double P, DenseMatrix & eps);
   void addPerpDiagComp(double S, DenseMatrix & eps);
   void addPerpSkewComp(double D, DenseMatrix & eps);
};

class DielectricTensor: public MatrixCoefficient, public StixTensorBase
{
public:
   DielectricTensor(const ParGridFunction & B,
                    const BlockVector & density,
                    const BlockVector & temp,
                    const ParFiniteElementSpace & L2FESpace,
                    const ParFiniteElementSpace & H1FESpace,
                    double omega,
                    const Vector & charges,
                    const Vector & masses,
                    bool realPart = true);

   DielectricTensor(StixCoefBase &s)
      : MatrixCoefficient(3), StixTensorBase(s) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~DielectricTensor() {}
};

class InverseDielectricTensor: public MatrixCoefficient, public StixTensorBase
{
public:
   InverseDielectricTensor(const ParGridFunction & B,
                           const BlockVector & density,
                           const BlockVector & temp,
                           const ParFiniteElementSpace & L2FESpace,
                           const ParFiniteElementSpace & H1FESpace,
                           double omega,
                           const Vector & charges,
                           const Vector & masses,
                           bool realPart = true);

   InverseDielectricTensor(StixCoefBase &s)
      : MatrixCoefficient(3), StixTensorBase(s) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~InverseDielectricTensor() {}
};

class SPDDielectricTensor: public MatrixCoefficient, public StixCoefBase
{
public:
   SPDDielectricTensor(const ParGridFunction & B,
                       const BlockVector & density,
                       const BlockVector & temp,
                       const ParFiniteElementSpace & L2FESpace,
                       const ParFiniteElementSpace & H1FESpace,
                       double omega,
                       const Vector & charges,
                       const Vector & masses);

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~SPDDielectricTensor() {}
   /*
   private:
    const ParGridFunction & B_;
    const BlockVector & density_;
    const BlockVector & temp_;
    const ParFiniteElementSpace & L2FESpace_;
    const ParFiniteElementSpace & H1FESpace_;

    double omega_;

    ParGridFunction density_gf_;
    ParGridFunction temperature_gf_;

    Vector density_vals_;
    Vector temp_vals_;
    const Vector & charges_;
    const Vector & masses_;
   */
};

/*
   The different types of plasma profiles (i.e. temp, density) require
   different sets of parameters, for example.

   CONSTANT: 1 parameter
      the constant value of parameter

   GRADIENT: 7 parameters
      The value of the parameter at one point
      The location of this point (3 parameters)
      The gradient of the parameter at this point (3 parameters)

   TANH: 9 parameters
      The value of the parameter when tanh equals zero
      The value of the parameter when tanh equals one
      The skin depth, defined as the distance, in the direction of the
      steepest gradient, between locations where tanh equals zero and
      where tanh equals one-half.
      The location of a point where tanh equals zero (3 parameters)
      The unit vector in the direction of the steepest gradient away from
      the location described by the previous parameter (3 parameters)

   ELLIPTIC_COS: 7 parameters
      The value of the parameter when cos equals minus one
      The value of the parameter when cos equals one
      The radius of the ellipse in the x direction
      The radius of the ellipse in the y direction
      The center of the ellipse
*/
class PlasmaProfile : public Coefficient
{
public:
   enum Type {CONSTANT, GRADIENT, TANH, ELLIPTIC_COS};

private:
   Type type_;
   Vector p_;

   const int np_[4] = {1, 7, 9, 7};

   mutable Vector x_;

public:
   PlasmaProfile(Type type, const Vector & params);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_COEFS
