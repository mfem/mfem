// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_COEFS
#define MFEM_COLD_PLASMA_DIELECTRIC_COEFS

#include "../common/pfem_extras.hpp"
#include "plasma.hpp"
#include "g_eqdsk_data.hpp"
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
   if (number < 0.0){number = -1.0*number;}
   return fabs(charge * q_) * 1.0 * sqrt(number / (epsilon0_ * mass * amu_));
}
inline std::complex<double> omega_p(double number, double charge,
                                    std::complex<double> mass)
{
   return fabs(charge * q_) * 1.0 * sqrt(number / (epsilon0_ * mass * amu_));
}

// Thermal Velocity
inline double vthermal(double Te   /* Joules */,
                       double mass   /* AMU */)
{
   return sqrt( (2.0 * Te) / ( mass * amu_ ) );
}

inline std::complex<double> vthermal(std::complex<double> Te   /* Joules */,
                                     std::complex<double> mass   /* AMU */)
{
   return sqrt( (2.0 * Te) / ( mass * amu_ ) );
}

// Coulomb logarithm
inline double CoulombLog(double n, double Te)
{
   double ntemp = n;
   if (n < 0) {ntemp = -1.0*n;}
   return log((4.0 * M_PI * pow(epsilon0_ * Te, 1.5)) / (pow(q_,
                                                             3) * sqrt(ntemp)));
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
inline double nu_art(double x)
{
   return (1e14*exp(-x/0.1));
}

void StixCoefs_cold_plasma(Vector &V, double omega, double Bmag,
                           double nue, double nui,
                           const Vector & number,
                           const Vector & charge,
                           const Vector & mass,
                           const Vector & temp,
                           double iontemp,
                           int nuprof,
                           bool realPart);

std::complex<double> R_cold_plasma(double omega, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   double iontemp,
                                   int nuprof);

std::complex<double> L_cold_plasma(double omega, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   double iontemp,
                                   int nuprof);

std::complex<double> P_cold_plasma(double omega,
                                   double kparallel, double nue,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   double iontemp,
                                   int nuprof);

std::complex<double> S_cold_plasma(double omega,
                                   double kparallel, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   double iontemp,
                                   int nuprof,
                                   double Rval,
                                   double Lval);

std::complex<double> D_cold_plasma(double omega,
                                   double kparallel, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   double iontemp,
                                   int nuprof,
                                   double Rval,
                                   double Lval);

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
              bool realPart);

   SheathBase(const SheathBase &sb, bool realPart = true);

   virtual void SetRealPart() { realPart_ = true; }
   virtual void SetImaginaryPart() { realPart_ = false; }

   virtual void SetPotential(ParComplexGridFunction & potential)
   { potential_ = &potential; }
   virtual void SetHiter(int * Hiter) { Hiter_ = Hiter; }
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
   int * Hiter_;

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
                            bool realPart);

   RectifiedSheathPotential(const SheathBase &sb,
                            bool realPart)
      : SheathBase(sb, realPart)
   {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
private:
};

class BFieldAngle : public SheathBase
{
public:
   BFieldAngle(const ParGridFunction & B,
               const BlockVector & density,
               const BlockVector & temp,
               const ParFiniteElementSpace & L2FESpace,
               const ParFiniteElementSpace & H1FESpace,
               double omega,
               const Vector & charges,
               const Vector & masses,
               bool realPart);

   BFieldAngle(const SheathBase &sb,
               const ParGridFunction & B,
               bool realPart)
      : SheathBase(sb, realPart), B_(B)
   {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
private:
   const ParGridFunction & B_;
};


class HTangential : public VectorCoefficient
{
public:
   HTangential(const ParGridFunction & H);

   void Eval(Vector &Ht,ElementTransformation &T,
             const IntegrationPoint &ip);
private:
   const ParGridFunction & H_;
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
                   bool realPart);

   SheathImpedance(const SheathBase &sb,
                   const ParGridFunction & B,
                   bool realPart)
      : SheathBase(sb, realPart), B_(B) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   const ParGridFunction & B_;
};

class SheathPower: public SheathBase
{
public:
   SheathPower(const ParGridFunction & B,
               const BlockVector & density,
               const BlockVector & temp,
               const ParFiniteElementSpace & L2FESpace,
               const ParFiniteElementSpace & H1FESpace,
               double omega,
               const Vector & charges,
               const Vector & masses,
               bool realPart);

   SheathPower(const SheathBase &sb,
               const ParGridFunction & B,
               bool realPart)
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
                const ParGridFunction & k,
                const ParGridFunction & nue,
                const ParGridFunction & nui,
                const BlockVector & density,
                const BlockVector & temp,
                const ParGridFunction & iontemp,
                const ParFiniteElementSpace & L2FESpace,
                const ParFiniteElementSpace & H1FESpace,
                double omega,
                const Vector & charges,
                const Vector & masses,
                int nuprof,
                double res_lim,
                bool realPart);

   // Copy constructor
   StixCoefBase(StixCoefBase & s);

   void SetRealPart() { realPart_ = true; }
   void SetImaginaryPart() { realPart_ = false; }
   bool GetRealPartFlag() const { return realPart_; }

   void SetOmega(double omega) { omega_ = omega; }
   double GetOmega() const { return omega_; }

   void SetNuProf(int nuprof) { nuprof_ = nuprof; }
   double GetNuProf() const { return nuprof_; }
   double GetResonanceLimitorFactor() const { return res_lim_; }

   const ParGridFunction & GetBField() const { return B_; }
   const ParGridFunction & GetKVec() const { return k_; }
   const ParGridFunction & GetNue() const { return nue_; }
   const ParGridFunction & GetNui() const { return nui_; }
   const ParGridFunction & GetTi() const { return iontemp_; }
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
   double getKvecMagnitude(ElementTransformation &T,
                           const IntegrationPoint &ip);
   void   fillDensityVals(ElementTransformation &T,
                          const IntegrationPoint &ip);
   void   fillTemperatureVals(ElementTransformation &T,
                              const IntegrationPoint &ip);

   const ParGridFunction & B_;
   const ParGridFunction & k_;
   const ParGridFunction & nue_;
   const ParGridFunction & nui_;
   const ParGridFunction & iontemp_;
   const BlockVector & density_;
   const BlockVector & temp_;
   const ParFiniteElementSpace & L2FESpace_;
   const ParFiniteElementSpace & H1FESpace_;

   double omega_;
   bool realPart_;
   int nuprof_;
   double res_lim_;

   mutable Vector BVec_;
   mutable Vector KVec_;
   ParGridFunction density_gf_;
   ParGridFunction temperature_gf_;

   Vector density_vals_;
   Vector temp_vals_;
   double nue_vals_;
   double nui_vals_;
   double Ti_vals_;
   const Vector & charges_;
   const Vector & masses_;
};

class StixLCoef: public Coefficient, public StixCoefBase
{
public:
   StixLCoef(const ParGridFunction & B,
             const ParGridFunction & k,
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParGridFunction & iontemp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
             double res_lim,
             bool realPart);

   StixLCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixLCoef() {}
};

class StixRCoef: public Coefficient, public StixCoefBase
{
public:
   StixRCoef(const ParGridFunction & B,
             const ParGridFunction & k,
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParGridFunction & iontemp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
             double res_lim,
             bool realPart);

   StixRCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixRCoef() {}
};

class StixSCoef: public Coefficient, public StixCoefBase
{
public:
   StixSCoef(const ParGridFunction & B,
             const ParGridFunction & k,
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParGridFunction & iontemp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
             double res_lim,
             bool realPart);

   StixSCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixSCoef() {}
};

class StixDCoef: public Coefficient, public StixCoefBase
{
public:
   StixDCoef(const ParGridFunction & B,
             const ParGridFunction & k,
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParGridFunction & iontemp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
             double res_lim,
             bool realPart);

   StixDCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixDCoef() {}
};

class StixPCoef: public Coefficient, public StixCoefBase
{
public:
   StixPCoef(const ParGridFunction & B,
             const ParGridFunction & k,
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParGridFunction & iontemp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
             double res_lim,
             bool realPart);

   StixPCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixPCoef() {}
};

class StixTensorBase: public StixCoefBase
{
public:
   StixTensorBase(const ParGridFunction & B,
                  const ParGridFunction & k,
                  const ParGridFunction & nue,
                  const ParGridFunction & nui,
                  const BlockVector & density,
                  const BlockVector & temp,
                  const ParGridFunction & iontemp,
                  const ParFiniteElementSpace & L2FESpace,
                  const ParFiniteElementSpace & H1FESpace,
                  double omega,
                  const Vector & charges,
                  const Vector & masses,
                  int nuprof,
                  double res_lim,
                  bool realPart);

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
                    const ParGridFunction & k,
                    const ParGridFunction & nue,
                    const ParGridFunction & nui,
                    const BlockVector & density,
                    const BlockVector & temp,
                    const ParGridFunction & iontemp,
                    const ParFiniteElementSpace & L2FESpace,
                    const ParFiniteElementSpace & H1FESpace,
                    double omega,
                    const Vector & charges,
                    const Vector & masses,
                    int nuprof,
                    double res_lim,
                    bool realPart);

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
                           const ParGridFunction & k,
                           const ParGridFunction & nue,
                           const ParGridFunction & nui,
                           const BlockVector & density,
                           const BlockVector & temp,
                           const ParGridFunction & iontemp,
                           const ParFiniteElementSpace & L2FESpace,
                           const ParFiniteElementSpace & H1FESpace,
                           double omega,
                           const Vector & charges,
                           const Vector & masses,
                           int nuprof,
                           double res_lim,
                           bool realPart);

   InverseDielectricTensor(StixCoefBase &s)
      : MatrixCoefficient(3), StixTensorBase(s) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~InverseDielectricTensor() {}
};

class SusceptibilityTensor: public MatrixCoefficient, public StixTensorBase
{
public:
   SusceptibilityTensor(const ParGridFunction & B,
                    const ParGridFunction & k,
                    const ParGridFunction & nue,
                    const ParGridFunction & nui,
                    const BlockVector & density,
                    const BlockVector & temp,
                    const ParGridFunction & iontemp,
                    const ParFiniteElementSpace & L2FESpace,
                    const ParFiniteElementSpace & H1FESpace,
                    double omega,
                    const Vector & charges,
                    const Vector & masses,
                    int nuprof,
                    double res_lim,
                    bool realPart);

   SusceptibilityTensor(StixCoefBase &s)
      : MatrixCoefficient(3), StixTensorBase(s) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~SusceptibilityTensor() {}
};

class SPDDielectricTensor: public MatrixCoefficient, public StixCoefBase
{
public:
   SPDDielectricTensor(const ParGridFunction & B,
                       const ParGridFunction & k,
                       const ParGridFunction & nue,
                       const ParGridFunction & nui,
                       const BlockVector & density,
                       const BlockVector & temp,
                       const ParGridFunction & iontemp,
                       const ParFiniteElementSpace & L2FESpace,
                       const ParFiniteElementSpace & H1FESpace,
                       double omega,
                       const Vector & charges,
                       const Vector & masses,
                       int nuprof,
                       double res_lim);

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
   /**  Currently the computational mesh is assumed to be 3D. However,
   the input mesh can be 1D or 2D. In these cases the input mesh
   is extruded to create a 3D mesh. By default this extrusion is
   performed in directions perpendicular to the input mesh.

        In the 2D codes the extrusion can be performed such that
   cylindrical symmetry can be imposed. In this special case the
   input mesh is assumed to represent a poloidal cross section
   with the x axis of the input mesh aligned with the radial
   coordinate and the y axis of the input mesh aligned with the z
   coordinate of the final mesh. Setting the CoordSystem to
   POLOIDAL should be done in this special case.
   */
   enum CoordSystem {CARTESIAN_3D, POLOIDAL};
   enum Type {CONSTANT               =  0,
              GRADIENT               =  1,
              TANH                   =  2,
              ELLIPTIC_COS           =  3,
              PARABOLIC              =  4,
              PEDESTAL               =  5,
              NUABSORB               =  6,
              NUE                    =  7,
              NUI                    =  8,
              CMODDEN                =  9,
              POLOIDAL_H_MODE_DEN    = 10,
              POLOIDAL_H_MODE_TEMP   = 11,
              POLOIDAL_CORE          = 12,
              POLOIDAL_SOL           = 13,
              POLOIDAL_MIN_TEMP      = 14
             };

private:

   struct ProfileParams
   {
      Type type;
      Vector params;
   };

   std::vector<ProfileParams> params_;
   std::map<int, int> paramsByAttr_;

   // Type temp_type_;
   // Vector temp_p_;
   bool cyl_; // Assume cylindrical symmetyry

   G_EQDSK_Data *eqdsk_;

   const int np_[15] = {1, 7, 9, 7, 4, 7, 3, 3, 3, 1, 1, 1, 8, 5, 2};

   mutable Vector xyz_; // 3D coordinate in computational mesh
   mutable Vector rz_;  // 2D coordinate in poloidal cross section

   double EvalByType(Type type,
                     const Vector &params,
                     ElementTransformation &T,
                     const IntegrationPoint &ip);

public:
   PlasmaProfile(Type type, const Vector & params,
                 CoordSystem coord_sys = CARTESIAN_3D,
                 G_EQDSK_Data *eqdsk = NULL);

   // Set default profile type and parameters
   // (replaces defaults passed to the constructor)
   void SetParams(Type type, const Vector &params);

   // Set profile type and parameters to be used
   // with element attributes listed in "attr"
   void SetParams(const Array<int> attr, Type type, const Vector &params);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

class BFieldProfile : public VectorCoefficient
{
public:
   /** See PlasmaProfile for documentation. */
   enum CoordSystem {CARTESIAN_3D, POLOIDAL};
   enum Type {CONSTANT         = 0,
              B_P1             = 1,
              B_P2             = 2,
              B_P_KOHNO        = 3,
              B_EQDSK_TOPDOWN  = 4,
              B_EQDSK_POLOIDAL = 5,
              B_WHAM           = 6
             };

private:
   Type type_;
   Vector p_;
   bool cyl_; // Assume cylindrical symmetyry
   bool unit_;

   G_EQDSK_Data *eqdsk_;

   const int np_[7] = {3, 7, 6, 8, 4, 1, 2};

   // mutable Vector x3_; // Not currently used
   mutable Vector xyz_; // 3D coordinate in computational mesh
   mutable Vector rz_;  // 2D coordinate in poloidal cross section

public:
   BFieldProfile(Type type, const Vector & params, bool unit,
                 CoordSystem coord_sys = CARTESIAN_3D,
                 G_EQDSK_Data *eqdsk = NULL);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class StixFrame : public VectorCoefficient
{
public:
   enum CoordSystem {CARTESIAN_3D, POLOIDAL};

private:
   bool cyl_; // Assume cylindrical symmetyry

   mutable Vector xyz_; // 3D coordinate in computational mesh
   mutable Vector rz_;  // 2D coordinate in poloidal cross section
   mutable Vector BUnitVec_; // 3D BField unit vector
   const ParGridFunction & B_;
   bool xdir_;

public:
   StixFrame(const ParGridFunction & B, bool xdir,
             CoordSystem coord_sys = CARTESIAN_3D);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_COEFS