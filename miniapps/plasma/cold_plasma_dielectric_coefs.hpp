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
inline double nu_art(double x)
{
   // return (3e11*exp(-x/0.1));
   // return (1e9*exp(-(x-0.65)/0.04));
   return (1e14*exp(-x/0.1));
}

void StixCoefs_cold_plasma(Vector &V, double omega, double Bmag,
                           double nue, double nui,
                           const Vector & number,
                           const Vector & charge,
                           const Vector & mass,
                           const Vector & temp,
                           int nuprof,
                           bool realPart);

std::complex<double> R_cold_plasma(double omega, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   int nuprof);

std::complex<double> L_cold_plasma(double omega, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   int nuprof);

std::complex<double> P_cold_plasma(double omega, double nue,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   int nuprof);

std::complex<double> S_cold_plasma(double omega, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   int nuprof);

std::complex<double> D_cold_plasma(double omega, double Bmag,
                                   double nue, double nui,
                                   const Vector & number,
                                   const Vector & charge,
                                   const Vector & mass,
                                   const Vector & temp,
                                   int nuprof);

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
                            bool realPart);

   RectifiedSheathPotential(const SheathBase &sb,
                            bool realPart)
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

class StixCoefBase
{
public:
   StixCoefBase(const ParGridFunction & B,
                const ParGridFunction & nue,
                const ParGridFunction & nui,
                const BlockVector & density,
                const BlockVector & temp,
                const ParFiniteElementSpace & L2FESpace,
                const ParFiniteElementSpace & H1FESpace,
                double omega,
                const Vector & charges,
                const Vector & masses,
                int nuprof,
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

   const ParGridFunction & GetBField() const { return B_; }
   const ParGridFunction & GetNue() const { return nue_; }
   const ParGridFunction & GetNui() const { return nui_; }
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
   const ParGridFunction & nue_;
   const ParGridFunction & nui_;
   const BlockVector & density_;
   const BlockVector & temp_;
   const ParFiniteElementSpace & L2FESpace_;
   const ParFiniteElementSpace & H1FESpace_;

   double omega_;
   bool realPart_;
   int nuprof_;

   mutable Vector BVec_;
   ParGridFunction density_gf_;
   ParGridFunction temperature_gf_;

   Vector density_vals_;
   Vector temp_vals_;
   double nue_vals_;
   double nui_vals_;
   const Vector & charges_;
   const Vector & masses_;
};

class StixLCoef: public Coefficient, public StixCoefBase
{
public:
   StixLCoef(const ParGridFunction & B,
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
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
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
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
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
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
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
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
             const ParGridFunction & nue,
             const ParGridFunction & nui,
             const BlockVector & density,
             const BlockVector & temp,
             const ParFiniteElementSpace & L2FESpace,
             const ParFiniteElementSpace & H1FESpace,
             double omega,
             const Vector & charges,
             const Vector & masses,
             int nuprof,
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
                  const ParGridFunction & nue,
                  const ParGridFunction & nui,
                  const BlockVector & density,
                  const BlockVector & temp,
                  const ParFiniteElementSpace & L2FESpace,
                  const ParFiniteElementSpace & H1FESpace,
                  double omega,
                  const Vector & charges,
                  const Vector & masses,
                  int nuprof,
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
                    const ParGridFunction & nue,
                    const ParGridFunction & nui,
                    const BlockVector & density,
                    const BlockVector & temp,
                    const ParFiniteElementSpace & L2FESpace,
                    const ParFiniteElementSpace & H1FESpace,
                    double omega,
                    const Vector & charges,
                    const Vector & masses,
                    int nuprof,
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
                           const ParGridFunction & nue,
                           const ParGridFunction & nui,
                           const BlockVector & density,
                           const BlockVector & temp,
                           const ParFiniteElementSpace & L2FESpace,
                           const ParFiniteElementSpace & H1FESpace,
                           double omega,
                           const Vector & charges,
                           const Vector & masses,
                           int nuprof,
                           bool realPart);

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
                       const ParGridFunction & nue,
                       const ParGridFunction & nui,
                       const BlockVector & density,
                       const BlockVector & temp,
                       const ParFiniteElementSpace & L2FESpace,
                       const ParFiniteElementSpace & H1FESpace,
                       double omega,
                       const Vector & charges,
                       const Vector & masses,
                       int nuprof);

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

   POWER: 4 parameters
      The direction of the variation (comp)
      The value at x[comp]=0
      The value at x[comp]=1
      The power (exponent)
*/
class PlasmaProfile : public Coefficient
{
public:
   enum Type {CONSTANT, GRADIENT, TANH, ELLIPTIC_COS, PARABOLIC, PEDESTAL,
              NUABSORB, NUE, NUI, CMODDEN, CUSTOM1, CUSTOM2, POWER
             };

private:
   Type type_;
   Vector p_;

   const int np_[13] = {1, 7, 9, 7, 7, 7, 3, 2, 2, 1, 2, 2, 4};

   mutable Vector x_;

public:
   PlasmaProfile(Type type, const Vector & params);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

class BFieldProfile : public VectorCoefficient
{
public:
   enum Type {CONSTANT, B_P, B_TOPDOWN, B_P_KOHNO, B_EQDSK, B_WHAM};

private:
   Type type_;
   Vector p_;
   bool unit_;

   G_EQDSK_Data *eqdsk_;

   const int np_[6] = {3, 7, 6, 8, 4, 2};

   mutable Vector x3_;
   mutable Vector x_;

public:
   BFieldProfile(Type type, const Vector & params, bool unit,
                 G_EQDSK_Data *eqdsk = NULL);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ComplexPhaseCoefficient : public Coefficient
{
private:
   VectorCoefficient * kReCoef_;
   VectorCoefficient * kImCoef_;
   Coefficient * vReCoef_;
   Coefficient * vImCoef_;

   bool realPart_;
   bool inv_k_;
   int kdim_;

   mutable Vector xk_;
   mutable Vector xs_;
   mutable Vector kRe_;
   mutable Vector kIm_;

public:
   ComplexPhaseCoefficient(VectorCoefficient *kRe,
                           VectorCoefficient *kIm,
                           Coefficient *vRe,
                           Coefficient *vIm,
                           bool realPart, bool inv_k);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

};

class ComplexPhaseVectorCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * kReCoef_;
   VectorCoefficient * kImCoef_;
   VectorCoefficient * vReCoef_;
   VectorCoefficient * vImCoef_;

   bool realPart_;
   bool inv_k_;
   int kdim_;

   mutable Vector xk_;
   mutable Vector xs_;
   mutable Vector kRe_;
   mutable Vector kIm_;
   mutable Vector vRe_;
   mutable Vector vIm_;

public:
   ComplexPhaseVectorCoefficient(VectorCoefficient *kRe,
                                 VectorCoefficient *kIm,
                                 VectorCoefficient *vRe,
                                 VectorCoefficient *vIm,
                                 bool realPart, bool inv_k);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

};

class ScalarR2DCoef : public Coefficient
{
private:
   Coefficient & coef_;
   ParMesh & pmesh_;
public:
   ScalarR2DCoef(Coefficient & coef, ParMesh & pmesh)
      : coef_(coef), pmesh_(pmesh) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      int e = T.ElementNo;
      ElementTransformation * pT = pmesh_.GetElementTransformation(e);
      pT->SetIntPoint(&ip);

      return coef_.Eval(*pT, ip);
   }
};

class VectorR2DCoef : public VectorCoefficient
{
private:
   VectorCoefficient & coef_;
   ParMesh & pmesh_;
public:
   VectorR2DCoef(VectorCoefficient & coef, ParMesh & pmesh)
      : VectorCoefficient(coef.GetVDim()), coef_(coef), pmesh_(pmesh) {}

   void Eval(Vector &v, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      int e = T.ElementNo;
      ElementTransformation * pT = pmesh_.GetElementTransformation(e);
      pT->SetIntPoint(&ip);

      coef_.Eval(v, *pT, ip);
   }
};

class PseudoScalarCoef : public Coefficient
{
private:
   bool cyl_;

   Coefficient & coef_;

   mutable Vector x_;

public:
   PseudoScalarCoef(Coefficient & coef, bool cyl = false)
      : cyl_(cyl), coef_(coef), x_(2) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double val = coef_.Eval(T, ip);

      if (!cyl_)
      {
         return val;
      }
      else
      {
         T.Transform(ip, x_);
         if (x_[1] == 0.0) { return 0.0; }
         return val / x_[1];
      }
   }
};

class VectorXYCoef : public VectorCoefficient
{
private:
   bool cyl_;

   VectorCoefficient & coef_;

   mutable Vector x_;
   mutable Vector v3_;

public:
   VectorXYCoef(VectorCoefficient & coef, bool cyl = false)
      : VectorCoefficient(2), cyl_(cyl), coef_(coef), x_(2), v3_(3) {}

   void Eval(Vector &v, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      coef_.Eval(v3_, T, ip);
      v.SetSize(2);
      if (!cyl_)
      {
         v[0] = v3_[0]; v[1] = v3_[1];
      }
      else
      {
         T.Transform(ip, x_);
         if (x_[1] == 0.0)
         {
            v = 0.0;
         }
         else
         {
            v[0] = v3_[0] / x_[1]; v[1] = v3_[1] / x_[1];
         }
      }
   }
};

class VectorZCoef : public Coefficient
{
private:
   bool cyl_;

   VectorCoefficient & coef_;

   mutable Vector x_;
   mutable Vector v3_;

public:
   VectorZCoef(VectorCoefficient & coef, bool cyl = false)
      : cyl_(cyl), coef_(coef), x_(2), v3_(3) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      coef_.Eval(v3_, T, ip);

      if (!cyl_)
      {
         return v3_[2];
      }
      else
      {
         T.Transform(ip, x_);
         if (x_[1] == 0.0) { return 0.0; }
         return v3_[2] / x_[1];
      }
   }
};

class HCurlCylMassCoefficient : public MatrixCoefficient
{
private:
   const int rho_ind_;
   const int phi_ind_;
   const int zed_ind_;

   Coefficient       * SQ_;
   MatrixCoefficient * MQ_;

   mutable Vector x_;

public:
   HCurlCylMassCoefficient(int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(NULL), MQ_(NULL), x_(3) {}

   HCurlCylMassCoefficient(Coefficient & Q, int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(&Q), MQ_(NULL), x_(3) {}

   HCurlCylMassCoefficient(MatrixCoefficient & Q, int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(NULL), MQ_(&Q), x_(3) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      K.SetSize(3);

      if (MQ_)
      {
         MQ_->Eval(K, T, ip);
      }
      else
      {
         K.Diag(1.0, 3);

         if (SQ_)
         {
            K *= SQ_->Eval(T, ip);
         }
      }

      T.Transform(ip, x_);

      double rho = x_(rho_ind_);

      MFEM_VERIFY(rho >= 0.0, "Radial component must be non-negative.");

      K(rho_ind_, rho_ind_) *= rho;
      K(rho_ind_, zed_ind_) *= rho;
      K(zed_ind_, rho_ind_) *= rho;
      K(zed_ind_, zed_ind_) *= rho;

      K(phi_ind_, phi_ind_) *= (rho > 0.0) ? (1.0/rho) : 0.0;
   }
};

class HCurlCylStiffnessCoefficient : public MatrixCoefficient
{
private:
   const int rho_ind_;
   const int phi_ind_;
   const int zed_ind_;

   Coefficient       * SQ_;
   MatrixCoefficient * MQ_;

   mutable Vector x_;

public:
   HCurlCylStiffnessCoefficient(int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(NULL), MQ_(NULL), x_(3) {}

   HCurlCylStiffnessCoefficient(Coefficient & Q, int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(&Q), MQ_(NULL), x_(3) {}

   HCurlCylStiffnessCoefficient(MatrixCoefficient & Q, int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(NULL), MQ_(&Q), x_(3) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      K.SetSize(3);

      if (MQ_)
      {
         MQ_->Eval(K, T, ip);
      }
      else
      {
         K.Diag(1.0, 3);

         if (SQ_)
         {
            K *= SQ_->Eval(T, ip);
         }
      }

      T.Transform(ip, x_);

      double rho = x_(rho_ind_);

      MFEM_VERIFY(rho >= 0.0, "Radial component must be non-negative.");

      double a = (rho > 0.0) ? (1.0/rho) : 0.0;
      K(rho_ind_, rho_ind_) *= a;
      K(rho_ind_, zed_ind_) *= a;
      K(zed_ind_, rho_ind_) *= a;
      K(zed_ind_, zed_ind_) *= a;

      K(phi_ind_, phi_ind_) *= rho;
   }
};

class HCurlCylSourceCoefficient : public MatrixCoefficient
{
private:
   const int rho_ind_;
   const int phi_ind_;
   const int zed_ind_;

   Coefficient       * SQ_;
   MatrixCoefficient * MQ_;

   mutable Vector x_;

public:
   HCurlCylSourceCoefficient(int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(NULL), MQ_(NULL), x_(3) {}

   HCurlCylSourceCoefficient(Coefficient & Q, int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(&Q), MQ_(NULL), x_(3) {}

   HCurlCylSourceCoefficient(MatrixCoefficient & Q, int radial_comp = 1)
      : MatrixCoefficient(3),
        rho_ind_(radial_comp),
        phi_ind_((rho_ind_+1)%3),
        zed_ind_((rho_ind_+2)%3),
        SQ_(NULL), MQ_(&Q), x_(3) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      K.SetSize(3);

      if (MQ_)
      {
         MQ_->Eval(K, T, ip);
      }
      else
      {
         K.Diag(1.0, 3);

         if (SQ_)
         {
            K *= SQ_->Eval(T, ip);
         }
      }

      T.Transform(ip, x_);

      double rho = x_(rho_ind_);

      MFEM_VERIFY(rho >= 0.0, "Radial component must be non-negative.");

      K(rho_ind_, rho_ind_) *= rho;
      K(rho_ind_, phi_ind_) *= rho;
      K(rho_ind_, zed_ind_) *= rho;

      K(zed_ind_, rho_ind_) *= rho;
      K(zed_ind_, phi_ind_) *= rho;
      K(zed_ind_, zed_ind_) *= rho;
   }
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_COEFS
