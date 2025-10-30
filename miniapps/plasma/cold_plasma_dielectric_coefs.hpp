// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include <complex>

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

// Cyclotron frequency (Radians / second)
inline complex_t omega_c(real_t Bmag             /* Tesla */,
                         real_t charge /* electron charge */,
                         complex_t mass /* AMU */)
{
   return (charge * q_ * Bmag) / (mass * amu_);
}

// Plasma frequency (Radians / second)
inline complex_t omega_p(real_t number /* particles / m^3 */,
                         real_t charge /* electron charge */,
                         complex_t mass /* AMU */)
{
   return abs(charge * q_) * sqrt(number / (epsilon0_ * mass * amu_));
}

// Coulomb logarithm
inline real_t CoulombLog(real_t n, real_t Te)
{
   return log((4.0 * M_PI * pow(epsilon0_ * Te, 1.5)) / (pow(q_, 3) * sqrt(n)));
}

// Collisional frequency between electrons and ions
inline real_t nu_ei(real_t charge, real_t coul_log, real_t mass,
                    real_t Te, real_t number)
{
   return (8.0 * number * M_PI * pow(charge * q_, 4) * coul_log) /
          (3.0 * sqrt(2.0 * M_PI * me_kg_) * pow(4.0 * M_PI * epsilon0_, 2)
           * pow(Te, 1.5));
}

complex_t R_cold_plasma(real_t omega, real_t Bmag,
                        real_t nue, real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof);

complex_t L_cold_plasma(real_t omega, real_t Bmag,
                        real_t nue, real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        real_t res_lim);

complex_t P_cold_plasma(real_t omega, real_t nue,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof);

complex_t S_cold_plasma(real_t omega, real_t Bmag,
                        real_t nue, real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        real_t res_lim);

complex_t D_cold_plasma(real_t omega, real_t Bmag,
                        real_t nue, real_t nui,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        real_t res_lim);

struct StixParams
{
   StixParams(VectorCoefficient &B,
              Coefficient &nue,
              Coefficient &nui,
              VectorCoefficient &spec_densities,
              VectorCoefficient &spec_temperatures,
              real_t omega_val,
              const Vector & spec_charges,
              const Vector & spec_masses,
              int nuprof,
              real_t res_lim)
      : BCoef(B), nueCoef(nue), nuiCoef(nui),
        specDensityCoef(spec_densities),
        specTemperatureCoef(spec_temperatures),
        omega(omega_val),
        charges(spec_charges),
        masses(spec_masses),
        nuProfile(nuprof),
        resLimit(res_lim)
   {}

   VectorCoefficient &BCoef;
   Coefficient &nueCoef;
   Coefficient &nuiCoef;
   VectorCoefficient &specDensityCoef;
   VectorCoefficient &specTemperatureCoef;
   real_t omega;
   const Vector &charges;
   const Vector &masses;
   int nuProfile;
   real_t resLimit;
};

class StixCoefBase
{
public:
   enum ReImPart {REAL_PART,
                  IMAG_PART,
                  IMAGINARY_PART /* Reduntant but fault tolerant */
                 };

   StixCoefBase(StixParams &stix_params,
                ReImPart re_im_part);

   // Copy constructor
   StixCoefBase(StixCoefBase & s);

   void SetRealPart() { re_im_part_ = REAL_PART; }
   void SetImaginaryPart() { re_im_part_ = IMAG_PART; }
   ReImPart GetRealPartFlag() const { return re_im_part_; }

   void SetOmega(real_t omega) { omega_ = omega; }
   real_t GetOmega() const { return omega_; }

   void SetNuProf(int nuprof) { nuprof_ = nuprof; }
   real_t GetNuProf() const { return nuprof_; }
   real_t GetResonanceLimitorFactor() const { return res_lim_; }

   VectorCoefficient & GetBCoef() const { return BCoef_; }
   Coefficient & GetNue() const { return nue_; }
   Coefficient & GetNui() const { return nui_; }
   VectorCoefficient & GetDensityCoefs() const { return spec_density_; }
   VectorCoefficient & GetTemperatureCoefs() const { return spec_temperature_; }

   const Vector & GetCharges() const { return charges_; }
   const Vector & GetMasses() const { return masses_; }

protected:
   real_t getBMagnitude(ElementTransformation &T,
                        const IntegrationPoint &ip);
   void   fillDensityVals(ElementTransformation &T,
                          const IntegrationPoint &ip);
   void   fillTemperatureVals(ElementTransformation &T,
                              const IntegrationPoint &ip);

   VectorCoefficient &BCoef_;
   Coefficient &nue_;
   Coefficient &nui_;
   VectorCoefficient &spec_density_;
   VectorCoefficient &spec_temperature_;

   real_t omega_;
   ReImPart re_im_part_;
   int nuprof_;
   real_t res_lim_;

   mutable Vector BVec_;

   Vector density_vals_;
   Vector temperature_vals_;
   real_t nue_vals_;
   real_t nui_vals_;
   const Vector & charges_;
   const Vector & masses_;
};

typedef StixCoefBase StixCoef;

class StixLCoef: public Coefficient, public StixCoefBase
{
public:
   StixLCoef(StixParams &stix_params, ReImPart re_im_part);

   StixLCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixLCoef() {}
};

class StixRCoef: public Coefficient, public StixCoefBase
{
public:
   StixRCoef(StixParams &stix_params, ReImPart re_im_part);

   StixRCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixRCoef() {}
};

class StixSCoef: public Coefficient, public StixCoefBase
{
public:
   StixSCoef(StixParams &stix_params, ReImPart re_im_part);

   StixSCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixSCoef() {}
};

class StixDCoef: public Coefficient, public StixCoefBase
{
public:
   StixDCoef(StixParams &stix_params, ReImPart re_im_part);

   StixDCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixDCoef() {}
};

class StixPCoef: public Coefficient, public StixCoefBase
{
public:
   StixPCoef(StixParams &stix_params, ReImPart re_im_part);

   StixPCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixPCoef() {}
};

class StixInvSPCoef: public Coefficient, public StixCoefBase
{
public:
   StixInvSPCoef(StixParams &stix_params, ReImPart re_im_part)
      : StixCoefBase(stix_params, re_im_part)
   {}

   StixInvSPCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);

   virtual ~StixInvSPCoef() {}
};

class StixWaveLengthCoef: public Coefficient, public StixCoefBase
{
private:
   char type_;

public:
   StixWaveLengthCoef(char type,
                      StixParams &stix_params,
                      ReImPart re_im);

   StixWaveLengthCoef(StixCoefBase &s) : StixCoefBase(s) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
   virtual ~StixWaveLengthCoef() {}
};

class StixAdmittanceCoef: public Coefficient, public StixCoefBase
{
private:
   char type_;

public:
   StixAdmittanceCoef(char type, StixParams & stix_params, ReImPart re_im_part);

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

class StixTensorBase: public StixCoefBase
{
public:
   StixTensorBase(StixParams &stix_params, ReImPart re_im_part);

   StixTensorBase(StixCoefBase &s) : StixCoefBase(s) {}

   virtual ~StixTensorBase() {}

protected:
   void addParallelComp(real_t P, DenseMatrix & eps);
   void addPerpDiagComp(real_t S, DenseMatrix & eps);
   void addPerpSkewComp(real_t D, DenseMatrix & eps);
};

class DielectricTensor: public MatrixCoefficient, public StixTensorBase
{
public:
   DielectricTensor(StixParams &stix_params, ReImPart re_im_part);

   DielectricTensor(StixCoefBase &s)
      : MatrixCoefficient(3), StixTensorBase(s) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~DielectricTensor() {}
};

class InverseDielectricTensor: public MatrixCoefficient, public StixTensorBase
{
public:
   InverseDielectricTensor(StixParams &stix_params, ReImPart re_im_part);

   InverseDielectricTensor(StixCoefBase &s)
      : MatrixCoefficient(3), StixTensorBase(s) {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~InverseDielectricTensor() {}
};

class SPDDielectricTensor: public MatrixCoefficient, public StixCoefBase
{
public:
   SPDDielectricTensor(StixParams &stix_params);

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~SPDDielectricTensor() {}
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
   enum Type { AVERAGE      = -2,
               NEUTRALITY   = -1,
               CONSTANT     =  0,
               GRADIENT     =  1,
               TANH         =  2,
               ELLIPTIC_COS =  3,
               PARABOLIC    =  4,
               PEDESTAL     =  5,
               NUABSORB     =  6,
               NUE          =  7,
               NUI          =  8,
               CMODDEN      =  9,
               SPARC_RES    = 10,
               SPARC_DEN    = 11,
               CUSTOM1      = 12,
               CUSTOM2      = 13,
               POWER        = 14,
               WHAM         = 15
             };

private:
   Type type_;
   Vector p_;

   static constexpr int np_[16] = {1, 7, 9, 7,
                                   7, 7, 3, 2,
                                   3, 1, 1, 1,
                                   2, 2, 4, 7
                                  };

   mutable Vector x_;

public:
   PlasmaProfile() = default;
   PlasmaProfile(Type type, const Vector & params);

   // Special Constructor for profile type AVERAGE
   PlasmaProfile(Type type, const Array<int> &types, const Vector &params);

   // Special Constructor for profile type NEUTRALITY
   PlasmaProfile(Type type, const Array<int> &types, const Vector &params,
                 const Vector &charges);

   void Init(Type type, const Vector & params);

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

   static int GetNumParams(Type type) { return np_[type]; }
};


class MultiSpeciesPlasmaProfiles : public VectorCoefficient
{
private:
   std::vector<PlasmaProfile> prof_;
   int electron_profile_type_;
   Vector ion_spec_coefs_;

public:
   MultiSpeciesPlasmaProfiles(Array<PlasmaProfile::Type> types,
                              const Vector & params);

   MultiSpeciesPlasmaProfiles(Array<PlasmaProfile::Type> types,
                              const Vector & params, const Vector & charges);

   void Eval(Vector &p, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class BFieldProfile : public VectorCoefficient
{
public:
   enum Type {CONSTANT   = 0
             };

private:
   Type type_;
   Vector p_;
   bool unit_;

   static constexpr int np_[1] = {3};

   mutable Vector x3_;
   mutable Vector x_;

public:
   BFieldProfile(Type type, const Vector & params, bool unit);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
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

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      real_t val = coef_.Eval(T, ip);

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

class ColdPlasmaPlaneWaveBase: public StixCoefBase, public VectorCoefficient
{
protected:
   char type_;
   real_t Bmag_;
   complex_t kappa_;
   Vector b_;   // Normalized vector in direction of B
   Vector bc_;  // Normalized vector perpendicular to b_, (by-bz,bz-bx,bx-by)
   Vector bcc_; // Normalized vector perpendicular to b_ and bc_
   Vector k_dir_;
   Vector kxb_;
   Vector k_r_;
   Vector k_i_;
   Vector beta_r_;
   Vector beta_i_;

   complex_t S_;
   complex_t D_;
   complex_t P_;

protected:
   ColdPlasmaPlaneWaveBase(Vector k_dir, char type,
                           StixParams & stix_params,
                           ReImPart re_im_part);

   void ComputeFieldAxes(ElementTransformation &T,
                         const IntegrationPoint &ip);

   void ComputeStixCoefs();

   void ComputeWaveNumber();

   void ComputeWaveVector();

public:
   void SetPhaseShift(const Vector & beta)
   { beta_r_ = beta; beta_i_ = 0.0; }

   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { beta_r_ = beta_r; beta_i_ = beta_i; }

   void ClearPhaseShift()
   { beta_r_ = 0.0; beta_i_ = 0.0; }

   void InvertPhaseShift()
   { beta_r_ *= -1.0; beta_i_ *= -1.0; }

   void GetWaveVector(Vector & k_r, Vector & k_i) const
   { k_r = k_r_; k_i = k_i_; }
};

class ColdPlasmaPlaneWaveE: public ColdPlasmaPlaneWaveBase
{
public:
   ColdPlasmaPlaneWaveE(Vector k_dir, char type,
                        StixParams & stix_params,
                        ReImPart re_im_part);

   void ComputePolarizationVectorE();

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

protected:
   Vector e_r_;
   Vector e_i_;
};

class ColdPlasmaPlaneWaveH: public ColdPlasmaPlaneWaveE
{
public:
   ColdPlasmaPlaneWaveH(Vector k_dir, char type,
                        StixParams & stix_params,
                        ReImPart re_im_part);

   void ComputePolarizationVectorH();

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

protected:
   Vector h_r_;
   Vector h_i_;
   mutable Vector v_tmp_;
};

class ColdPlasmaCenterFeedBase
{
protected:
   ColdPlasmaCenterFeedBase(real_t j_pos, real_t j_dx, int j_profile);

   static complex_t sinc(complex_t z)
   { return (abs(z) > 1e-6)? (sin(z)/z) : 1.0; }

   real_t j_pos_;
   real_t j_dx_;
   int j_prof_;
};

class ColdPlasmaCenterFeedE: public ColdPlasmaPlaneWaveE,
   public ColdPlasmaCenterFeedBase
{
public:
   ColdPlasmaCenterFeedE(Vector k_dir, char type,
                         real_t j_pos, real_t j_dx, int j_profile,
                         StixParams & stix_params,
                         ReImPart re_im_part);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ColdPlasmaCenterFeedH: public ColdPlasmaPlaneWaveH,
   public ColdPlasmaCenterFeedBase
{
public:
   ColdPlasmaCenterFeedH(Vector k_dir, char type,
                         real_t j_pos, real_t j_dx, int j_profile,
                         StixParams & stix_params,
                         ReImPart re_im_part);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ColdPlasmaCenterFeedJ: public ColdPlasmaCenterFeedH
{
public:
   ColdPlasmaCenterFeedJ(Vector k_dir, char type,
                         real_t j_pos, real_t j_dx, int j_profile,
                         StixParams & stix_params,
                         ReImPart re_im_part);

   void ComputePolarizationVectorJ();

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

protected:
   Vector j_r_;
   Vector j_i_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_COEFS
