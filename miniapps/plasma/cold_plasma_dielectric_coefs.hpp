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
#include <complex>

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

// Cyclotron frequency
inline std::complex<double> omega_c(double Bmag, double charge,
                                    std::complex<double> mass)
{
   return (charge * q_ * Bmag) / (mass * amu_);
}

// Plasma frequency
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

class DielectricTensor: public MatrixCoefficient
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

   void SetRealPart() { realPart_ = true; }
   void SetImaginaryPart() { realPart_ = false; }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
   // virtual void Dval(DenseMatrix &K, ElementTransformation &T,
   //                   const IntegrationPoint &ip);
   virtual ~DielectricTensor() {}

private:
   const ParGridFunction & B_;
   const BlockVector & density_;
   const BlockVector & temp_;
   const ParFiniteElementSpace & L2FESpace_;
   const ParFiniteElementSpace & H1FESpace_;

   double omega_;
   bool realPart_;

   ParGridFunction density_gf_;
   ParGridFunction temperature_gf_;

   Vector density_vals_;
   Vector temp_vals_;
   const Vector & charges_;
   const Vector & masses_;
};

class SPDDielectricTensor: public MatrixCoefficient
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
