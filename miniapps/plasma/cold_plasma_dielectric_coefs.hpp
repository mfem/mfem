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

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

// Cyclotron frequency
inline double omega_c(double Bmag, double charge, double mass)
{
   return charge * q_ * Bmag / (mass * amu_);
}

// Plasma frequency
inline double omega_p(double number, double charge, double mass)
{
   return fabs(charge * q_) * sqrt(number / (epsilon0_ * mass * amu_));
}

double R_cold_plasma(double omega, double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass);

double L_cold_plasma(double omega, double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass);

double P_cold_plasma(double omega,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass);

double S_cold_plasma(double omega, double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass);

double D_cold_plasma(double omega, double Bmag,
                     const Vector & number,
                     const Vector & charge,
                     const Vector & mass);

class DielectricTensor: public MatrixCoefficient
{
public:
   DielectricTensor(const ParGridFunction & B,
                    const BlockVector & density,
                    const ParFiniteElementSpace & L2FESpace,
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
   const ParFiniteElementSpace & L2FESpace_;

   double omega_;
   bool realPart_;

   ParGridFunction density_gf_;

   Vector density_vals_;
   const Vector & charges_;
   const Vector & masses_;
};

class SPDDielectricTensor: public MatrixCoefficient
{
public:
   SPDDielectricTensor(const ParGridFunction & B,
                       const BlockVector & density,
                       const ParFiniteElementSpace & L2FESpace,
                       double omega,
                       const Vector & charges,
                       const Vector & masses);

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~SPDDielectricTensor() {}

private:
   const ParGridFunction & B_;
   const BlockVector & density_;
   const ParFiniteElementSpace & L2FESpace_;

   double omega_;

   ParGridFunction density_gf_;

   Vector density_vals_;
   const Vector & charges_;
   const Vector & masses_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_COEFS
