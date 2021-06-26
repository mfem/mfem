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

#ifndef MFEM_PLASMA_HPP
#define MFEM_PLASMA_HPP

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../mfem.hpp"
#include "../../general/text.hpp"

namespace mfem
{

namespace plasma
{

// Physical Constants

// Permittivity of Free Space (units F/m)
static const double epsilon0_ = 8.8541878176e-12;

// Permeability of Free Space (units H/m)
static const double mu0_ = 4.0e-7 * M_PI;

// Speed of light in Free Space (units m/s)
static const double c0_ = 1.0 / sqrt(epsilon0_ * mu0_);

static const double q_     = 1.602176634e-19; // Elementary charge in coulombs
static const double eV_    = 1.602176634e-19; // 1 eV in Joules
static const double amu_   = 1.660539040e-27; // Atomic mass unit in kilograms
static const double me_kg_ = 9.10938356e-31;  // Mass of electron in kilograms
static const double me_u_  = 5.4857990907e-4; // Mass of electron in a.m.u

/**
   Returns the cyclotron frequency in radians/second
   m is the mass in a.m.u
   q is the charge in units of elementary electric charge
   B is the magnetic field magnitude in tesla
 */
inline double cyclotronFrequency(double B, double m, double q)
{
   return fabs(q * q_ * B / (m * amu_));
}

class G_EQDSK_Data
{
public:
   G_EQDSK_Data(std::istream &is);

   void PrintInfo(std::ostream & out = std::cout) const;
   void DumpGnuPlotData(const std::string &file) const;

   double InterpPsi(const Vector &x);
   void InterpNxGradPsi(const Vector &x, Vector &b);

private:
   void initInterpolation();

   std::vector<std::string> CASE_; // Identification character string

   int NW_; // Number of horizontal R grid points
   int NH_; // Number of vertical Z grid points

   double RDIM_;    // Horizontal dimension in meter of computational box
   double ZDIM_;    // Vertical dimension in meter of computational box
   double RLEFT_;   // Minimum R in meter of rectangular computational box
   double ZMID_;    // Z of center of computational box in meter
   double RMAXIS_;  // R of magnetic axis in meter
   double ZMAXIS_;  // Z of magnetic axis in meter
   double SIMAG_;   // poloidal flux at magnetic axis in Weber /rad
   double SIBRY_;   // poloidal flux at the plasma boundary in Weber /rad
   double RCENTR_;  // R in meter of vacuum toroidal magnetic field BCENTR
   double BCENTR_;  // Vacuum toroidal magnetic field in Tesla at RCENTR
   double CURRENT_; // Plasma current in Ampere

   // Poloidal current function in m-T, F = RBT on flux grid
   std::vector<double> FPOL_;

   // Plasma pressure in nt / m^2 on uniform flux grid
   std::vector<double> PRES_;

   // FF’(ψ) in (mT)2 / (Weber /rad) on uniform flux grid
   std::vector<double> FFPRIM_;

   // P’(ψ) in (nt /m2) / (Weber /rad) on uniform flux grid
   std::vector<double> PPRIME_;

   // Poloidal flux in Weber / rad on the rectangular grid points
   std::vector<double> PSIRZ_;

   // q values on uniform flux grid from axis to boundary
   std::vector<double> QPSI_;

   int                 NBBBS_;  // Number of boundary points
   std::vector<double> RBBBS_;  // R of boundary points in meter
   std::vector<double> ZBBBS_;  // Z of boundary points in meter

   int                 LIMITR_; // Number of limiter points
   std::vector<double> RLIM_;   // R of surrounding limiter contour in meter
   std::vector<double> ZLIM_;   // Z of surrounding limiter contour in meter

   // Divided differences for Akima's interpolation method
   double dr_, dz_;
   DenseMatrix c_;
   DenseMatrix d_;
   DenseMatrix e_;
};

class G_EQDSK_Psi_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_Psi_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpPsi(transip);
   }

};

class G_EQDSK_NxGradPsi_Coefficient : public VectorCoefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_NxGradPsi_Coefficient(G_EQDSK_Data &g_eqdsk)
      : VectorCoefficient(2), eqdsk(g_eqdsk) {}

   void Eval(Vector &b, ElementTransformation & T,
             const IntegrationPoint & ip)
   {
      double x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      eqdsk.InterpNxGradPsi(transip, b);
   }

};

} // namespace plasma

} // namespace mfem

#endif // MFEM_PLASMA_HPP

