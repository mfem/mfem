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

#ifndef MFEM_G_EQDSK_DATA_HPP
#define MFEM_G_EQDSK_DATA_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mfem.hpp"
#include "../../general/text.hpp"

namespace mfem
{

namespace plasma
{

class G_EQDSK_Data
{
public:
   G_EQDSK_Data(std::istream &is, int logging = 0);

   // Number of points in radial direction
   int GetNumPtsR() const { return NW_; }

   // Number of points in z direction
   int GetNumPtsZ() const { return NH_; }

   // Width of domain in radial dimension (in meters)
   real_t GetRExtent() const { return RDIM_; }

   // Height of domain in z dimension (in meters)
   real_t GetZExtent() const { return ZDIM_; }

   // Radial coordinate at innermost edge of domain (in meters)
   real_t GetRMin() const { return RLEFT_; }

   // Z coordinate of the middle of the domain (in meters)
   real_t GetZMid() const { return ZMID_; }

   // R coordinate of the magnetic axis (in meters)
   real_t GetRMagAxis() const { return RMAXIS_; }

   // Z coordinate of the magnetic axis (in meters)
   real_t GetZMagAxis() const { return ZMAXIS_; }

   // Value of poloidal flux at the magnetic axis (in Weber / rad)
   real_t GetPsiMagAxis() const {return SIMAG_; }

   // Value of poloidal flux at the plasma boundary (in Weber / rad)
   real_t GetPsiBdry() const {return SIBRY_; }

   // Value of plasma current (in Ampere)
   real_t GetPlasmaCurrent() const {return CURRENT_; }

   // Values of poloidal flux (in Weber / rad) on the full grid in a
   // flattened array with z-direction cycling the fastest
   std::vector<real_t> & GetPsi() { return PSIRZ_ ;}

   // Print a text block to the output stream containing basic
   // information about the domain and the fields defined in the eqdsk
   // file.
   void PrintInfo(std::ostream &out = std::cout) const;

   // Create a GnuPlot input file and associated data file for
   // visualizing the fields stored in the eqdsk file.
   void DumpGnuPlotData(const std::string &file) const;

   // In the following interpolation functions the Vector argument rz
   // is a two component vector containing first the radial coordinate
   // and nex the z coordinate both expressed in meters.

   // Interpolate the toroidal field function, F(Psi(rz) / SIMAG)
   // (in Tesla meters), at the point rz
   real_t InterpFPolRZ(const Vector &rz);

   // Interpolate the pressure, P(Psi(rz) / SIMAG) (in N / m^2), at the
   // point rz
   real_t InterpPresRZ(const Vector &rz);

   // Interpolate the function, F(Psi(rz) / SIMAG) * F'(Psi / SIMAG)
   // (in (m T)^2 / (Weber / rad)), at the point rz
   real_t InterpFFPrimeRZ(const Vector &rz);

   // Interpolate the function, P'(Psi(rz) / SIMAG)
   // (in (N / m^2) / (Weber / rad)), at the point rz
   real_t InterpPPrimeRZ(const Vector &rz);

   // Interpolate the poloidal flux function, Psi(rz) (in Weber / rad), at
   // the point rz
   real_t InterpPsiRZ(const Vector &rz);

   // Interpolate the safety factor, q(Psi(rz) / SIMAG), at the point rz
   real_t InterpQRZ(const Vector &rz);

   // Interpolate the toroidal magnetic fleid (in Tesla) at the
   // point rz
   //    B_T = F(Psi(rz) / SIMAG) / r
   real_t InterpBTorRZ(const Vector &rz);

   // Interpolate the toroidal current density (in Ampere / m^2) at
   // the point rz
   //    J_T = r P'((Psi(rz) / SIMAG) + FF'(Psi(rz) / SIMAG) / (r mu0)
   real_t InterpJTorRZ(const Vector &rz);

   // Interpolate the rotated gradient of Psi (in Tesla) at the
   // point rz
   //    nxdp = (n x Grad Psi(rz))
   // where n is the unit vector in the toroidal direction
   void InterpNxGradPsiRZ(const Vector &rz, Vector &nxdp);

   // Interpolate the poloidal magnetic field (in Tesla) at the
   // point rz
   //    B_P = (n x Grad Psi(rz)) / r
   // where n is the unit vector in the toroidal direction
   void InterpBPolRZ(const Vector &rz, Vector &b);

   int GetNumBoundaryPts() const { return NBBBS_; }
   const std::vector<real_t> & GetBoundaryRVals() const { return RBBBS_; }
   const std::vector<real_t> & GetBoundaryZVals() const { return ZBBBS_; }

   int GetNumLimiterPts() const { return LIMITR_; }
   const std::vector<real_t> & GetLimiterRVals() const { return RLIM_; }
   const std::vector<real_t> & GetLimiterZVals() const { return ZLIM_; }

private:
   class ShiftedVector;
   class ShiftedDenseMatrix;
   class ExtendedDenseMatrix;

   enum FieldType {FPOL, PRES, FFPRIM, PPRIME, PSIRZ, QPSI/*, BTOR*/};

   int logging_;
   int init_flag_;
   inline bool checkFlag(int flag) { return (init_flag_ >> flag) & 1; }
   inline void   setFlag(int flag) { init_flag_ |= (1 << flag); }
   inline void clearFlag(int flag) { init_flag_ &= ~(1 << flag); }

   void checkPsiBoundary();

   void initInterpPsi(const std::vector<real_t> &v,
                      std::vector<real_t> &t);
   void initInterpRZ(const std::vector<real_t> &v,
                     ShiftedDenseMatrix &c,
                     ShiftedDenseMatrix &d,
                     ShiftedDenseMatrix &e);

   real_t interpRZ(const Vector &rz,
                   const std::vector<real_t> &v,
                   const ShiftedDenseMatrix &c,
                   const ShiftedDenseMatrix &d,
                   const ShiftedDenseMatrix &e);
   void interpNxGradRZ(const Vector &rz,
                       const std::vector<real_t> &v,
                       const ShiftedDenseMatrix &c,
                       const ShiftedDenseMatrix &d,
                       const ShiftedDenseMatrix &e,
                       Vector &b);
   real_t interpPsi(real_t psi, const std::vector<real_t> &v,
                    const std::vector<real_t> &t);

   /// The following variable names are taken from the C-Mod Wiki at
   /// https://cmodwiki.psfc.mit.edu/index.php/G_EQDSK

   std::vector<std::string> CASE_; // Identification character string

   int NW_; // Number of horizontal R grid points
   int NH_; // Number of vertical Z grid points

   real_t RDIM_;    // Horizontal dimension in meter of computational box
   real_t ZDIM_;    // Vertical dimension in meter of computational box
   real_t RLEFT_;   // Minimum R in meter of rectangular computational box
   real_t ZMID_;    // Z of center of computational box in meter
   real_t RMAXIS_;  // R of magnetic axis in meter
   real_t ZMAXIS_;  // Z of magnetic axis in meter
   real_t SIMAG_;   // poloidal flux at magnetic axis in Weber /rad
   real_t SIBRY_;   // poloidal flux at the plasma boundary in Weber /rad
   real_t RCENTR_;  // R in meter of vacuum toroidal magnetic field BCENTR
   real_t BCENTR_;  // Vacuum toroidal magnetic field in Tesla at RCENTR
   real_t CURRENT_; // Plasma current in Ampere

   // Poloidal current function in m-T, F = RBT on flux grid
   std::vector<real_t> FPOL_;

   // Plasma pressure in nt / m^2 on uniform flux grid
   std::vector<real_t> PRES_;

   // FF’(ψ) in (mT)^2 / (Weber /rad) on uniform flux grid
   std::vector<real_t> FFPRIM_;

   // P’(ψ) in (nt /m^2) / (Weber /rad) on uniform flux grid
   std::vector<real_t> PPRIME_;

   // Poloidal flux in Weber / rad on the rectangular grid points
   std::vector<real_t> PSIRZ_;

   // q values on uniform flux grid from axis to boundary
   std::vector<real_t> QPSI_;

   int                 NBBBS_;  // Number of boundary points
   std::vector<real_t> RBBBS_;  // R of boundary points in meter
   std::vector<real_t> ZBBBS_;  // Z of boundary points in meter

   int                 LIMITR_; // Number of limiter points
   std::vector<real_t> RLIM_;   // R of surrounding limiter contour in meter
   std::vector<real_t> ZLIM_;   // Z of surrounding limiter contour in meter

   class ShiftedVector : public Vector
   {
   private:
      int si_;
   public:
      ShiftedVector()
         : si_(0) {}

      ShiftedVector(int s, int si)
         : Vector(s+2*si), si_(si) {}

      void SetShift(int si) { si_ = si; }

      ShiftedVector &operator=(real_t c)
      { Vector::operator=(c); return *this; }

      inline real_t &operator()(int i)
      { return Vector::operator()(i + si_); }

      inline const real_t &operator()(int i) const
      { return Vector::operator()(i + si_); }
   };

   class ShiftedDenseMatrix : public DenseMatrix
   {
   private:
      int si_, sj_;
   public:
      ShiftedDenseMatrix()
         : si_(0), sj_(0) {}

      ShiftedDenseMatrix(int m, int n, int si, int sj)
         : DenseMatrix(m+2*si, n+2*sj), si_(si), sj_(sj) {}

      void SetShifts(int si, int sj) { si_ = si; sj_ = sj; }

      ShiftedDenseMatrix &operator=(real_t c)
      { DenseMatrix::operator=(c); return *this; }

      inline real_t &operator()(int i, int j)
      { return DenseMatrix::operator()(i + si_, j + sj_); }

      inline const real_t &operator()(int i, int j) const
      { return DenseMatrix::operator()(i + si_, j + sj_); }
   };

   class ExtendedDenseMatrix
   {
   private:
      int m_, n_;
      const real_t *C_;
      DenseMatrix N_;
      DenseMatrix S_;
      DenseMatrix E_;
      DenseMatrix W_;
      real_t SW_, SE_, NW_, NE_, DUMMY_;

      void init();

   public:
      ExtendedDenseMatrix(const real_t *C, int m, int n)
         : m_(m), n_(n), C_(C),
           N_(2, n), S_(2, n),
           E_(m, 2), W_(m, 2),
           SW_(0.0), SE_(0.0), NW_(0.0), NE_(0.0), DUMMY_(0.0)
      { N_ = 0.0; S_ = 0.0; E_ = 0.0; W_ = 0.0; init(); }

      const real_t &operator()(int i, int j) const
      {
         if (i >= 0 && i < m_ && j >= 0 && j < n_)
         {
            return C_[n_ * i + j];
         }
         else if (i >= 0 && i < m_)
         {
            if (j < 0)
            {
               return W_(i, j + 2);
            }
            else
            {
               return E_(i, j - n_);
            }
         }
         else if (j >= 0 && j < n_)
         {
            if (i < 0)
            {
               return S_(i + 2, j);
            }
            else
            {
               return N_(i - m_, j);
            }
         }
         else if (i == -1 && j == -1)
         {
            return SW_;
         }
         else if (i == -1 && j == n_)
         {
            return SE_;
         }
         else if (i == m_ && j == -1)
         {
            return NW_;
         }
         else if (i == m_ && j == n_)
         {
            return NE_;
         }
         return DUMMY_;
      }
   };

   // Divided differences for Akima's interpolation method
   real_t dr_, dz_, dpsi_;

   std::vector<real_t> FPOL_t_;
   std::vector<real_t> PRES_t_;
   std::vector<real_t> FFPRIM_t_;
   std::vector<real_t> PPRIME_t_;
   ShiftedDenseMatrix  PSIRZ_c_;
   ShiftedDenseMatrix  PSIRZ_d_;
   ShiftedDenseMatrix  PSIRZ_e_;
   std::vector<real_t> QPSI_t_;
};

class G_EQDSK_Psi_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_Psi_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   real_t Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpPsiRZ(transip);
   }
};

class G_EQDSK_FPol_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_FPol_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   real_t Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpFPolRZ(transip);
   }
};

class G_EQDSK_Pres_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_Pres_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   real_t Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpPresRZ(transip);
   }
};

class G_EQDSK_Q_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_Q_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   real_t Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpQRZ(transip);
   }
};

class G_EQDSK_BTor_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_BTor_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   real_t Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpBTorRZ(transip);
   }
};

class G_EQDSK_JTor_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_JTor_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   real_t Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpJTorRZ(transip);
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
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      eqdsk.InterpNxGradPsiRZ(transip, b);
   }

};

class G_EQDSK_BPol_Coefficient : public VectorCoefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_BPol_Coefficient(G_EQDSK_Data &g_eqdsk)
      : VectorCoefficient(2), eqdsk(g_eqdsk) {}

   void Eval(Vector &b, ElementTransformation & T,
             const IntegrationPoint & ip)
   {
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      eqdsk.InterpBPolRZ(transip, b);
   }

};

class G_EQDSK_BField_VecCoefficient : public VectorCoefficient
{
private:
   G_EQDSK_Data &eqdsk;
   bool unit_;

public:

   G_EQDSK_BField_VecCoefficient(G_EQDSK_Data &g_eqdsk, bool unit)
      : VectorCoefficient(3), eqdsk(g_eqdsk), unit_(unit) {}

   void Eval(Vector &V, ElementTransformation & T,
             const IntegrationPoint & ip)
   {
      V.SetSize(3);
      Vector b;
      b.SetSize(2);
      real_t x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      eqdsk.InterpBPolRZ(transip, b);
      real_t btor = eqdsk.InterpBTorRZ(transip);

      V[0] = b[0];
      V[1] = b[1];
      V[2] = btor;

      if ( unit_ )
      {
         real_t bmag = sqrt(V * V);
         V /= bmag;
      }
   }
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_G_EQDSK_DATA_HPP
