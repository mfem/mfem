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
   G_EQDSK_Data(std::istream &is);

   int GetNumPtsR() const { return NW_; }
   int GetNumPtsZ() const { return NH_; }

   double GetRExtent() const { return RDIM_; }
   double GetZExtent() const { return ZDIM_; }

   double GetRMin() const { return RLEFT_; }
   double GetZMid() const { return ZMID_; }

   double GetPsiCenter() const {return SIMAG_; }
   double GetPsiBdry() const {return SIBRY_; }

   std::vector<double> & GetPsi() { return PSIRZ_ ;}
   // std::vector<double> & GetBTor() { return BTOR_; }

   void PrintInfo(std::ostream &out = std::cout) const;
   void DumpGnuPlotData(const std::string &file) const;

   // double InterpFPol(double r);
   // double InterpPres(double r);
   // double InterpFFPrime(double r);
   // double InterpPPrime(double r);
   // double InterpQPsi(double r);
   double InterpFPolRZ(const Vector &rz);
   double InterpPresRZ(const Vector &rz);
   double InterpFFPrimeRZ(const Vector &rz);
   double InterpPPrimeRZ(const Vector &rz);
   double InterpPsiRZ(const Vector &rz);
   double InterpQRZ(const Vector &rz);
   double InterpBTorRZ(const Vector &rz);
   double InterpJTorRZ(const Vector &rz);

   void InterpNxGradPsiRZ(const Vector &rz, Vector &nxdp);
   void InterpBPolRZ(const Vector &rz, Vector &b);
   // double InterpBTor(double r);

   int GetNumBoundaryPts() const { return NBBBS_; }
   const std::vector<double> & GetBoundaryRVals() const { return RBBBS_; }
   const std::vector<double> & GetBoundaryZVals() const { return ZBBBS_; }

   int GetNumLimiterPts() const { return LIMITR_; }
   const std::vector<double> & GetLimiterRVals() const { return RLIM_; }
   const std::vector<double> & GetLimiterZVals() const { return ZLIM_; }

private:
   class ShiftedVector;
   class ShiftedDenseMatrix;
   class ExtendedDenseMatrix;

   enum FieldType {FPOL, PRES, FFPRIM, PPRIME, PSIRZ, QPSI/*, BTOR*/};

   int init_flag_;
   inline bool checkFlag(int flag) { return (init_flag_ >> flag) & 1; }
   inline void   setFlag(int flag) { init_flag_ |= (1 << flag); }
   inline void clearFlag(int flag) { init_flag_ &= ~(1 << flag); }

   double checkPsiBoundary();

   void initInterpR(const std::vector<double> &v,
                    std::vector<double> &t);
   void initInterpPsi(const std::vector<double> &v,
                      std::vector<double> &t);
   void initInterpRZ(const std::vector<double> &v,
                     ShiftedDenseMatrix &c,
                     ShiftedDenseMatrix &d,
                     ShiftedDenseMatrix &e);

   double interpR(double r, const std::vector<double> &v,
                  const std::vector<double> &t);
   double interpRZ(const Vector &rz,
                   const std::vector<double> &v,
                   const ShiftedDenseMatrix &c,
                   const ShiftedDenseMatrix &d,
                   const ShiftedDenseMatrix &e);
   void interpNxGradRZ(const Vector &rz,
                       const std::vector<double> &v,
                       const ShiftedDenseMatrix &c,
                       const ShiftedDenseMatrix &d,
                       const ShiftedDenseMatrix &e,
                       Vector &b);
   double interpPsi(double psi, const std::vector<double> &v,
                    const std::vector<double> &t);

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

   // Toroidal B field dervided from FPOL_
   // std::vector<double> BTOR_;

   int                 NBBBS_;  // Number of boundary points
   std::vector<double> RBBBS_;  // R of boundary points in meter
   std::vector<double> ZBBBS_;  // Z of boundary points in meter

   int                 LIMITR_; // Number of limiter points
   std::vector<double> RLIM_;   // R of surrounding limiter contour in meter
   std::vector<double> ZLIM_;   // Z of surrounding limiter contour in meter

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

      ShiftedVector &operator=(double c)
      { Vector::operator=(c); return *this; }

      inline double &operator()(int i)
      { return Vector::operator()(i + si_); }

      inline const double &operator()(int i) const
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

      ShiftedDenseMatrix &operator=(double c)
      { DenseMatrix::operator=(c); return *this; }

      inline double &operator()(int i, int j)
      { return DenseMatrix::operator()(i + si_, j + sj_); }

      inline const double &operator()(int i, int j) const
      { return DenseMatrix::operator()(i + si_, j + sj_); }
   };

   class ExtendedDenseMatrix
   {
   private:
      int m_, n_;
      const double *C_;
      DenseMatrix N_;
      DenseMatrix S_;
      DenseMatrix E_;
      DenseMatrix W_;
      double SW_, SE_, NW_, NE_, DUMMY_;

      void init();

   public:
      ExtendedDenseMatrix(const double *C, int m, int n)
         : m_(m), n_(n), C_(C),
           N_(2, n), S_(2, n),
           E_(m, 2), W_(m, 2),
           SW_(0.0), SE_(0.0), NW_(0.0), NE_(0.0), DUMMY_(0.0)
      { N_ = 0.0; S_ = 0.0; E_ = 0.0; W_ = 0.0; init(); }

      const double &operator()(int i, int j) const
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
   double dr_, dz_, dpsi_;

   std::vector<double> FPOL_t_;
   std::vector<double> PRES_t_;
   std::vector<double> FFPRIM_t_;
   std::vector<double> PPRIME_t_;
   ShiftedDenseMatrix  PSIRZ_c_;
   ShiftedDenseMatrix  PSIRZ_d_;
   ShiftedDenseMatrix  PSIRZ_e_;
   std::vector<double> QPSI_t_;

   // std::vector<double> BTOR_t_;
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

      return eqdsk.InterpPsiRZ(transip);
   }
};

class G_EQDSK_FPol_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_FPol_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
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

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
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

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
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

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
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

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
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
      double x[3];
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
      double x[3];
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
      double x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      eqdsk.InterpBPolRZ(transip, b);
      double btor = eqdsk.InterpBTorRZ(transip);

      V[0] = b[0];
      V[1] = b[1];
      V[2] = btor;

      if ( unit_ )
      {
         double bmag = sqrt(V * V);
         V /= bmag;
      }
   }
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_G_EQDSK_DATA_HPP
