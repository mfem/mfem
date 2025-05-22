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

   real_t GetRExtent() const { return RDIM_; }
   real_t GetZExtent() const { return ZDIM_; }

   real_t GetRMin() const { return RLEFT_; }
   real_t GetZMid() const { return ZMID_; }

   real_t GetPsiCenter() const {return SIMAG_; }
   real_t GetPsiBdry() const {return SIBRY_; }

   std::vector<real_t> & GetPsi() { return PSIRZ_ ;}
   // std::vector<real_t> & GetBTor() { return BTOR_; }

   void PrintInfo(std::ostream &out = std::cout) const;
   void DumpGnuPlotData(const std::string &file) const;

   // real_t InterpFPol(real_t r);
   // real_t InterpPres(real_t r);
   // real_t InterpFFPrime(real_t r);
   // real_t InterpPPrime(real_t r);
   // real_t InterpQPsi(real_t r);
   real_t InterpFPolRZ(const Vector &rz);
   real_t InterpPresRZ(const Vector &rz);
   real_t InterpFFPrimeRZ(const Vector &rz);
   real_t InterpPPrimeRZ(const Vector &rz);
   real_t InterpPsiRZ(const Vector &rz);
   real_t InterpQRZ(const Vector &rz);
   real_t InterpBTorRZ(const Vector &rz);
   real_t InterpJTorRZ(const Vector &rz);

   void InterpNxGradPsiRZ(const Vector &rz, Vector &nxdp);
   void InterpBPolRZ(const Vector &rz, Vector &b);
   // real_t InterpBTor(real_t r);

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

   void initInterpR(const std::vector<real_t> &v,
                    std::vector<real_t> &t);
   void initInterpPsi(const std::vector<real_t> &v,
                      std::vector<real_t> &t);
   void initInterpRZ(const std::vector<real_t> &v,
                     ShiftedDenseMatrix &c,
                     ShiftedDenseMatrix &d,
                     ShiftedDenseMatrix &e);

   real_t interpR(real_t r, const std::vector<real_t> &v,
                  const std::vector<real_t> &t);
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

   // FF’(ψ) in (mT)2 / (Weber /rad) on uniform flux grid
   std::vector<real_t> FFPRIM_;

   // P’(ψ) in (nt /m2) / (Weber /rad) on uniform flux grid
   std::vector<real_t> PPRIME_;

   // Poloidal flux in Weber / rad on the rectangular grid points
   std::vector<real_t> PSIRZ_;

   // q values on uniform flux grid from axis to boundary
   std::vector<real_t> QPSI_;

   // Toroidal B field dervided from FPOL_
   // std::vector<real_t> BTOR_;

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

   // std::vector<real_t> BTOR_t_;
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
