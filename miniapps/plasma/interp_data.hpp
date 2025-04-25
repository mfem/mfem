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

#ifndef MFEM_INTERP_DATA_HPP
#define MFEM_INTERP_DATA_HPP

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

class Interp_Data
{
public:
   Interp_Data(std::istream &is);

   int GetNumPtsR() const { return NW_; }
   int GetNumPtsZ() const { return NH_; }

   double GetRExtent() const { return RDIM_; }
   double GetZExtent() const { return ZDIM_; }

   double GetRMin() const { return RLEFT_; }
   double GetZMid() const { return ZMID_; }

   void PrintInfo(std::ostream &out = std::cout) const;

   double InterpDataRZ(const Vector &rz);

private:
   class ShiftedVector;
   class ShiftedDenseMatrix;
   class ExtendedDenseMatrix;

   int init_flag_;

   void initInterpRZ(const std::vector<double> &v,
                     ShiftedDenseMatrix &c,
                     ShiftedDenseMatrix &d,
                     ShiftedDenseMatrix &e);

   double interpRZ(const Vector &rz,
                   const std::vector<double> &v,
                   const ShiftedDenseMatrix &c,
                   const ShiftedDenseMatrix &d,
                   const ShiftedDenseMatrix &e);

   std::vector<std::string> CASE_; // Identification character string

   int NW_; // Number of horizontal R grid points
   int NH_; // Number of vertical Z grid points

   double RDIM_;    // Horizontal dimension in meter of computational box
   double ZDIM_;    // Vertical dimension in meter of computational box
   double RLEFT_;   // Minimum R in meter of rectangular computational box
   double ZMID_;    // Z of center of computational box in meter

   // Field on grid
   std::vector<double> FIELD_;

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
   double dr_, dz_;
   ShiftedDenseMatrix  DATA_c_;
   ShiftedDenseMatrix  DATA_d_;
   ShiftedDenseMatrix  DATA_e_;
};

class Interp_Data_Coefficient : public Coefficient
{
private:
   Interp_Data &interp_data;

public:

   Interp_Data_Coefficient(Interp_Data &i_data) : interp_data(i_data) {}

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return interp_data.InterpDataRZ(transip);
   }
};


} // namespace plasma

} // namespace mfem

#endif // MFEM_INTERP_DATA_HPP
