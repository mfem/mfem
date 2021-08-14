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

#ifndef TADDENSEMATRIX_H
#define TADDENSEMATRIX_H

#include "mfem.hpp"
#include "tadvector.hpp"

namespace mfem
{
/// Templated dense matrix data type.
/** The main goal of the TADDenseMatrix class is to serve as a data
  container for representing dense matrices in classes, methods, and
  functions utilized with automatic differentiation (AD). The
  functionality/interface is copied from the standard MFEM dense
  matrix mfem::DenseMatrix. The basic idea is to utilize the templated
  vector class in combination with AD during the development phase.
  The AD parts can be replaced with optimized code once the initial
  development of the application is complete.  The common interface
  between TADDenseMatrix and DenseMatrix will ease the transition
  from AD to hand-optimized code as it does not require a change
  in the interface or the code structure. TADDenseMatrix is intended
  to be utilized for dense serial matrices. The objects can be combined
  with TADVector or standard Vector.*/
template<typename dtype>
class TADDenseMatrix
{
private:
   int height; ///< Dimension of the output / number of rows in the matrix.
   int width;  ///< Dimension of the input / number of columns in the matrix.
   dtype *data;
   int capacity; // zero or negative capacity means we do not own the data.

public:
   /// Get the height (size of output) of the Operator. Synonym with NumRows().
   inline int Height() const { return height; }
   /** @brief Get the number of rows (size of output) of the Operator. Synonym
       with Height(). */
   inline int NumRows() const { return height; }

   /// Get the width (size of input) of the Operator. Synonym with NumCols().
   inline int Width() const { return width; }
   /** @brief Get the number of columns (size of input) of the Operator. Synonym
       with Width(). */
   inline int NumCols() const { return width; }

   /** Default constructor for TADDenseMatrix.
      Sets data = NULL and height = width = 0. */
   TADDenseMatrix()
   {
      data = nullptr;
      capacity = 0;
      height = 0;
      width = 0;
   }

   /// Copy constructor
   template<typename idtype>
   TADDenseMatrix(const TADDenseMatrix<idtype> &m)
   {
      height = m.GetHeight();
      width = m.GetWidth();
      const int hw = height * width;
      if (hw > 0)
      {
         idtype *mdata = m.Data();
         MFEM_ASSERT(mdata, "invalid source matrix");
         data = new dtype[hw];
         capacity = hw;
         for (int i = 0; i < hw; i++)
         {
            data[i] = mdata[i];
         }
      }
      else
      {
         data = nullptr;
         capacity = 0;
         width = 0;
         height = 0;
      }
   }
   /// Copy constructor using standard DenseMatrix
   TADDenseMatrix(const DenseMatrix &m)
   {
      height = m.Height();
      width = m.Width();
      const int hw = height * width;
      if (hw > 0)
      {
         double *mdata = m.Data();
         MFEM_ASSERT(mdata, "invalid source matrix");
         data = new dtype[hw];
         capacity = hw;
         for (int i = 0; i < hw; i++)
         {
            data[i] = mdata[i];
         }
      }
      else
      {
         data = nullptr;
         capacity = 0;
         width = 0;
         height = 0;
      }
   }

   /// Creates square matrix of size s.
   explicit TADDenseMatrix(int s)
   {
      MFEM_ASSERT(s >= 0, "invalid DenseMatrix size: " << s);
      height = s;
      width = s;
      capacity = s * s;
      if (capacity > 0)
      {
         data = new dtype[capacity](); // init with zeroes
      }
      else
      {
         data = nullptr;
      }
   }

   /// Creates rectangular matrix of size m x n.
   TADDenseMatrix(int m, int n)
   {
      MFEM_ASSERT(m >= 0 && n >= 0,
                  "invalid DenseMatrix size: " << m << " x " << n);
      height = m;
      width = n;
      capacity = m * n;
      if (capacity > 0)
      {
         data = new dtype[capacity](); // init with zeroes
      }
      else
      {
         data = nullptr;
      }
   }

   TADDenseMatrix(const TADDenseMatrix<dtype> &mat, char ch)
   {
      height = mat.Width();
      width = mat.Height();
      capacity = height * width;
      if (capacity > 0)
      {
         data = new dtype[capacity];

         for (int i = 0; i < height; i++)
         {
            for (int j = 0; j < width; j++)
            {
               (*this)(i, j) = mat(j, i);
            }
         }
      }
      else
      {
         data = nullptr;
      }
   }

   /// Change the size of the DenseMatrix to s x s.
   void SetSize(int s) { SetSize(s, s); }

   /// Change the size of the DenseMatrix to h x w.
   void SetSize(int h, int w)
   {
      MFEM_ASSERT(h >= 0 && w >= 0,
                  "invalid DenseMatrix size: " << h << " x " << w);
      if (Height() == h && Width() == w)
      {
         return;
      }
      height = h;
      width = w;
      const int hw = h * w;
      if (hw > std::abs(capacity))
      {
         if (capacity > 0)
         {
            delete[] data;
         }
         capacity = hw;
         data = new dtype[hw](); // init with zeroes
      }
   }

   /// Returns the matrix data array.
   inline dtype *Data() const { return data; }
   /// Returns the matrix data array.
   inline dtype *GetData() const { return data; }

   inline bool OwnsData() const { return (capacity > 0); }

   /// Returns reference to a_{ij}.
   dtype &operator()(int i, int j)
   {
      MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
      return data[i + j * height];
   }

   const dtype &operator()(int i, int j) const
   {
      MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
      return data[i + j * height];
   }

   dtype &Elem(int i, int j) { return (*this)(i, j); }

   const dtype &Elem(int i, int j) const { return (*this)(i, j); }

   void Mult(const dtype *x, dtype *y) const
   {
      if (width == 0)
      {
         for (int row = 0; row < height; row++)
         {
            y[row] = 0.0;
         }
         return;
      }
      dtype *d_col = data;
      dtype x_col = x[0];
      for (int row = 0; row < height; row++)
      {
         y[row] = x_col * d_col[row];
      }
      d_col += height;
      for (int col = 1; col < width; col++)
      {
         x_col = x[col];
         for (int row = 0; row < height; row++)
         {
            y[row] += x_col * d_col[row];
         }
         d_col += height;
      }
   }

   void Mult(const TADVector<dtype> &x, TADVector<dtype> &y) const
   {
      MFEM_ASSERT(height == y.Size() && width == x.Size(),
                  "incompatible dimensions");

      Mult((const dtype *) x, (dtype *) y);
   }

   dtype operator*(const TADDenseMatrix<dtype> &m) const
   {
      MFEM_ASSERT(Height() == m.Height() && Width() == m.Width(),
                  "incompatible dimensions");

      const int hw = height * width;
      dtype a = 0.0;
      for (int i = 0; i < hw; i++)
      {
         a += data[i] * m.data[i];
      }

      return a;
   }

   void MultTranspose(const dtype *x, dtype *y) const
   {
      dtype *d_col = data;
      for (int col = 0; col < width; col++)
      {
         double y_col = 0.0;
         for (int row = 0; row < height; row++)
         {
            y_col += x[row] * d_col[row];
         }
         y[col] = y_col;
         d_col += height;
      }
   }

   void MultTranspose(const TADVector<dtype> &x, TADVector<dtype> &y) const
   {
      MFEM_ASSERT(height == x.Size() && width == y.Size(),
                  "incompatible dimensions");

      MultTranspose((const dtype *) x, (dtype *) y);
   }

   void Randomize(int seed)
   {
      // static unsigned int seed = time(0);
      const double max = (double) (RAND_MAX) + 1.;

      if (seed == 0)
      {
         seed = (int) time(0);
      }

      // srand(seed++);
      srand((unsigned) seed);

      for (int i = 0; i < capacity; i++)
      {
         data[i] = (dtype)(std::abs(rand() / max));
      }
   }

   void RandomizeDiag(int seed)
   {
      // static unsigned int seed = time(0);
      const double max = (double) (RAND_MAX) + 1.;

      if (seed == 0)
      {
         seed = (int) time(0);
      }

      // srand(seed++);
      srand((unsigned) seed);

      for (int i = 0; i < std::min(height, width); i++)
      {
         Elem(i, i) = (dtype)(std::abs(rand() / max));
      }
   }

   /// Creates n x n diagonal matrix with diagonal elements c
   void Diag(dtype c, int n)
   {
      SetSize(n);

      const int N = n * n;
      for (int i = 0; i < N; i++)
      {
         data[i] = (dtype) 0.0;
      }
      for (int i = 0; i < n; i++)
      {
         data[i * (n + 1)] = c;
      }
   }
   /// Creates n x n diagonal matrix with diagonal given by diag
   template<typename itype>
   void Diag(itype *diag, int n)
   {
      SetSize(n);

      int i, N = n * n;
      for (i = 0; i < N; i++)
      {
         data[i] = 0.0;
      }
      for (i = 0; i < n; i++)
      {
         data[i * (n + 1)] = (dtype) diag[i];
      }
   }

   /// (*this) = (*this)^t
   void Transpose()
   {
      int i, j;
      dtype t;

      if (Width() == Height())
      {
         for (i = 0; i < Height(); i++)
            for (j = i + 1; j < Width(); j++)
            {
               t = (*this)(i, j);
               (*this)(i, j) = (*this)(j, i);
               (*this)(j, i) = t;
            }
      }
      else
      {
         TADDenseMatrix<dtype> T(*this, 't');
         (*this) = T;
      }
   }
   /// (*this) = A^t
   template<typename itype>
   void Transpose(const TADDenseMatrix<itype> &A)
   {
      SetSize(A.Width(), A.Height());

      for (int i = 0; i < Height(); i++)
         for (int j = 0; j < Width(); j++)
         {
            (*this)(i, j) = (dtype) A(j, i);
         }
   }

   /// (*this) = 1/2 ((*this) + (*this)^t)
   void Symmetrize()
   {
#ifdef MFEM_DEBUG
      if (Width() != Height())
      {
         mfem_error("DenseMatrix::Symmetrize() : not a square matrix!");
      }
#endif

      for (int i = 0; i < Height(); i++)
         for (int j = 0; j < i; j++)
         {
            dtype a = 0.5 * ((*this)(i, j) + (*this)(j, i));
            (*this)(j, i) = (*this)(i, j) = a;
         }
   }

   void Lump()
   {
      for (int i = 0; i < Height(); i++)
      {
         dtype L = 0.0;
         for (int j = 0; j < Width(); j++)
         {
            L += (*this)(i, j);
            (*this)(i, j) = (dtype) 0.0;
         }
         (*this)(i, i) = L;
      }
   }
};

template<typename dtype>
void CalcAdjugate(const TADDenseMatrix<dtype> &a, TADDenseMatrix<dtype> &adja)
{
#ifdef MFEM_DEBUG
   if (a.Width() > a.Height() || a.Width() < 1 || a.Height() > 3)
   {
      mfem_error("CalcAdjugate(...)");
   }
   if (a.Width() != adja.Height() || a.Height() != adja.Width())
   {
      mfem_error("CalcAdjugate(...)");
   }
#endif

   if (a.Width() < a.Height())
   {
      const dtype *d = a.Data();
      dtype *ad = adja.Data();
      if (a.Width() == 1)
      {
         // N x 1, N = 2,3
         ad[0] = d[0];
         ad[1] = d[1];
         if (a.Height() == 3)
         {
            ad[2] = d[2];
         }
      }
      else
      {
         // 3 x 2
         double e, g, f;
         e = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
         g = d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
         f = d[0] * d[3] + d[1] * d[4] + d[2] * d[5];

         ad[0] = d[0] * g - d[3] * f;
         ad[1] = d[3] * e - d[0] * f;
         ad[2] = d[1] * g - d[4] * f;
         ad[3] = d[4] * e - d[1] * f;
         ad[4] = d[2] * g - d[5] * f;
         ad[5] = d[5] * e - d[2] * f;
      }
      return;
   }

   if (a.Width() == 1)
   {
      adja(0, 0) = (dtype) 1.0;
   }
   else if (a.Width() == 2)
   {
      adja(0, 0) = a(1, 1);
      adja(0, 1) = -a(0, 1);
      adja(1, 0) = -a(1, 0);
      adja(1, 1) = a(0, 0);
   }
   else
   {
      adja(0, 0) = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
      adja(0, 1) = a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2);
      adja(0, 2) = a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1);

      adja(1, 0) = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
      adja(1, 1) = a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0);
      adja(1, 2) = a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2);

      adja(2, 0) = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
      adja(2, 1) = a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1);
      adja(2, 2) = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
   }
}

} // namespace mfem

#endif
