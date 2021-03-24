#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ComplexDenseMatrix 
{
private:
   std::complex<double> * data = nullptr;
   int height = 0;
   int width  = 0;
public:
   ComplexDenseMatrix();
   
   /// Creates square matrix of size s.
   explicit ComplexDenseMatrix(int s);

   /// Creates rectangular matrix of size m x n.
   ComplexDenseMatrix(int m, int n);

   /// Change the size of the DenseMatrix to s x s.
   void SetSize(int s) { SetSize(s, s); }

   /// Change the size of the DenseMatrix to h x w.
   void SetSize(int h, int w);

   /// Returns the matrix data array.
   inline complex<double> *Data() const
   { return const_cast<complex<double>*>((const complex<double>*)data);}

   /// Returns the matrix data array.
   inline complex<double> *GetData() const { return Data(); }

   /// Returns reference to a_{ij}.
   inline complex<double> &operator()(int i, int j);
   inline const complex<double> &operator()(int i, int j) const;

   inline int Height() const { return height; }
   inline int Width() const { return width; }

   /// Sets the matrix elements equal to constant c
   ComplexDenseMatrix &operator=(std::complex<double> c);
   ComplexDenseMatrix &operator=(double c);

   /// Sets the matrix size and elements equal to those of m
   ComplexDenseMatrix &operator=(const ComplexDenseMatrix &m);
   ComplexDenseMatrix &operator+=(const complex<double> *m);
   ComplexDenseMatrix &operator+=(const ComplexDenseMatrix &m);
   ComplexDenseMatrix &operator-=(const ComplexDenseMatrix &m);
   ComplexDenseMatrix &operator*=(complex<double> c);

   /// Calculates the determinant of the matrix
   /// (for 2x2, 3x3)
   std::complex<double> Det() const;

   virtual void Print(std::ostream &out = mfem::out, int width_ = 4) const;
   virtual void PrintMatlab(std::ostream &out = mfem::out) const;

   DenseMatrix * real() const;
   DenseMatrix * imag() const;

   void GetReal(DenseMatrix & Ar);
   void GetImag(DenseMatrix & Ai);
};

inline complex<double> &ComplexDenseMatrix::operator()(int i, int j)
{
   MFEM_VERIFY(data && i >= 0 && i < height && j >= 0 && j < width, "");
   // return data[i*width+j];
   return data[j*height+i];
}

inline const complex<double> &ComplexDenseMatrix::operator()(int i, int j) const
{
   MFEM_VERIFY(data && i >= 0 && i < height && j >= 0 && j < width, "");
   // return data[i*width+j];
   return data[j*height+i];
}


class ComplexDenseMatrixInverse : public ComplexDenseMatrix
{
private:
public:
   ComplexDenseMatrixInverse(const ComplexDenseMatrix & );   
};

/// Matrix matrix multiplication.  A = B * C.
void Mult(const ComplexDenseMatrix &b, const ComplexDenseMatrix &c, ComplexDenseMatrix &a);

/// Multiply the transpose of a matrix A with a matrix B:   At*B
void MultAtB(const ComplexDenseMatrix &A, const ComplexDenseMatrix &B, ComplexDenseMatrix &AtB);

/// Multiply the conjugate transpose of a matrix A with a matrix B:   At*B
void MultAhB(const ComplexDenseMatrix &A, const ComplexDenseMatrix &B, ComplexDenseMatrix &AtB);