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

   /// Calculates the determinant of the matrix
   /// (for 2x2, 3x3)
   std::complex<double> Det() const;

   virtual void Print(std::ostream &out = mfem::out, int width_ = 4) const;
   virtual void PrintMatlab(std::ostream &out = mfem::out) const;
};

inline complex<double> &ComplexDenseMatrix::operator()(int i, int j)
{
   MFEM_VERIFY(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}

inline const complex<double> &ComplexDenseMatrix::operator()(int i, int j) const
{
   MFEM_VERIFY(data && i >= 0 && i < height && j >= 0 && j < width, "");
   return data[i+j*height];
}


class ComplexDenseMatrixInverse : public ComplexDenseMatrix
{
private:
public:
   ComplexDenseMatrixInverse(const ComplexDenseMatrix & );   
};



// class ComplexDenseMatrixInverse : public ComplexDenseMatrix
// {
// private:
//    int dim;
// public:
//    ComplexDenseMatrixInverse(ComplexDenseMatrix & MatZ)
//    {
//       SetMatrix(MatZ);
//    }
//    void SetMatrix(ComplexDenseMatrix & MatZ) 
//    {
//       MFEM_VERIFY(MatZ.Height() == MatZ.Width(), "Not a square matrix");
//       dim = MatZ.Height();
//       std::complex<double> detZ = MatZ.GetDeterminant();
//       MFEM_VERIFY(std::abs(detZ) >1e-15, "Zero Determinant, matrix is singular");

//       (*this).SetSize(dim,dim);

//       // fill in the Inv;
//       switch (dim)
//       {
//       case 1:
//       {
//          (*this)(0,0) = 1.0/MatZ(0,0);
//       }
//       break;
//       case 2:
//       {
//          (*this)(0,0) =  1.0/detZ * MatZ(1,1);
//          (*this)(0,1) = -1.0/detZ * MatZ(0,1);
//          (*this)(1,0) = -1.0/detZ * MatZ(1,0);
//          (*this)(1,1) =  1.0/detZ * MatZ(0,0);
//       }
//       break;
//       case 3:
//       {
//          (*this)(0,0) =  1.0/detZ*(MatZ(1,1)*MatZ(2,2) - MatZ(1,2)*MatZ(2,1));
//          (*this)(0,1) = -1.0/detZ*(MatZ(0,1)*MatZ(2,2) - MatZ(0,2)*MatZ(2,1));
//          (*this)(0,2) =  1.0/detZ*(MatZ(0,1)*MatZ(1,2) - MatZ(0,2)*MatZ(1,1));
//          (*this)(1,0) = -1.0/detZ*(MatZ(1,0)*MatZ(2,2) - MatZ(1,2)*MatZ(2,0));
//          (*this)(1,1) =  1.0/detZ*(MatZ(0,0)*MatZ(2,2) - MatZ(2,0)*MatZ(0,2));
//          (*this)(1,2) = -1.0/detZ*(MatZ(0,0)*MatZ(1,2) - MatZ(0,2)*MatZ(1,0));
//          (*this)(2,0) =  1.0/detZ*(MatZ(1,0)*MatZ(2,1) - MatZ(1,1)*MatZ(2,0));
//          (*this)(2,1) = -1.0/detZ*(MatZ(0,0)*MatZ(2,1) - MatZ(0,1)*MatZ(2,0));
//          (*this)(2,2) =  1.0/detZ*(MatZ(0,0)*MatZ(1,1) - MatZ(0,1)*MatZ(1,0));
//          /* code */
//       }
//       break;                  
//       default: MFEM_ABORT("dim>3 not supported yet");
//          break;
//       }
//    }
// };
