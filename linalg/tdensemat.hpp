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

#ifndef MFEM_TDENSEMAT
#define MFEM_TDENSEMAT

#ifdef MFEM_USE_X86INTRIN
#include "../general/x86intrin.hpp"
#endif

#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cstdlib>

namespace mfem{

/// Data type dense matrix using column-major storage
template<typename data_t = double>
class TDenseMatrix : public Matrix{
private:
   data_t *data;
   int capacity; // zero or negative capacity means we do not own the data.
public:
  
  /// Creates rectangular matrix of size m x n.
  TDenseMatrix(int m, int n) : Matrix(m, n){
    std::cout<<"[33;1m[TDenseMatrix][m"<<std::endl;

    MFEM_ASSERT(m >= 0 && n >= 0,
                "invalid TDenseMatrix size: " << m << " x " << n);
    capacity = m*n;
    MFEM_ASSERT(capacity>0,"invalid TDenseMatrix capacity");
#ifndef MFEM_USE_X86INTRIN
    data = new data_t[capacity]();
#else
    data = (data_t*)aligned_alloc(x86::align,capacity*sizeof(data_t));
#endif
  }
   
   TDenseMatrix(data_t *d, int h, int w) : Matrix(h, w)
   { data = d; capacity = -h*w; }

  /// Returns reference to a_{ij}.
  inline data_t &operator()(int i, int j){
    MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
    return data[i+j*height];
  }
  /// Returns constant reference to a_{ij}.
  inline const data_t &operator()(int i, int j) const{
    MFEM_ASSERT(data && i >= 0 && i < height && j >= 0 && j < width, "");
    return data[i+j*height];
  } 

  inline DenseMatrix &simd(DenseMatrix &D, int k) const{
    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        D(i,j)=(*this)(i,j)[k];
    return D;
  }

  double &Elem(int i, int j){
    MFEM_ASSERT(false,"Elem SIMD HACK");
    return (*this)(i,j)[0]; // SIMD HACK
  }
  
  /// Returns reference to a_{ij}.
  const double &Elem(int i, int j) const {
    MFEM_ASSERT(false,"Elem not implemented");
    return (*this)(i,j)[0];
  }
  
  void Mult(const Vector &x, Vector &y) const {
    MFEM_ASSERT(false,"Mult not implemented");
  }
  
  virtual MatrixInverse *Inverse() const {
    MFEM_ASSERT(false,"Inverse not implemented");
    return NULL;
  }

  /** Copy the m x n submatrix of A at row/col offsets Aro/Aco to *this at
      row_offset, col_offset */
  void CopyMN(const TDenseMatrix<data_t> &A,
              int m, int n, int Aro, int Aco,
              int row_offset, int col_offset){}

  /// Destroys dense matrix.
  ~TDenseMatrix(){}

  void Print(int k,std::ostream &out = std::cout, int width_ = 4) const{
    std::ios::fmtflags old_flags = out.flags();
    // output flags = scientific + show sign
    out << setiosflags(std::ios::scientific | std::ios::showpos);
    for (int i = 0; i < height; i++){
      out << "[row " << i << "]\n";
      for (int j = 0; j < width; j++){
        out << (*this)(i,j)[k];
        if (j+1 == width || (j+1) % width_ == 0){
          out << '\n';
        }else{
          out << ' ';
        }
      }
    }
   // reset output flags to original values
   out.flags(old_flags);
  }
  
};

} // namespace mfem

#endif
