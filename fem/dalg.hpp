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

// This file contains operator-based bilinear form integrators used
// with BilinearFormOperator.

#ifndef MFEM_DUMMYALGEBRA
#define MFEM_DUMMYALGEBRA

#include "../config/config.hpp"
#include "bilininteg.hpp"
#include <vector>
#include <iostream>

using namespace std;
using std::vector;

namespace mfem
{

/**
* A dummy Matrix implementation that handles any type
*/
template <typename Scalar>
class DummyMatrix
{
protected:
   Scalar* data;
   int sizes[2];

public:
   DummyMatrix()
   {
      sizes[0] = 0;
      sizes[1] = 0;
      data = NULL;
   }

   DummyMatrix(int rows, int cols)
   {
      sizes[0] = rows;
      sizes[1] = cols;
      data = new Scalar[rows*cols];//rows*cols*sizeof(Scalar) );
   }

   // Sets all the coefficients to zero
   void Zero()
   {
      for (int i = 0; i < sizes[0]*sizes[1]; ++i)
      {
         data[i] = Scalar();
      }
   }

   // Accessor for the Matrix
   const Scalar operator()(int row, int col) const
   {
      return data[ row + sizes[0]*col ];
   }

   // Accessor for the Matrix
   Scalar& operator()(int row, int col)
   {
      return data[ row + sizes[0]*col ];
   }

   int Height() const
   {
      return sizes[0];
   }

   int Width() const
   {
      return sizes[1];
   }

   friend ostream& operator<<(ostream& os, const DummyMatrix& M){
      for (int i = 0; i < M.Height(); ++i)
      {
         for (int j = 0; j < M.Width(); ++j)
         {
            os << M(i,j) << " ";
         }
         os << "\n";
      }
      return os;
   }

};

typedef DummyMatrix<double> DMatrix;
typedef DummyMatrix<int> IntMatrix;

/**
*  A dummy tensor class
*/
class DummyTensor
{
protected:
   double* data;
   const int dim;
   bool own_data;
   vector<int> sizes;
   vector<int> offsets;

public:
   DummyTensor(int dim) : dim(dim), own_data(true), sizes(dim,1), offsets(dim,1)
   {
   }

   DummyTensor(int dim, double* _data, int* dimensions)
   : data(_data), dim(dim), own_data(false)
   {
      SetSize(dimensions);
   }

   int GetNumVal()
   {
      int result = 1;
      for (int i = 0; i < dim; ++i)
      {
         result *= sizes[i];
      }
      return result;
   }

   // Memory leak if used more than one time
   void SetSize(int *_sizes)
   {
      for (int i = 0; i < dim; ++i)
      {
         sizes[i] = _sizes[i];
         int dim_ind = 1;
         // We don't really need to recompute from beginning, but that shouldn't
         // be a performance issue...
         for (int j = 0; j < i; ++j)
         {
            dim_ind *= sizes[j];
         }
         offsets[i] = dim_ind;
      }
      data = new double[GetNumVal()];//(double*)malloc(GetNumVal()*sizeof(double));
   }

   // Returns the data pointer, to change container for instance, or access data
   // in an unsafe way...
   double* GetData()
   {
      return data;
   }

   // The function that defines the Layout
   int GetRealInd(int* ind)
   {
      int real_ind = 0;
      for (int i = 0; i < dim; ++i)
      {
         real_ind += ind[i]*offsets[i];
      }
      return real_ind;
   }

   // really unsafe!
   void SetVal(int* ind, double val)
   {
      int real_ind = GetRealInd(ind);
      data[real_ind] = val;
   }

   double GetVal(int real_ind)
   {
      return data[real_ind];
   }

   double GetVal(int* ind)
   {
      int real_ind = GetRealInd(ind);
      return data[real_ind];
   }

   double& operator()(int real_ind)
   {
      return data[real_ind];
   }

   double& operator()(int* ind)
   {
      int real_ind = GetRealInd(ind);
      return data[real_ind];
   }

   // Assumes that elements/faces indice is always the last indice
   double* GetElmtData(int e){
      return &data[ e * offsets[dim-1] ];
   }

   friend ostream& operator<<(ostream& os, const DummyTensor& T){
      int nb_elts = 1;
      for (int i = 0; i < T.dim; ++i)
      {
         nb_elts *= T.sizes[i];
      }
      for (int i = 0; i < nb_elts; ++i)
      {
         os << T.data[i] << " ";
         if ((i+1)%T.sizes[0]==0)
         {
            os << "\n";
         }
      }
      os << "\n";
      return os;
   }

};


}

#endif //MFEM_DUMMYALGEBRA