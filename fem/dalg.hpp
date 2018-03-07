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
* A Class to compute the real indice from the multi-indice of a tensor
*/
template <int N, int Dim, typename T, typename... Args>
class TensorInd
{
public:
   static int result(const int* sizes, T first, Args... args)
   {
      return first + sizes[N-1]*TensorInd<N+1,Dim,Args...>::result(sizes,args...);
   }
};
//Terminal case
template <int Dim, typename T, typename... Args>
class TensorInd<Dim,Dim,T,Args...>
{
public:
   static int result(const int* sizes, T first, Args... args)
   {
      return first;
   }
};

/**
* A class to initialize the size of a Tensor
*/
template <int N, int Dim, typename T, typename... Args>
class Init
{
public:
   static int result(int* sizes, T first, Args... args){
      sizes[N-1] = first;
      return first * Init<N+1,Dim,Args...>::result(sizes,args...);
   }
};
//Terminal case
template <int Dim, typename T, typename... Args>
class Init<Dim,Dim,T,Args...>
{
public:
   static int result(int* sizes, T first, Args... args){
      sizes[Dim-1] = first;
      return first;
   }
};

/**
*  A basic generic Tensor class
*/
template<int Dim, typename Scalar=double>
class Tensor
{
protected:
   Scalar* data;
   bool own_data;
   int capacity;
   int sizes[Dim];

public:
   /**
   *  A default constructor
   */
   Tensor()
   : data(NULL), own_data(true), capacity(0)
   {
   }

   /**
   *  A default destructor
   */
   ~Tensor()
   {
      if (own_data)
      {
         delete [] data;
      }
   }

   /**
   *  A constructor to initialize the sizes of a tensor with an array of integers
   */
   Tensor(int* _sizes)
   : own_data(true)
   {
      int nb = 1;
      for (int i = 0; i < Dim; ++i)
      {
         sizes[i] = _sizes[i];
         nb *= sizes[i];
      }
      capacity = nb;
      data = new Scalar[nb]();
   }

   /**
   *  A constructor to initialize the sizes of a tensor with a variadic function
   */
   template <typename... Args>
   Tensor(Args... args)
   : own_data(true)
   {
      static_assert(sizeof...(args)==Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      long int nb = Init<1,Dim,Args...>::result(sizes,args...);
      capacity = nb;
      data = new Scalar[nb]();
   }

   /**
   *  A constructor to initialize a tensor from the Scalar array _data
   */
   template <typename... Args>
   Tensor(Scalar* _data, Args... args)
   : own_data(false)
   {
      static_assert(sizeof...(args)==Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      long int nb = Init<1,Dim,Args...>::result(sizes,args...);
      capacity = nb;
      data = _data;
   }

   /**
   *  Sets the size of the tensor, and allocate memory if necessary
   */
   template <typename... Args>
   void setSize(Args... args)
   {
      static_assert(sizeof...(args)==Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      long int nb = Init<1,Dim,Args...>::result(sizes,args...);
      if(nb>capacity)
      {
         Scalar* _data = new Scalar[nb];
         for (int i = 0; i < capacity; ++i)
         {
            _data[i] = data[i];
         }
         for (int i = capacity; i < nb; ++i)
         {
            _data[i] = Scalar();
         }
         if (own_data)
         {
            delete [] data;
         }
         data = _data;
         own_data = true;
         capacity = nb;
      }
   }

   /**
   *  A const accessor for the data
   */
   template <typename... Args>
   Scalar operator()(Args... args) const
   {
      static_assert(sizeof...(args)==Dim, "Wrong number of arguments");
      return data[ TensorInd<1,Dim,Args...>::result(sizes,args...) ];
   }

   /**
   *  A reference accessor to the data
   */
   template <typename... Args>
   Scalar& operator()(Args... args)
   {
      static_assert(sizeof...(args)==Dim, "Wrong number of arguments");
      return data[ TensorInd<1,Dim,Args...>::result(sizes,args...) ];
   }

   /**
   *  A const accessor for the data
   */
   template <typename... Args>
   Scalar operator[](Args... args) const
   {
      static_assert(sizeof...(args)==Dim, "Wrong number of arguments");
      return data[ TensorInd<1,Dim,Args...>::result(sizes,args...) ];
   }

   /**
   *  A reference accessor to the data
   */
   template <typename... Args>
   Scalar& operator[](Args... args)
   {
      static_assert(sizeof...(args)==Dim, "Wrong number of arguments");
      return data[ TensorInd<1,Dim,Args...>::result(sizes,args...) ];
   }

   void zero()
   {
      for (int i = 0; i < capacity; ++i)
      {
         data[i] = Scalar();
      }
   }

   /**
   *  Returns the size of the i-th dimension #UNSAFE#
   */
   int size(int i) const
   {
      return sizes[i];
   }

   /**
   *  Returns the dimension of the tensor.
   */
   int dimension() const
   {
      return Dim;
   }

   /**
   *  Returns the Scalar array data
   */
   Scalar* getData()
   {
      return data;
   }

   const Scalar* getData() const
   {
      return data;
   }

   /**
   *  Basic printing method.
   */
   friend std::ostream& operator<<(std::ostream& os, const Tensor& T)
   {
      int nb_elts = 1;
      for (int i = 0; i < T.dimension(); ++i)
      {
          nb_elts *= T.size(i);
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

   ~DummyMatrix()
   {
      delete[] data;
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
   DummyTensor(int dim)
   : data(NULL), dim(dim), own_data(true), sizes(dim,1), offsets(dim,1)
   {
   }

   DummyTensor(int dim, double* _data, int* dimensions)
   : data(_data), dim(dim), own_data(false), sizes(dim,1), offsets(dim,1)
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

   void Zero()
   {
      for (int i = 0; i < GetNumVal(); ++i)
      {
         data[i] = 0.0;
      }
   }

   void SetSize(int *_sizes)
   {
      sizes[0] = _sizes[0];
      int dim_ind = 1;
      offsets[0] = dim_ind;
      for (int i = 1; i < dim; ++i)
      {
         sizes[i] = _sizes[i];
         dim_ind *= sizes[i-1];
         offsets[i] = dim_ind;
      }
      if (own_data && data==NULL)
      {
         data = new double[GetNumVal()]();
      }
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