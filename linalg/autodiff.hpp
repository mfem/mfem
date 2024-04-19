
// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#ifndef MFEM_AD
#define MFEM_AD

#include "vector.hpp"
#include "densemat.hpp"

#ifdef MFEM_USE_CODIPACK
#include <codi.hpp>
#endif

//#include <vector>
#include "tadvector.hpp"
#include "taddensemat.hpp"
#include <functional>

namespace mfem
{

#ifdef MFEM_USE_CODIPACK
#ifdef MFEM_USE_ADFORWARD
/// Forward AD type declaration
typedef codi::RealForward ADFloat;
#else
/// Reverse AD type declaration
typedef codi::RealReverse ADFloat;
#endif
#else
typedef real_t ADFloat;
#endif

/// Vector type for AD-numbers
typedef TAutoDiffVector<ADFloat> ADVector;
/// Matrix type for AD-numbers
typedef TAutoDiffDenseMatrix<ADFloat> ADMatrix;


/// The class provides an evaluation of the Jacobian of a templated vector
/// function provided in the constructor. The Jacobian is evaluated with the
/// help of automatic differentiation (AD). The template parameters specify the
/// size of the return vector (vector_size), the size of the input vector
/// (state_size), and the size of the parameters supplied to the function.
template<int param_size=0, int vector_size=1, int state_size=1>
class ADVectorFunc
{
private:
   std::function<void(Vector&, ADVector&, ADVector&)> F;
   Vector param;
public:
   /// F_ is user implemented function to be differentiated by
   /// ADVectorFunc. The signature of the function is: F_(Vector&
   /// parameters, ad::ADVectorType& state_vector, ad::ADVectorType& result).
   /// The parameters vector should have size param_size. The state_vector
   /// should have size state_size, and the result vector should have size
   /// vector_size. All size parameters are teplate parameters in
   /// ADVectorFunc.
   ADVectorFunc(
      std::function<void(Vector&, ADVector&, ADVector&)> F_,
      Vector &par)
   {
      F=F_;
      param = par;
   }

   ADVectorFunc(
      std::function<void(Vector&, ADVector&, ADVector&)> F_)
   {
      F=F_;
   }

   /// Evaluates the Jacobian of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor. The Jacobian (jac) has dimensions
   /// [vector_size x state_size].
   void Jacobian(Vector &vstate,DenseMatrix &jac);

   /// Evaluates the Curl of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor. The Curl only works in 3D.
   void Curl(Vector &vstate, Vector &curl);

   /// Evaluates the Divergence of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor.
   real_t Divergence(Vector &vstate);

   /// Evaluates the Gradient of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The Gradient (grad) has dimensions
   /// [ state_size].
   void Gradient(mfem::Vector &vstate, mfem::Vector &grad);

   /// Evaluates the Gradient of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The Solution (sol) has dimensions
   /// [ state_size]. The function and parameters are provided in the Constructor.
   void Solution(Vector &vstate, Vector &sol);

   /// Evaluates the Gradient of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate.  The function and parameters are
   /// provided in the Constructor.
   real_t Solution(Vector &vstate);
}; // ADVectorFunc

template class ADVectorFunc<0, 1, 1>;
template class ADVectorFunc<0, 1, 2>;
template class ADVectorFunc<0, 1, 3>;

template class ADVectorFunc<0, 2, 1>;
template class ADVectorFunc<0, 2, 2>;
template class ADVectorFunc<0, 2, 3>;

template class ADVectorFunc<0, 3, 1>;
template class ADVectorFunc<0, 3, 2>;
template class ADVectorFunc<0, 3, 3>;

template class ADVectorFunc<1, 1, 1>;
template class ADVectorFunc<1, 1, 2>;
template class ADVectorFunc<1, 1, 3>;

template class ADVectorFunc<1, 2, 1>;
template class ADVectorFunc<1, 2, 2>;
template class ADVectorFunc<1, 2, 3>;

template class ADVectorFunc<1, 3, 1>;
template class ADVectorFunc<1, 3, 2>;
template class ADVectorFunc<1, 3, 3>;

template class ADVectorFunc<2, 1, 1>;
template class ADVectorFunc<2, 1, 2>;
template class ADVectorFunc<2, 1, 3>;

template class ADVectorFunc<2, 2, 1>;
template class ADVectorFunc<2, 2, 2>;
template class ADVectorFunc<2, 2, 3>;

template class ADVectorFunc<2, 3, 1>;
template class ADVectorFunc<2, 3, 2>;
template class ADVectorFunc<2, 3, 3>;

}
#endif
