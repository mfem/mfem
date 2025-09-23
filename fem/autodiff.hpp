
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

#include <vector>
#include <functional>
#include "../linalg/vector.hpp"
#include "../linalg/densemat.hpp"

#ifdef MFEM_USE_CODIPACK
#include <codi.hpp>
#endif

using namespace std::placeholders;

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
/// Default to standard real
typedef real_t ADFloat;
#endif

/// Vector type for ADFloat
typedef std::vector<ADFloat> ADVector;

/// This helper class provides evaluation of first order derivatives, namely
/// gradient, curl and diverence, of a vector function provided in the
/// constructor. The derivatives are evaluated with the help of automatic
/// differentiation (AD). The derivative member functions are provided in a format
/// compatbile with the coefficient class.
class ADVectorFunc
{
private:
   std::function<void(const Vector&, const ADVector&, ADVector&)> F;
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
      std::function<void(const Vector&, const ADVector&, ADVector&)> F_,
      Vector &par)
   {
      F=F_;
      param = par;
   }

   ADVectorFunc(
      std::function<void(const Vector&, const ADVector&, ADVector&)> F_)
   {
      F=F_;
   }

   /// Evaluates the Jacobian of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor. The Jacobian (jac) has dimensions
   /// [vector_size x state_size].
   void Jacobian(const Vector &vstate,DenseMatrix &jac);

   /// Evaluates the Curl of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor. The Curl only works in 3D.
   void Curl(const Vector &vstate, Vector &curl);

   /// Evaluates the Divergence of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor.
   real_t Divergence(const Vector &vstate);

   /// Evaluates the Gradient of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The Gradient (grad) has dimensions
   /// [ state_size].
   void Gradient(const Vector &vstate, Vector &grad);

   /// Evaluates the Solution of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The Solution (sol) has dimensions
   /// [ state_size]. The function and parameters are provided in the Constructor.
   void Solution(const Vector &vstate, Vector &sol);

   /// Evaluates the Solution of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate.  The function and parameters are
   /// provided in the Constructor.
   real_t ScalarSolution(const Vector &vstate);

   /// Function type definitions
   using ScalFunc = real_t(const Vector &);
   using VecFunc = void(const Vector &, Vector &);
   using MatFunc = void(const Vector &, DenseMatrix &);

   /// Access function to get the Jacobian Function
   std::function<MatFunc> GetJacobian()
   {
      return std::bind(&ADVectorFunc::Jacobian, *this, _1, _2);
   }

   /// Access function to get the Curl Function
   std::function<VecFunc> GetCurl()
   {
      return std::bind(&ADVectorFunc::Curl, *this, _1, _2);
   }

   /// Access function to get the Divergence Function
   std::function<ScalFunc> GetDivergence()
   {
      return std::bind(&ADVectorFunc::Divergence, *this, _1);
   }

   /// Access function to get the Gradient Function
   std::function<VecFunc> GetGradient()
   {
      return std::bind(&ADVectorFunc::Gradient, *this, _1, _2);
   }

   /// Access function to get the Scalar Function
   std::function<VecFunc> GetSolution()
   {
      return std::bind(&ADVectorFunc::Solution, *this, _1, _2);
   }

   /// Evaluates the Solution of the vector function F_ for a set of parameters
   std::function<ScalFunc> GetScalarSolution()
   {
      return std::bind(&ADVectorFunc::ScalarSolution, *this, _1);
   }
};

/// This helper class provides evaluation of first order derivatives, namely
/// gradient, curl and diverence, of a vector function provided in the
/// constructor. The derivatives are evaluated with the help of automatic
/// differentiation (AD). The derivative member functions are provided in a format
/// compatbile with the coefficient class.
class ADVectorTDFunc
{
private:
   std::function<void(const Vector&, const ADVector&, const ADFloat&, ADVector&)>
   F;
   Vector param;

public:
   /// F_ is user implemented function to be differentiated by
   /// ADVectorFunc. The signature of the function is: F_(Vector&
   /// parameters, ad::ADVectorType& state_vector, ad::ADVectorType& result).
   /// The parameters vector should have size param_size. The state_vector
   /// should have size state_size, and the result vector should have size
   /// vector_size. All size parameters are teplate parameters in
   /// ADVectorFunc.
   ADVectorTDFunc(
      std::function<void(const Vector&, const ADVector&, const ADFloat&, ADVector&)>
      F_,
      Vector &par)
   {
      F=F_;
      param = par;
   }

   ADVectorTDFunc(
      std::function<void(const Vector&, const ADVector&, const ADFloat&, ADVector&)>
      F_)
   {
      F=F_;
   }

   /// Evaluates the Jacobian of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor. The Jacobian (jac) has dimensions
   /// [vector_size x state_size].
   void Jacobian(const Vector &vstate, const real_t time, DenseMatrix &jac);

   /// Evaluates the Curl of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor. The Curl only works in 3D.
   void Curl(const Vector &vstate, const real_t time, Vector &curl);

   /// Evaluates the Divergence of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The function and parameters are
   /// provided in the Constructor.
   real_t Divergence(const Vector &vstate, const real_t time);

   /// Evaluates the Gradient of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The Gradient (grad) has dimensions
   /// [ state_size].
   void Gradient(const Vector &vstate, const real_t time, Vector &grad);

   /// Evaluates the Rate of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The Solution (sol) has dimensions
   /// [ state_size]. The function and parameters are provided in the Constructor.
   void Rate(const Vector &vstate, const real_t time, Vector &sol);

   /// Evaluates the Rate of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate.  The function and parameters are
   /// provided in the Constructor.
   real_t ScalarRate(const Vector &vstate, const real_t time);

   /// Evaluates the Solution of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate. The Solution (sol) has dimensions
   /// [ state_size]. The function and parameters are provided in the Constructor.
   void Solution(const Vector &vstate, const real_t time, Vector &sol);

   /// Evaluates the Solution of the vector function F_ for a set of parameters
   /// (vparam) and state vector vstate.  The function and parameters are
   /// provided in the Constructor.
   real_t ScalarSolution(const Vector &vstate, const real_t time);

   /// Function type definitions
   using ScalTDFunc = real_t(const Vector &, const real_t time);
   using VecTDFunc = void(const Vector &, const real_t time, Vector &);
   using MatTDFunc = void(const Vector &, const real_t time, DenseMatrix &);

   /// Access function to get the Jacobian Function
   std::function<MatTDFunc> GetJacobian()
   {
      return std::bind(&ADVectorTDFunc::Jacobian, *this, _1, _2, _3);
   }

   /// Access function to get the Curl Function
   std::function<VecTDFunc> GetCurl()
   {
      return std::bind(&ADVectorTDFunc::Curl, *this, _1, _2, _3);
   }

   /// Access function to get the Divergence Function
   std::function<ScalTDFunc> GetDivergence()
   {
      return std::bind(&ADVectorTDFunc::Divergence, *this, _1, _2);
   }

   /// Access function to get the Gradient Function
   std::function<VecTDFunc> GetGradient()
   {
      return std::bind(&ADVectorTDFunc::Gradient, *this, _1, _2, _3);
   }

   /// Access function to get the Rate Function
   std::function<VecTDFunc> GetRate()
   {
      return std::bind(&ADVectorTDFunc::Rate, *this, _1, _2, _3);
   }

   /// Access function to get the Scalar Rate Function
   std::function<ScalTDFunc> GetScalarRate()
   {
      return std::bind(&ADVectorTDFunc::ScalarRate, *this, _1, _2);
   }

   /// Access function to get the Solution Function
   std::function<VecTDFunc> GetSolution()
   {
      return std::bind(&ADVectorTDFunc::Solution, *this, _1, _2, _3);
   }

   /// Access function to get the Scalar Solution Function
   std::function<ScalTDFunc> GetScalarSolution()
   {
      return std::bind(&ADVectorTDFunc::ScalarSolution, *this, _1, _2);
   }

};

}
#endif
