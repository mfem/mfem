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

#ifndef ADMFEM_HPP 
#define ADMFEM_HPP

#include "mfem.hpp"
#include "fdual.hpp"
#include "tadvector.hpp"
#include "taddensemat.hpp"

#ifdef MFEM_USE_CODIPACK
#include <codi.hpp>

namespace mfem {

namespace ad {
#ifdef MFEM_USE_ADFORWARD
/// Forward AD type declaration
typedef codi::RealForward ADFloatType;
/// Vector type for AD-numbers
typedef TAutoDiffVector<ADFloatType> ADVectorType;
/// Matrix type for AD-numbers
typedef TAutoDiffDenseMatrix<ADFloatType> ADMatrixType;
#else
/// Reverse AD type declaration
typedef codi::RealReverse ADFloatType;
/// Vector type for AD-numbers
typedef TAutoDiffVector<ADFloatType> ADVectorType;
/// Matrix type for AD-numbers
typedef TAutoDiffDenseMatrix<ADFloatType> ADMatrixType;
#endif
}

/// The class provides an evaluation of the Jacobian of a
/// templated vector function provided in the constructor.
/// The Jacobian is evaluated with the help of automatic
/// differentiation (AD)
/// https://en.wikipedia.org/wiki/Automatic_differentiation.
/// The template parameters specify the size of the return
/// vector (vector_size), the size of the input vector
/// (state_size), and the size of the parameters supplied
/// to the function.
template<int vector_size=1, int state_size=1, int param_size=0>
class VectorFuncAutoDiff
{
public:
    /// F_ is user implemented function to be differentiated by VectorFuncAutoDiff.
    /// The signature of the function is:
    /// F_(mfem::Vector& parameters, ad::ADVectroType& state_vector, ad::ADVectorType& result).
    /// The parameters vector should have size param_size. The state_vector should have size
    /// state_size, and the result vector should have size vector_size. All size parameters are
    /// teplate parameters in VectorFuncAutoDiff.
    VectorFuncAutoDiff(std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F_)
    {
            F=F_;
    }

    /// Evaluates the Jacobian of the vector function F_ for a set of parameters (vparam) and
    /// state vector uu. The Jacobian (jac) has dimensions [vector_size x state_size].
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
#ifdef MFEM_USE_ADFORWARD
        // use forward mode
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i].setValue(uu[i]);
                aduu[i].setGradient(0.0);
            }
            for(int ii=0;ii<state_size;ii++){
                aduu[ii].setGradient(1.0);
                F(vparam,aduu,rr);
                for(int jj=0;jj<vector_size;jj++)
                {
                    jac(jj,ii)=rr[jj].getGradient();
                }
                aduu[ii].setGradient(0.0);
            }
        }
#else // use reverse mode
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i]=uu[i];
            }

            ad::ADFloatType::TapeType& tape =ad::ADFloatType::getGlobalTape();
            typename ad::ADFloatType::TapeType::Position pos=tape.getPosition();

            tape.setActive();
            for(int ii=0;ii<state_size;ii++){ tape.registerInput(aduu[ii]); }
            F(vparam,aduu,rr);
            for(int ii=0;ii<vector_size;ii++){ tape.registerOutput(rr[ii]); }
            tape.setPassive();

            for(int jj=0;jj<vector_size;jj++){
                rr[jj].setGradient(1.0);
                tape.evaluate();
                for(int ii=0;ii<state_size;ii++){
                    jac(jj,ii)=aduu[ii].getGradient();
                }
                tape.clearAdjoints();
                rr[jj].setGradient(0.0);
            }
            tape.reset(pos);
        }
#endif
    }
private:
    std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F;
}; //VectorFuncAutoDiff

/// The class provides an evaluation of the Jacobian of a templated
/// vector function provided as a functor TFunctor.  The Jacobian is
/// evaluated with the help of automatic differentiation (AD)
/// https://en.wikipedia.org/wiki/Automatic_differentiation.
/// The template parameters specify the size of the return
/// vector (vector_size), the size of the input vector (state_size),
/// and the size of the parameters supplied to the function.
/// The TFunctor functor is a template class with parameters [Float data type],
/// [Vector type for the additional parameters],
/// [Vector type for the state vector and the return residual].
/// The integer template parameters are the same ones
/// passed to QVectorFuncAutoDiff.
template<template<typename, typename, typename, int, int, int> class TFunctor
         , int vector_size=1, int state_size=1, int param_size=0>
class QVectorFuncAutoDiff
{
public:
    /// Evaluates the vector function for given set of parameters and state
    /// values in vector uu. The result is returned in vector rr.
    void QVectorFunc(const mfem::Vector &vparam, mfem::Vector &uu, mfem::Vector& rr)
    {
        rf(vparam,uu,rr);
    }

    /// Returns the gradient of TFunctor(...) in the dense matrix jac.
    /// The dimensions of jac are vector_size x state_size, where state_size is the
    /// length of vector uu.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
#ifdef MFEM_USE_ADFORWARD
        // use forward mode
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i].setValue(uu[i]);
                aduu[i].setGradient(0.0);
            }

            for(int ii=0;ii<state_size;ii++){
                aduu[ii].setGradient(1.0);
                tf(vparam,aduu,rr);
                for(int jj=0;jj<vector_size;jj++)
                {
                    jac(jj,ii)=rr[jj].getGradient();
                }
                aduu[ii].setGradient(0.0);
            }
        }
#else //end MFEM_USE_ADFORWARD
        // use reverse mode
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i]=uu[i];
            }

            ad::ADFloatType::TapeType& tape =ad::ADFloatType::getGlobalTape();
            typename ad::ADFloatType::TapeType::Position pos=tape.getPosition();

            tape.setActive();
            for(int ii=0;ii<state_size;ii++){ tape.registerInput(aduu[ii]); }
            tf(vparam,aduu,rr);
            for(int ii=0;ii<vector_size;ii++){ tape.registerOutput(rr[ii]); }
            tape.setPassive();
            for(int jj=0;jj<vector_size;jj++){
                rr[jj].setGradient(1.0);
                tape.evaluate();
                for(int ii=0;ii<state_size;ii++){
                    jac(jj,ii)=aduu[ii].getGradient();
                }
                tape.clearAdjoints();
                rr[jj].setGradient(0.0);
            }
            tape.reset(pos);
        }
#endif
    }
private:
   

    TFunctor<ad::ADFloatType, const Vector, ad::ADVectorType,
                vector_size, state_size, param_size> tf;

    TFunctor<double,const mfem::Vector, mfem::Vector,
                        vector_size, state_size, param_size> rf;

};

/// The class provides an evaluation of the first derivatives
/// and the Hessian of a templated scalar function provided as
///  a functor TFunctor. Both the first and the second derivatives
/// are evaluated with the help of automatic differentiation (AD)
/// https://en.wikipedia.org/wiki/Automatic_differentiation.
/// The template parameters specify the size of the input
/// vector (state_size) and the size of the parameters
/// supplied to the function. The TFunctor functor is a template
/// class with parameters [Float data type],
/// [Vector type for the additional parameters],
/// [Vector type for the state vector and the return residual].
/// The integer template parameters are the same ones passed
/// to QFunctionAutoDiff.
template<template<typename, typename, typename, int, int> class TFunctor
         , int state_size=1, int param_size=0>
class QFunctionAutoDiff
{
public:

    /// Evaluates a function for arguments vparam and uu.
    /// The evaluation is based on the operator() in the
    /// user provided functor TFunctor.
    double QEval(const mfem::Vector &vparam, mfem::Vector &uu)
    {
        return rf(vparam,uu);
    }

    /// Provides the same functionality as QGrad.
    void QVectorFunc(const mfem::Vector &vparam, mfem::Vector &uu, mfem::Vector &rr)
    {
        QGrad(vparam,uu,rr);
    }

    /// Returns the first derivative of TFunctor(...) with
    /// respect to the active arguments proved in vector uu.
    /// The length of rr is the same as for uu.
    void QGrad(const mfem::Vector &vparam, mfem::Vector &uu, mfem::Vector &rr)
    {

#ifdef MFEM_USE_ADFORWARD
        // use forward mode
        rr.SetSize(state_size);
        {
            ad::ADVectorType aduu(state_size);
            for(int i=0;i<state_size;i++){
                aduu[i].setValue(uu[i]);
                aduu[i].setGradient(0.0);
            }

            ad::ADFloatType rez;

            for(int ii=0;ii<state_size;ii++){
                aduu[ii].setGradient(1.0);
                rez=tf(vparam,aduu);
                rr[ii]=rez.getGradient();
                aduu[ii].setGradient(0.0);
            }
        }

#else
        {
            ad::ADVectorType aduu(state_size);
            ad::ADFloatType rez;
            for(int i=0;i<state_size;i++){
                aduu[i]=uu[i];
            }

            ad::ADFloatType::TapeType& tape =ad::ADFloatType::getGlobalTape();
            typename ad::ADFloatType::TapeType::Position pos=tape.getPosition();

            tape.setActive();
            for(int ii=0;ii<state_size;ii++){ tape.registerInput(aduu[ii]); }

            rez=tf(vparam,aduu);
            tape.registerOutput(rez);
            tape.setPassive();

            rez.setGradient(1.0);
            tape.evaluate();
            for(int i=0;i<state_size;i++){
                rr[i]=aduu[i].getGradient();
            }
            tape.reset(pos);
        }
#endif
    }

    /// Provides same functionality as QHessian.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        QHessian(vparam,uu,jac);
    }

#ifdef MFEM_USE_ADFORWARD
    // use forward-forward mode
    typedef codi::RealForwardGen<double>    ADFType;
    typedef TAutoDiffVector<ADFType>              ADFVector;
    typedef TAutoDiffDenseMatrix<ADFType>         ADFDenseMatrix;

    typedef codi::RealForwardGen<ADFType>   ADSType;
    typedef TAutoDiffVector<ADSType>              ADSVector;
    typedef TAutoDiffDenseMatrix<ADSType>         ADSDenseMatrix;
#else
    //use mixed forward and reverse mode
    typedef codi::RealForwardGen<double> ADFType;
    typedef TAutoDiffVector<ADFType>           ADFVector;
    typedef TAutoDiffDenseMatrix<ADFType>      ADFDenseMatrix;

    typedef codi::RealReverseGen<ADFType>   ADSType;
    typedef TAutoDiffVector<ADSType>              ADSVector;
    typedef TAutoDiffDenseMatrix<ADSType>         ADSDenseMatrix;
#endif


    /// Returns the Hessian of TFunctor(...) in the dense matrix jac.
    /// The dimensions of jac are state_size x state_size, where state_size is the
    /// length of vector uu.
    void QHessian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
#ifdef MFEM_USE_ADFORWARD
        // use forward-forward mode
        jac.SetSize(state_size);
        jac=0.0;
        {
            ADSVector aduu(state_size);
            for(int ii = 0;  ii < state_size; ii++)
            {
                aduu[ii].value().value()=uu[ii];
                aduu[ii].value().gradient()=0.0;
                aduu[ii].gradient().value()=0.0;
                aduu[ii].gradient().gradient()=0.0;
            }

            for(int ii = 0; ii < state_size; ii++)
            {
                aduu[ii].value().gradient()=1.0;
                for(int jj=0; jj<(ii+1); jj++)
                {
                    aduu[jj].gradient().value()=1.0;
                    ADSType rez=sf(vparam,aduu);
                    jac(ii,jj)=rez.gradient().gradient();
                    jac(jj,ii)=jac(ii,jj);
                    aduu[jj].gradient().value()=0.0;
                }
                aduu[ii].value().gradient()=0.0;
            }
        }
#else
        //use mixed forward and reverse mode
        jac.SetSize(state_size);
        jac=0.0;
        {
            ADSVector aduu(state_size);
            for(int ii=0;ii < state_size ; ii++)
            {
                aduu[ii].value().value()=uu[ii];
            }

            ADSType rez;

            ADSType::TapeType& tape = ADSType::getGlobalTape();
            typename ADSType::TapeType::Position pos;
            for(int ii = 0; ii < state_size ; ii++)
            {
                pos=tape.getPosition();
                tape.setActive();

                for(int jj=0;jj < state_size; jj++) {
                    if(jj==ii) {aduu[jj].value().gradient()=1.0;}
                    else {aduu[jj].value().gradient()=0.0;}
                    tape.registerInput(aduu[jj]);
                }

                rez=sf(vparam,aduu);
                tape.registerOutput(rez);
                tape.setPassive();

                rez.gradient().value()=1.0;
                tape.evaluate();

                for(int jj=0; jj<(ii+1); jj++)
                {
                    jac(ii,jj)=aduu[jj].gradient().gradient();
                    jac(jj,ii)=jac(ii,jj);
                }
                tape.reset(pos);
            }

        }
#endif
    }
private:
    TFunctor<double, const mfem::Vector, 
	    mfem::Vector, state_size, param_size> rf;

    TFunctor<ad::ADFloatType, const mfem::Vector,
	    ad::ADVectorType, state_size, param_size> tf;

    TFunctor<ADSType, const mfem::Vector, ADSVector, 
	    state_size, param_size> sf;
};

}
#else //end MFEM_USE_CODIPACK

//USE NATIVE IMPLEMENTATION
namespace mfem {

namespace ad {
/// MFEM native forward AD-type
typedef FDualNumber<double> ADFloatType;
/// Vector type for AD-type numbers
typedef TAutoDiffVector<ADFloatType> ADVectorType;
/// Matrix type for AD-type numbers
typedef TAutoDiffDenseMatrix<ADFloatType> ADMatrixType;
}

/// The class provides an evaluation of the Jacobian of a
/// templated vector function provided in the constructor.
/// The Jacobian is evaluated with the help of automatic
/// differentiation (AD)
/// https://en.wikipedia.org/wiki/Automatic_differentiation.
/// The template parameters specify the size of the return
/// vector (vector_size), the size of the input vector
/// (state_size), and the size of the parameters supplied
/// to the function.
template<int vector_size=1, int state_size=1, int param_size=0>
class VectorFuncAutoDiff
{
public:
    /// F_ is user implemented function to be differentiated by VectorFuncAutoDiff.
    /// The signature of the function is:
    /// F_(mfem::Vector& parameters, ad::ADVectroType& state_vector, ad::ADVectorType& result).
    /// The parameters vector should have size param_size. The state_vector should have size
    /// state_size, and the result vector should have size vector_size. All size parameters are
    /// teplate parameters in VectorFuncAutoDiff. 	
    VectorFuncAutoDiff(std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F_)
    {
            F=F_;
    }

    /// Evaluates the Jacobian of the vector function F_ for a set of parameters (vparam) and
    /// state vector uu. The Jacobian (jac) has dimensions [vector_size x state_size].
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
           ad::ADVectorType aduu(uu); //all dual numbers are initialized to zero
           ad::ADVectorType rr(vector_size);

           for (int ii = 0; ii < state_size; ii++)
           {
              aduu[ii].dual(1.0);
              F(vparam,aduu,rr);
              for (int jj = 0; jj < vector_size; jj++)
              {
                 jac(jj, ii) = rr[jj].dual();
              }
              aduu[ii].dual(0.0);
           }
        }
    }

private:
    std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F;

};


/// The class provides an evaluation of the Jacobian of a templated
/// vector function provided as a functor TFunctor.  The Jacobian is
/// evaluated with the help of automatic differentiation (AD)
/// https://en.wikipedia.org/wiki/Automatic_differentiation.
/// The template parameters specify the size of the return
/// vector (vector_size), the size of the input vector (state_size),
/// and the size of the parameters supplied to the function.
/// The TFunctor functor is a template class with parameters [Float data type],
/// [Vector type for the additional parameters],
/// [Vector type for the state vector and the return residual].
/// The integer template parameters are the same ones
/// passed to QVectorFuncAutoDiff. \n
/// Example: f={sin(a*x*y), cos(b*x*y*z), x*x+y*x} \n
/// The vector function has vector_size=3, and state_size=3, i.e., it has 
/// three arguments [x,y,z]. The parameters are [a,b] and the vector has 
/// size 2.  The functor class will have the following form 
/// \code{.cpp}
/// template<typename TDataType, typename TParamVector, typename TStateVector,
///         int residual_size, int state_size, int param_size>
/// class MyVectorFunction{
/// public:
/// TDataType operator() (TParamVector& vparam, TStateVector& uu, TStateVector& rr)
/// {
///    auto a=vparam[0];
///    auto b=vparam[1];
///    rr[0]=sin(a*uu[0]*uu[1]);     
///    rr[1]=cos(b*uu[0]*uu[1]*uu[2]);
///    rr[2]=uu[0]*uu[0]+uu[0]*uu[1];
/// }
//
/// };
/// \endcode
template<template<typename, typename, typename, int, int, int> class TFunctor
         , int vector_size=1, int state_size=1, int param_size=0>
class QVectorFuncAutoDiff
{
private:
    /// MFEM native forward AD-type
    typedef ad::FDualNumber<double> ADFType;
    /// Vector type for AD-type numbers
    typedef TAutoDiffVector<ADFType> ADFVector;
    /// Matrix type for AD-type numbers
    typedef TAutoDiffDenseMatrix<ADFType> ADFDenseMatrix;

public:
    /// Returns a vector valued function rr for supplied passive arguments
    /// vparam and active arguments uu. The evaluation is based on the
    /// user supplied TFunctor template class.
    void QVectorFunc(const Vector &vparam, Vector &uu, Vector &rr)
    {
       func(vparam, uu, rr);
    }

    /// Returns the gradient of TFunctor(...) residual in the dense matrix jac.
    /// The dimensions of jac are vector_size x state_size, where state_size is the
    /// length of vector uu.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        //use native AD package
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
           ADFVector aduu(uu); //all dual numbers are initialized to zero
           ADFVector rr(vector_size);

           for (int ii = 0; ii < state_size; ii++)
           {
              aduu[ii].dual(1.0);
              QEval(vparam, aduu, rr);
              for (int jj = 0; jj < vector_size; jj++)
              {
                 jac(jj, ii) = rr[jj].dual();
              }
              aduu[ii].dual(0.0);
           }
        }
    }

private:
    /// Evaluates the residual from TFunctor(...).
    /// Intended for internal use only.
    void QEval(const Vector &vparam, ADFVector &uu, ADFVector &rr)
    {
       tf(vparam, uu, rr);
    }

    TFunctor<double, const Vector, Vector,
               vector_size, state_size, param_size> func;

    TFunctor<ADFType, const Vector, ADFVector,
               vector_size, state_size, param_size> tf;

};

/// The class provides an evaluation of the first derivatives
/// and the Hessian of a templated scalar function provided as
///  a functor TFunctor. Both the first and the second derivatives
/// are evaluated with the help of automatic differentiation (AD)
/// https://en.wikipedia.org/wiki/Automatic_differentiation.
/// The template parameters specify the size of the input
/// vector (state_size) and the size of the parameters
/// supplied to the function. The TFunctor functor is a template
/// class with parameters [Float data type],
/// [Vector type for the additional parameters],
/// [Vector type for the state vector and the return residual].
/// The integer template parameters are the same ones passed
/// to QFunctionAutoDiff.
template<template<typename, typename, typename, int, int> class TFunctor
         , int state_size=1, int param_size=0>
class QFunctionAutoDiff
{
private:
    /// MFEM native AD-type for first derivatives
    typedef ad::FDualNumber<double> ADFType;
    /// Vector type for AD-numbers(first derivatives)
    typedef TAutoDiffVector<ADFType> ADFVector;
    /// Matrix type for AD-numbers(first derivatives)
    typedef TAutoDiffDenseMatrix<ADFType> ADFDenseMatrix;
    /// MFEM native AD-type for second derivatives
    typedef ad::FDualNumber<ADFType> ADSType;
    /// Vector type for AD-numbers (second derivatives)
    typedef TAutoDiffVector<ADSType> ADSVector;
    /// Vector type fpr AD-numbers (second derivatives)
    typedef TAutoDiffDenseMatrix<ADSType> ADSDenseMatrix;

public:
    /// Evaluates a function for arguments vparam and uu.
    /// The evaluatin is based on the operator() in the
    /// user provided functor TFunctor.
    double QEval(const Vector &vparam, Vector &uu)
    {
        return tf(vparam,uu);
    }

    /// Provides the same functionality as QGrad.
    void QVectorFunc(const Vector &vparam, Vector &uu, Vector &rr)
    {
        QGrad(vparam,uu,rr);
    }

    /// Returns the first derivative of TFunctor(...) with
    /// respect to the active arguments proved in vector uu.
    /// The length of rr is the same as for uu.
    void QGrad(const Vector &vparam, Vector &uu, Vector &rr)
    {
        int n = uu.Size();
        rr.SetSize(n);
        ADFVector aduu(uu);
        ADFType rez;
        for (int ii = 0; ii < n; ii++)
        {
           aduu[ii].dual(1.0);
           rez = ff(vparam, aduu);
           rr[ii] = rez.dual();
           aduu[ii].dual(0.0);
        }
    }

    /// Provides same functionality as QHessian.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        QHessian(vparam,uu,jac);
    }

    /// Returns the Hessian of TFunctor(...) in the dense matrix jac.
    /// The dimensions of jac are state_size x state_size, where state_size is the
    /// length of vector uu.
    void QHessian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        int n = uu.Size();
        jac.SetSize(n);
        jac = 0.0;
        {
           ADSVector aduu(n);
           for (int ii = 0; ii < n; ii++)
           {
              aduu[ii].real(ADFType(uu[ii], 0.0));
              aduu[ii].dual(ADFType(0.0, 0.0));
           }

           for (int ii = 0; ii < n; ii++)
           {
              aduu[ii].real(ADFType(uu[ii], 1.0));
              for (int jj = 0; jj < (ii + 1); jj++)
              {
                 aduu[jj].dual(ADFType(1.0, 0.0));
                 ADSType rez = sf(vparam, aduu);
                 jac(ii, jj) = rez.dual().dual();
                 jac(jj, ii) = rez.dual().dual();
                 aduu[jj].dual(ADFType(0.0, 0.0));
              }
              aduu[ii].real(ADFType(uu[ii], 0.0));
           }
        }
    }

private:
    TFunctor<double, const Vector, Vector, state_size, param_size> tf;
    TFunctor<ADFType, const Vector, ADFVector, state_size, param_size> ff;
    TFunctor<ADSType, const Vector, ADSVector, state_size, param_size> sf;

};
}//end namespace mfem
#endif // NATIVE
#endif // ADMFEM_HPP

