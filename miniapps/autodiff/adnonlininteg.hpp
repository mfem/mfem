// Copyright (c) 2020, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_ADNONLININTEG
#define MFEM_ADNONLININTEG

#include "mfem.hpp"
#include "tadvector.hpp"
#include "taddensemat.hpp"
#include "fdual.hpp"

#if defined MFEM_USE_ADEPT
#include <adept.h>
#elif defined MFEM_USE_FADBADPP
#include <fadiff.h>
#include <badiff.h>
#endif


namespace mfem
{

///Automatic differentiation class - for user coded functor CTD returning  PDE's
/// energy and residual at a point, provides the derivative of the residual
/// with respect to the active arguments of the functor.
///
/**
   The two templated parameters are
   template<typename, typename> class CTD and int m.
   The templated class CTD is a user-provided functor which implements
   the function of interest. The CTD class should have the following signature:

template<typename DType, typename MVType>

class MyQFunctorJ

{

    DType operator()(const Vector &vparam, MVType &uu)

        {...}

    void operator()(const Vector &vparam, MVType &uu, MVType &rr)

    {...}

};


DType - represents the scalars in the functor, and  MVType - represents
the vectors in the functor. In ADQFunctionTJ,  DType will be replaced
either with double or AD-type.
The first operator()(const Vector &vparam, MVType &uu) takes as input
standard MFEM Vector vparam (passive arguments), which holds all parameters
 supplied to the functor and MVType vector uu, which holds all active
arguments provided to the functor. The operator returns a scalar
representing the value of the function(energy) for passive parameters vparam
and active arguments uu.


The second void operator()(const Vector &vparam, MVType &uu, MVType &rr)
returns a vector-valued function rr with passive input arguments vparam
and active (AD) arguments supplied in vector uu. Since the size of the
return vector rr is not known in advance, it should be provided explicitly
to the ADQFunctionTJ class, i.e., the template parameter m in ADQFunctionTJ
represents the dimensions of the return vector rr.


For PDE discretized problem, the first operator should return the energy
evaluated a point, and the second operator should return the residual
evaluated at the point. The derivative of the residual vector rr with
respect to the active arguments uu is evaluated automatically by
ADQFunctionTJ and returned by the ADQFunctionTJ:: QFunctionDD(..) method.
*/
template<template<typename, typename> class CTD, int m>
class ADQFunctionTJ
{
   // m - dimension of the residual vector
   // the Jacobian will have dimensions [m,length(uu)]
protected:
#ifdef MFEM_USE_ADEPT
   adept::Stack m_stack;
#endif

public:
#if defined MFEM_USE_ADEPT
   /// AD-type based on the Adept AD library.
   typedef adept::adouble ADFType;
   typedef TADVector<ADFType> ADFVector;
   typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
#elif defined MFEM_USE_FADBADPP
#ifdef MFEM_USE_ADFORWARD
   /// AD-type based on the FADBAD++ library -
   /// used only for forward differentiation.
   typedef fadbad::F<double> ADFType;
   typedef TADVector<ADFType> ADFVector;
   typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
#else
   /// AD-type based on the FADBAD++ library -
   /// used only for reverse AD mode.
   typedef fadbad::B<double> ADFType;
   typedef TADVector<ADFType> ADFVector;
   typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
#endif
#else
   /// MFEM native forward AD-type
   typedef ad::FDual<double> ADFType;
   typedef TADVector<ADFType> ADFVector;
   typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
#endif

#ifdef MFEM_USE_ADEPT
   ADQFunctionTJ() : m_stack(false) {}
#else
   ADQFunctionTJ() {}
#endif

   ~ADQFunctionTJ() {}

   ///Returns the energy for passive arguments vparam and
   /// active arguments uu. The evaluation is based on the
   /// first operator in the user-supplied CTD template class.
   double QFunction(const Vector &vparam, Vector &uu)
   {
      CTD<double, Vector> func;
      return func(vparam, uu);
   }

   ///Returns vector valued function rr for passive input parametrs vparam
   /// and active arguments provided in uu.
   void QFunctionDU(const Vector &vparam, ADFVector &uu, ADFVector &rr)
   {
      CTD<ADFType, ADFVector> func;
      func(vparam, uu, rr);
   }

   ///Evaluates automatically the first derivative of
   /// QFunction(const Vector &vparam, Vector &uu) with respect to
   /// the active vector arguments uu. The dimension or rr is the same
   /// as the dimension of uu. The method can be used for testing
   /// the correctness of hand-coded gradients of  QFunction(...).
   void QFunctionAU(const Vector &vparam, Vector &uu, Vector &rr)
   {
      //the result is computed automatically by differentiating
      //QFunction with respect to uu
      CTD<ADFType, ADFVector> func;
      int n = uu.Size();
      rr.SetSize(n);

#if defined MFEM_USE_ADEPT
      //use ADEPT package
      adept::Stack *p_stack = adept::active_stack();
      p_stack->deactivate();

      m_stack.activate();
      {
         ADFVector aduu(uu);
         ADFType rez;
         m_stack.new_recording();
         rez = func(vparam, aduu);
         m_stack.independent(aduu.GetData(), n); //independent variables
         m_stack.dependent(&rez, 1);             //dependent variables
         m_stack.jacobian(rr.GetData());
      }
      m_stack.deactivate();
#elif defined MFEM_USE_FADBADPP
      //use FADBAD++
#ifdef MFEM_USE_ADFORWARD
      {
         ADFVector aduu(uu);
         ADFType rez;
         for (int ii = 0; ii < n; ii++)
         {
            aduu[ii].diff(ii, n);
         }
         rez = func(vparam, aduu);
         for (int ii = 0; ii < n; ii++)
         {
            rr[ii] = rez.d(ii);
         }
      }
#else
      {
         ADFVector aduu(uu);
         ADFType rez;
         rez = func(vparam, aduu);
         rez.diff(0, 1);
         for (int ii = 0; ii < n; ii++)
         {
            rr[ii] = aduu[ii].d(0);
         }
      }
#endif
#else
      //use native AD package
      {
         ADFVector aduu(uu); //all dual numbers are initialized to zero
         ADFType rez;

         for (int ii = 0; ii < n; ii++)
         {
            aduu[ii].dual(1.0);
            rez = func(vparam, aduu);
            rr[ii] = rez.dual();
            aduu[ii].dual(0.0);
         }
      }
#endif
   }

   ///Returns vector valued function rr for supplied passive arguments
   /// vparam and active arguments uu. The evaluation is based on the
   /// user supplied CTD template class (second operator).
   void QFunctionDU(const Vector &vparam, Vector &uu, Vector &rr)
   {
      CTD<double, Vector> func;
      func(vparam, uu, rr);
   }

   ///Evaluates automatically the derivative or the vector function
   /// QFunctionDU(...), i.e., the retuned vector rr, with respect to
   /// the active arguments uu. The dimensions of the the dense matix
   /// jac are [m,n] where m is the size of the vector rr and n is the
   /// size of the active vector uu. The parameter m should be supplied as
   /// template parameter int m in ADQFunctionTJ.
   void QFunctionDD(const Vector &vparam, Vector &uu, DenseMatrix &jac)
   {
#if defined MFEM_USE_ADEPT
      //use ADEPT package
      adept::Stack *p_stack = adept::active_stack();
      p_stack->deactivate();

      int n = uu.Size();
      jac.SetSize(m, n);
      jac = 0.0;
      m_stack.activate();
      {
         ADFVector aduu(uu);
         ADFVector rr(m); //residual vector
         m_stack.new_recording();
         QFunctionDU(vparam, aduu, rr);
         m_stack.independent(aduu.GetData(), n); //independent variables
         m_stack.dependent(rr.GetData(), m);     //dependent variables
         m_stack.jacobian(jac.Data());
      }
      m_stack.deactivate();
#elif defined MFEM_USE_FADBADPP
      //use FADBAD++
#ifdef MFEM_USE_ADFORWARD
      int n = uu.Size();
      jac.SetSize(m, n);
      jac = 0.0;
      {
         ADFVector aduu(uu);
         ADFVector rr(m);

         for (int ii = 0; ii < n; ii++)
         {
            aduu[ii].diff(ii, n);
         }
         QFunctionDU(vparam, aduu, rr);
         for (int ii = 0; ii < n; ii++)
         {
            for (int jj = 0; jj < m; jj++)
            {
               jac(jj, ii) = rr[jj].d(ii);
            }
         }
      }
#else
      int n = uu.Size();
      jac.SetSize(m, n);
      jac = 0.0;
      {
         ADFVector aduu(uu);
         ADFVector rr(m);
         QFunctionDU(vparam, aduu, rr);
         for (int ii = 0; ii < m; ii++)
         {
            rr[ii].diff(ii, m);
         }
         for (int ii = 0; ii < n; ii++)
         {
            for (int jj = 0; jj < m; jj++)
            {
               jac(jj, ii) = aduu[ii].d(jj);
            }
         }
      }
#endif
#else
      //use native AD package
      int n = uu.Size();
      jac.SetSize(m, n);
      jac = 0.0;
      {
         ADFVector aduu(uu); //all dual numbers are initialized to zero
         ADFVector rr(m);

         for (int ii = 0; ii < n; ii++)
         {
            aduu[ii].dual(1.0);
            QFunctionDU(vparam, aduu, rr);
            for (int jj = 0; jj < m; jj++)
            {
               jac(jj, ii) = rr[jj].dual();
            }
            aduu[ii].dual(0.0);
         }
      }
#endif
   }
};

//template class for differentiation; the function
//for differentiation is supplied as a functor
//the operator()(scalar,vector) defines the actual function

///ADQFunctionTH is a templated class evaluating the first and
/// the second derivatives of a user supplied function encoded
/// in a functor class CTD. The signature of CTD is as follows:
/**

template<typename DType, typename MVType> class CTD

{

    DType operator()(const Vector &vparam, MVType &uu)
    {...}

};

The operator returns a scalar representing the value of the
function for passive arguments vparam and active arguments uu.
The first derivative of the operator is evaluated automatically
by ADQFunctionTH::QFunctionDU(const Vector &vparam, Vector &uu, Vector &rr)
method. The length of the return vector rr is the same as for
the input vector uu.  The second derivate (the Hessian) of the function
is evaluated again automatically by
QFunctionDD(const Vector &vparam, const Vector &uu, DenseMatrix &jac).

Hessian evaluation is an expensive process. The provided AD-functionality
is intended to be utilized for development purposes. The performance will
increase significantly by replacing the AD-provided derivatives with hand-coded.
*/
template<template<typename, typename> class CTD>
class ADQFunctionTH
{
public:
#if defined MFEM_USE_FADBADPP
   ///AD-type derived from FADBAD++
   typedef fadbad::B<double> ADFType;
   ///AD vector type for evaluation of first derivatives
   typedef TADVector<ADFType> ADFVector;
   ///AD dense matrix type for evaluation of first derivatives
   typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
   ///AD-type for the second derivatives derived from
   ///FADBAD++
   typedef fadbad::B<fadbad::F<double>> ADSType;
   /// AD vector type for evaluation of second derivatives
   typedef TADVector<ADSType> ADSVector;
   /// AD dense matrix type for  evaluation of second derivatives
   typedef TADDenseMatrix<ADSType> ADSDenseMatrix;
#else
   ///MFEM native AD-type for first derivatives
   typedef ad::FDual<double> ADFType;
   typedef TADVector<ADFType> ADFVector;
   typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
   ///MFEM native AD-type for second derivatives
   typedef ad::FDual<ADFType> ADSType;
   typedef TADVector<ADSType> ADSVector;
   typedef TADDenseMatrix<ADSType> ADSDenseMatrix;
#endif

   ADQFunctionTH() {}

   ~ADQFunctionTH() {}

   ///Evaluates a function for arguments vparam and uu.
   /// The evaluatin is based on the operator() in the
   /// user provided functor CTD.
   double QFunction(const Vector &vparam, Vector &uu)
   {
      CTD<double, Vector> tf;
      return tf(vparam, uu);
   }

   ///Evaluates the first derivative of QFunction(...).
   /// Intended for internal use only.
   ADFType QFunction(const Vector &vparam, ADFVector &uu)
   {
      CTD<ADFType, ADFVector> tf;
      return tf(vparam, uu);
   }

   ///Evaluates the second derivative of QFunction(...).
   /// Intended for internal use only.
   ADSType QFunction(const Vector &vparam, ADSVector &uu)
   {
      CTD<ADSType, ADSVector> tf;
      return tf(vparam, uu);
   }

   ///Returns the first derivative of QFunction(...) with
   /// respect to the active arguments proved in vector uu.
   /// The length of rr is the same as for uu.
   void QFunctionDU(const Vector &vparam, Vector &uu, Vector &rr)
   {
#if defined MFEM_USE_FADBADPP
      int n = uu.Size();
      rr.SetSize(n);
      ADFVector aduu(uu);
      ADFType rez;
      rez = QFunction(vparam, aduu);
      rez.diff(0, 1);
      for (int ii = 0; ii < n; ii++)
      {
         rr[ii] = aduu[ii].d(0);
      }
#else
      int n = uu.Size();
      rr.SetSize(n);
      ADFVector aduu(uu);
      ADFType rez;
      for (int ii = 0; ii < n; ii++)
      {
         aduu[ii].dual(1.0);
         rez = QFunction(vparam, aduu);
         rr[ii] = rez.dual();
         aduu[ii].dual(0.0);
      }
#endif
   }

   ///Returns the Hessian of QFunction(...) in the dense matrix jac.
   /// The dimensions of jac are m x m, where m is the length of vector uu.
   void QFunctionDD(const Vector &vparam, const Vector &uu, DenseMatrix &jac)
   {
#if defined MFEM_USE_FADBADPP
      int n = uu.Size();
      jac.SetSize(n);
      jac = 0.0;
      {
         ADSVector aduu(n);
         for (int ii = 0; ii < n; ii++)
         {
            aduu[ii] = uu[ii];
            aduu[ii].x().diff(ii, n);
         }
         ADSType rez = QFunction(vparam, aduu);
         rez.diff(0, 1);
         for (int ii = 0; ii < n; ii++)
         {
            for (int jj = 0; jj < ii; jj++)
            {
               jac(ii, jj) = aduu[ii].d(0).d(jj);
               jac(jj, ii) = aduu[jj].d(0).d(ii);
            }
            jac(ii, ii) = aduu[ii].d(0).d(ii);
         }
      }
#else
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
               ADSType rez = QFunction(vparam, aduu);
               jac(ii, jj) = rez.dual().dual();
               jac(jj, ii) = rez.dual().dual();
               aduu[jj].dual(ADFType(0.0, 0.0));
            }
            aduu[ii].real(ADFType(uu[ii], 0.0));
         }
      }
#endif
   }

};

} // namespace mfem

#endif
