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



#ifndef MFEM_ADNONLININTEG
#define MFEM_ADNONLININTEG

#include "../config/config.hpp"
#include "fe.hpp"
#include "coefficient.hpp"
#include "fespace.hpp"
#include "nonlininteg.hpp"
#include "../linalg/tadvector.hpp"
#include "../linalg/taddensemat.hpp"
#include "../linalg/fdual.hpp"

#if defined MFEM_USE_ADEPT
#include <adept.h>
#elif defined  MFEM_USE_FADBADPP
#include <fadiff.h>
#include <badiff.h>
#endif

//define Forward AD mode
//#define MFEM_USE_ADFORWARD

namespace mfem
{

// m - dimension of the residual vector
// the Jacobian will have dimensions [m,length(uu)]
template<template <typename, typename> class CTD, int m>
class ADQFunctionTJ{
protected:
#ifdef MFEM_USE_ADEPT
   adept::Stack  m_stack;
#endif

public:
#if defined MFEM_USE_ADEPT
   typedef adept::adouble             ADFType;
   typedef TADVector<ADFType>         ADFVector;
   typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
#elif defined MFEM_USE_FADBADPP
#ifdef MFEM_USE_ADFORWARD
   typedef fadbad::F<double>         ADFType;
   typedef TADVector<ADFType>         ADFVector;
   typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
#else
   typedef fadbad::B<double>         ADFType;
   typedef TADVector<ADFType>         ADFVector;
   typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
#endif
#else
   typedef mfem::ad::FDual<double>    ADFType;
   typedef TADVector<ADFType>         ADFVector;
   typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
#endif

#ifdef MFEM_USE_ADEPT
   ADQFunctionTJ():m_stack(false)  {}
#else
   ADQFunctionTJ(){}
#endif

   ~ADQFunctionTJ(){}

   double QFunction(const mfem::Vector& vparam, mfem::Vector& uu)
   {
        CTD<double,mfem::Vector> func;
        return func(vparam,uu);
   }

   void   QFunctionDU(const mfem::Vector& vparam, ADFVector& uu,
                                                       ADFVector& rr)
   {
        CTD<ADFType,ADFVector> func;
        func(vparam,uu,rr);
   }

   void  QFunctionAU(const Vector &vparam, mfem::Vector &uu,
                                                    mfem::Vector &rr)
   {
       //the result is computed automaticaly by differentiating
       //QFunction with respect to uu
       CTD<ADFType,ADFVector> func;
       int n=uu.Size();
       rr.SetSize(n);

#if defined MFEM_USE_ADEPT
       //use ADEPT package
       adept::Stack* p_stack=adept::active_stack();
       p_stack->deactivate();

       m_stack.activate();
       {
          ADFVector aduu(uu);
          ADFType   rez;
          m_stack.new_recording();
          rez=func(vparam,aduu);
          m_stack.independent(aduu.GetData(), n);//independent variables
          m_stack.dependent(&rez, 1);//dependent variables
          m_stack.jacobian(rr.GetData());
       }
       m_stack.deactivate();
#elif  defined MFEM_USE_FADBADPP
       //use FADBAD++
    #ifdef MFEM_USE_ADFORWARD
           {
              ADFVector aduu(uu);
              ADFType   rez;
              for (int ii=0; ii<n; ii++)
              {
                 aduu[ii].diff(ii,n);
              }
              rez=func(vparam,aduu);
              for (int ii=0; ii<n; ii++)
              {
                  rr[ii]=rez.d(ii);
              }
           }
    #else
           {
              ADFVector aduu(uu);
              ADFType   rez;
              rez=func(vparam,aduu);
              rez.diff(0,1);
              for (int ii=0; ii<n; ii++)
              {
                  rr[ii]=aduu[ii].d(0);
              }
           }
    #endif
#else
       //use native AD package
       {
          ADFVector aduu(uu); //all dual numbers are initialized to zero
          ADFType rez;

          for (int ii=0; ii<n; ii++)
          {
             aduu[ii].dual(1.0);
             rez=func(vparam,aduu);
             rr[ii]=rez.dual();
             aduu[ii].dual(0.0);
          }
       }
#endif

   }

   void QFunctionDU(const mfem::Vector& vparam, mfem::Vector& uu,
                                                    mfem::Vector& rr)
   {
       CTD<double,mfem::Vector> func;
       func(vparam,uu,rr);
   }

   void QFunctionDD(const mfem::Vector& vparam, mfem::Vector& uu,
                                                mfem::DenseMatrix& jac)
   {
#if defined MFEM_USE_ADEPT
       //use ADEPT package
       adept::Stack* p_stack=adept::active_stack();
       p_stack->deactivate();

       int n=uu.Size();
       jac.SetSize(m,n);
       jac=0.0;
       m_stack.activate();
       {
          ADFVector aduu(uu);
          ADFVector rr(m); //residual vector
          m_stack.new_recording();
          QFunctionDU(vparam,aduu,rr);
          m_stack.independent(aduu.GetData(), n);//independent variables
          m_stack.dependent(rr.GetData(), m);//dependent variables
          m_stack.jacobian(jac.Data());
       }
       m_stack.deactivate();
#elif defined MFEM_USE_FADBADPP
   //use FADBAD++
#ifdef MFEM_USE_ADFORWARD
       int n=uu.Size();
       jac.SetSize(m,n);
       jac=0.0;
       {
          ADFVector aduu(uu);
          ADFVector rr(m);

          for (int ii=0; ii<n; ii++)
          {
             aduu[ii].diff(ii,n);
          }
          QFunctionDU(vparam,aduu,rr);
          for (int ii=0; ii<n; ii++)
          {
             for (int jj=0; jj<m; jj++)
             {
                jac(jj,ii)=rr[jj].d(ii);
             }
          }
       }
#else
       int n=uu.Size();
       jac.SetSize(m,n);
       jac=0.0;
       {
          ADFVector aduu(uu);
          ADFVector rr(m);
          QFunctionDU(vparam,aduu,rr);
          for (int ii=0; ii<m; ii++)
          {
             rr[ii].diff(ii,m);
          }
          for (int ii=0; ii<n; ii++)
          {
             for (int jj=0; jj<m; jj++)
             {
                jac(jj,ii)=aduu[ii].d(jj);
             }
          }

       }
#endif
#else
       //use native AD package
       int n=uu.Size();
       jac.SetSize(m,n);
       jac=0.0;
       {
          ADFVector aduu(uu); //all dual numbers are initialized to zero
          ADFVector rr(m);

          for (int ii=0; ii<n; ii++)
          {
             aduu[ii].dual(1.0);
             QFunctionDU(vparam,aduu,rr);
             for (int jj=0; jj<m; jj++)
             {
                jac(jj,ii)=rr[jj].dual();
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
template<template <typename, typename> class CTD>
class ADQFunctionTH{
public:
#if defined MFEM_USE_FADBADPP
   typedef fadbad::B<double>             ADFType;
   typedef TADVector<ADFType>            ADFVector;
   typedef TADDenseMatrix<ADFType>       ADFDenseMatrix;

   typedef fadbad::B<fadbad::F<double>>    ADSType;
   typedef TADVector<ADSType>              ADSVector;
   typedef TADDenseMatrix<ADSType>         ADSDenseMatrix;
#else
   typedef mfem::ad::FDual<double>    ADFType;
   typedef TADVector<ADFType>         ADFVector;
   typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;

   typedef mfem::ad::FDual<ADFType>   ADSType;
   typedef TADVector<ADSType>         ADSVector;
   typedef TADDenseMatrix<ADSType>    ADSDenseMatrix;
#endif

    ADQFunctionTH() {}

    ~ADQFunctionTH() {}

    double QFunction(const mfem::Vector& vparam, const mfem::Vector& uu){
        CTD<double,const mfem::Vector> tf;
        return tf(vparam, uu);
    }

    ADFType QFunction(const mfem::Vector& vparam, ADFVector& uu){
        CTD<ADFType,ADFVector> tf;
        return tf(vparam, uu);
    }

    ADSType QFunction(const mfem::Vector &vparam, ADSVector& uu){
        CTD<ADSType,ADSVector> tf;
        return tf(vparam, uu);
    }

    void QFunctionDU(const mfem::Vector& vparam, mfem::Vector& uu,
                             mfem::Vector& rr){
#if defined MFEM_USE_FADBADPP
        int n=uu.Size();
        rr.SetSize(n);
        ADFVector aduu(uu);
        ADFType   rez;
        rez=QFunction(vparam,aduu);
        rez.diff(0,1);
        for (int ii=0; ii<n; ii++)
        {
            rr[ii]=aduu[ii].d(0);
        }
#else
        int n=uu.Size();
        rr.SetSize(n);
        ADFVector aduu(uu);
        ADFType   rez;
        for (int ii=0; ii<n; ii++)
        {
            aduu[ii].dual(1.0);
            rez=QFunction(vparam,aduu);
            rr[ii]=rez.dual();
            aduu[ii].dual(0.0);
        }
#endif
    }

    void QFunctionDD(const mfem::Vector& vparam, const mfem::Vector& uu,
                     mfem::DenseMatrix& jac){
#if defined MFEM_USE_FADBADPP
       int n=uu.Size();
       jac.SetSize(n);
       jac=0.0;
       {
        ADSVector aduu(n);
        for (int ii = 0;  ii < n ; ii++)
        {
             aduu[ii]=uu[ii];
             aduu[ii].x().diff(ii,n);
        }
        ADSType rez=QFunction(vparam,aduu);
        rez.diff(0,1);
        for (int ii = 0; ii < n ; ii++)
        {
          for (int jj=0; jj<ii; jj++)
          {
            jac(ii,jj)=aduu[ii].d(0).d(jj);
            jac(jj,ii)=aduu[jj].d(0).d(ii);
          }
          jac(ii,ii)=aduu[ii].d(0).d(ii);
        }
       }
#else
       int n=uu.Size();
       jac.SetSize(n);
       jac=0.0;
       {
          ADSVector aduu(n);
          for (int ii = 0;  ii < n ; ii++)
          {
             aduu[ii].real(ADFType(uu[ii],0.0));
             aduu[ii].dual(ADFType(0.0,0.0));
          }

          for (int ii = 0; ii < n ; ii++)
          {
             aduu[ii].real(ADFType(uu[ii],1.0));
             for (int jj=0; jj<(ii+1); jj++)
             {
                aduu[jj].dual(ADFType(1.0,0.0));
                ADSType rez=QFunction(vparam,aduu);
                jac(ii,jj)=rez.dual().dual();
                jac(jj,ii)=rez.dual().dual();
                aduu[jj].dual(ADFType(0.0,0.0));
             }
             aduu[ii].real(ADFType(uu[ii],0.0));
          }
       }
#endif
    }


};// end template ADFunctionTH



}

#endif

