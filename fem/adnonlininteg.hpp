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

#ifdef MFEM_USE_ADEPT
#include <adept.h>
#elif  MFEM_USE_CODIPACK
#include <codi.hpp>
#endif

namespace mfem
{

class ADQIntegratorJ
{
  private:
    int m;  //dimension of the residual vector
            //the Jacobian will have dimensions [m,length(uu)]
  protected:protected:
#ifdef MFEM_USE_ADEPT
    adept::Stack  m_stack;
#endif

  public:
#ifdef MFEM_USE_ADEPT
    typedef adept::adouble 			   ADFType;
    typedef TADVector<ADFType>         ADFVector;
    typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
#elif  MFEM_USE_CODIPACK
    typedef codi::RealForward 		   ADFType;
    typedef TADVector<ADFType>         ADFVector;
    typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
#else
    typedef mfem::ad::FDual<double>    ADFType;
    typedef TADVector<ADFType>         ADFVector;
    typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
#endif

#ifdef MFEM_USE_ADEPT
    ADQIntegratorJ(int m_=1):m_stack(false)
    {
        m=m_;
    }
#else
    ADQIntegratorJ(int m_=1){ m=m_;}
#endif

    virtual ~ADQIntegratorJ(){}

    virtual double QIntegrator(const mfem::Vector& vparam, const mfem::Vector& uu)=0;
    virtual void QIntegratorDU(const mfem::Vector& vparam, ADFVector& uu, ADFVector& rr)=0;
    virtual void QIntegratorDU(const mfem::Vector& vparam, mfem::Vector& uu, mfem::Vector& rr)=0;

    void QIntegratorDD(const mfem::Vector& vparam, const mfem::Vector& uu, mfem::DenseMatrix& jj);

};


class ADQIntegratorH
{
  public:
    typedef mfem::ad::FDual<double>    ADFType;
    typedef TADVector<ADFType>         ADFVector;
    typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;

    typedef mfem::ad::FDual<ADFType>   ADSType;
    typedef TADVector<ADSType>         ADSVector;
    typedef TADDenseMatrix<ADSType>    ADSDenseMatrix;

    ADQIntegratorH(){}

    virtual ~ADQIntegratorH(){}

    virtual double QIntegrator(const mfem::Vector& vparam, const mfem::Vector& uu)=0;
    virtual ADFType QIntegrator(const mfem::Vector& vparam, ADFVector& uu)=0;
    virtual ADSType QIntegrator(const mfem::Vector &vparam, ADSVector& uu)=0;

    virtual void QIntegratorDU(const mfem::Vector& vparam, mfem::Vector& uu, mfem::Vector& rr);
    void QIntegratorDD(const mfem::Vector& vparam, const mfem::Vector& uu, mfem::DenseMatrix& jj);
};



class ADNonlinearFormIntegratorH: public NonlinearFormIntegrator
{
public:
    
    typedef mfem::ad::FDual<double>    ADFType;
    typedef TADVector<ADFType>         ADFVector;
    typedef TADDenseMatrix<ADFType>    ADFDenseMatrix;
    
    typedef mfem::ad::FDual<ADFType>   ADSType;
    typedef TADVector<ADSType>         ADSVector;
    typedef TADDenseMatrix<ADSType>    ADSDenseMatrix;
    
    ADNonlinearFormIntegratorH(){}
    
    virtual ~ADNonlinearFormIntegratorH(){}
    
    virtual ADSType ElementEnergy(const mfem::FiniteElement & el,
                                 mfem::ElementTransformation & Tr,
                                 const ADSVector & elfun)=0;
                                 
    virtual ADFType ElementEnergy(const mfem::FiniteElement & el,
                                 mfem::ElementTransformation & Tr,
                                 const ADFVector & elfun)=0;
                                
    virtual double ElementEnergy(const mfem::FiniteElement & el, 
                                 mfem::ElementTransformation & Tr,
                                 const mfem::Vector & elfun)=0;
                                    
                                    
    virtual double GetElementEnergy(const mfem::FiniteElement & el, 
                                    mfem::ElementTransformation & Tr, 
                                    const mfem::Vector & elfun) override;
    
    virtual void AssembleElementVector(const mfem::FiniteElement & el,
                                            mfem::ElementTransformation & Tr, 
                                            const mfem::Vector & elfun, mfem::Vector & elvect) override;
    
    virtual void AssembleElementGrad(const mfem::FiniteElement & el, 
                                          mfem::ElementTransformation & Tr, 
                                          const mfem::Vector & elfun, 
                                          mfem::DenseMatrix & elmat) override;
};


}

#endif    

