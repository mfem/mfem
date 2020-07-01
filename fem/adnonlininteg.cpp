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

#include "fem.hpp"
#include "../general/forall.hpp"
#include "adnonlininteg.hpp"

namespace mfem
{


void ADQFunctionJ::QFunctionDD(const Vector &vparam, const Vector &uu, DenseMatrix &jac)
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
        this->QFunctionDU(vparam,aduu,rr);
        m_stack.independent(aduu.GetData(), n);//independent variables
        m_stack.dependent(rr.GetData(), m);//dependent variables
        m_stack.jacobian(jac.Data());
    }
    m_stack.deactivate();
#elif defined MFEM_USE_CODIPACK
#if defined MFEM_USE_ADFORWARD
    //use CoDipack
    int n=uu.Size();
    jac.SetSize(m,n);
    jac=0.0;
    {
        ADFVector aduu(n);
        ADFVector rr(m);
        for(int i=0;i<n;i++)
        {
            aduu[i]=uu[i];
            aduu[i].setGradient(0.0);
        }

        for(int ii=0;ii<n;ii++){
            aduu[ii].setGradient(1.0);
            this->QFunctionDU(vparam,aduu,rr);
            for(int jj=0;jj<m;jj++)
            {
                jac(jj,ii)=rr[jj].getGradient();
            }
            aduu[ii].setGradient(0.0);
        }

    }
#else
    //use CoDiPack in reverse mode
    int n=uu.Size();
    jac.SetSize(m,n);
    jac=0.0;
    {
        ADFVector aduu(n);
        ADFVector rr(m);
        for(int i=0;i<n;i++)
        {
            aduu[i]=uu[i];
        }

        ADFType::TapeType& tape= ADFType::getGlobalTape();
        typename ADFType::TapeType::Position pos=tape.getPosition();

        tape.setActive();
        for(int ii=0;ii<n;ii++){ tape.registerInput(aduu[ii]); }
        this->QFunctionDU(vparam,aduu,rr);
        for(int ii=0;ii<m;ii++){ tape.registerOutput(rr[ii]); }
        tape.setPassive();

        for(int jj=0;jj<m;jj++){
            rr[jj].setGradient(1.0);
            tape.evaluate();
            for(int ii=0;ii<n;ii++){
                jac(jj,ii)=aduu[ii].getGradient();
            }
            rr[jj].setGradient(0.0);
        }
        tape.reset(pos);
    }
#endif
#else
    //use native AD package
   int n=uu.Size();
   jac.SetSize(m,n);
   jac=0.0;
   {
       ADFVector aduu(uu); //all dual numbers are initialized to zero
       ADFVector rr;

       for(int ii=0;ii<n;ii++){
           aduu[ii].dual(1.0);
           this->QFunctionDU(vparam,aduu,rr);
           for(int jj=0;jj<m;jj++)
           {
               jac(jj,ii)=rr[jj].dual();
           }
           aduu[ii].dual(0.0);
       }
   }
#endif
}

void ADQFunctionH::QFunctionDU(const Vector &vparam, Vector &uu, Vector &rr)
{

#if defined MFEM_USE_CODIPACK
    int n=uu.Size();
    rr.SetSize(n);
    ADFVector aduu(n);
    ADFType   rez;
    for(int ii=0;ii<n;ii++)
    {
        aduu[ii].setValue(uu[ii]);
        aduu[ii].setGradient(0.0);
    }
    for(int ii=0;ii<n;ii++)
    {
        aduu[ii].setGradient(1.0);
        rez=this->QFunction(vparam,aduu);
        rr[ii]=rez.getGradient();
        aduu[ii].setGradient(0.0);
    }
#else
    int n=uu.Size();
    rr.SetSize(n);
    ADFVector aduu(uu);
    ADFType   rez;
    for(int ii=0;ii<n;ii++)
    {
        aduu[ii].dual(1.0);
        rez=this->QFunction(vparam,aduu);
        rr[ii]=rez.dual();
        aduu[ii].dual(0.0);
    }
#endif
}

void ADQFunctionH::QFunctionDD(const Vector &vparam, const Vector &uu, DenseMatrix &jac)
{

#if defined MFEM_USE_CODIPACK
#if defined MFEM_USE_ADFORWARD
    int n=uu.Size();
    jac.SetSize(n);
    jac=0.0;
    {
        ADSVector aduu(n);
        for(int ii = 0;  ii < n ; ii++)
        {
            aduu[ii].value().value()=uu[ii];
            aduu[ii].value().gradient()=0.0;
            aduu[ii].gradient().value()=0.0;
            aduu[ii].gradient().gradient()=0.0;
        }

        for(int ii = 0; ii < n ; ii++)
        {
            aduu[ii].value().gradient()=1.0;
            for(int jj=0; jj<(ii+1); jj++)
            {
                aduu[ii].gradient().value()=1.0;
                ADSType rez= this->QFunction(vparam,aduu);
                jac(ii,jj)=rez.gradient().gradient();
                jac(jj,ii)=jac(ii,jj);
                aduu[jj].gradient().value()=0.0;
            }
            aduu[ii].value().gradient()=0.0;
        }
    }
#else
    int n=uu.Size();
    jac.SetSize(n);
    jac=0.0;
    {
        ADSVector aduu(n);
        for(int ii=0;ii < n ; ii++)
        {
            aduu[ii].value().value()=uu[ii];
        }

        ADSType rez;
        ADSType::TapeType& tape = ADSType::getGlobalTape();
        typename ADSType::TapeType::Position pos;

        for(int ii = 0; ii < n ; ii++)
        {
            pos=tape.getPosition();
            tape.setActive();

            for(int jj=0;jj < n; jj++) {
                if(jj==ii) {aduu[jj].value().gradient()=1.0;}
                else {aduu[jj].value().gradient()=0.0;}
                tape.registerInput(aduu[jj]);
            }

            rez=this->QFunction(vparam,aduu);
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
#else
    int n=uu.Size();
    jac.SetSize(n);
    jac=0.0;
    {
        ADSVector aduu(n);
        for(int ii = 0;  ii < n ; ii++)
        {
            aduu[ii].real(ADFType(uu[ii],0.0));
            aduu[ii].dual(ADFType(0.0,0.0));
        }

        for(int ii = 0; ii < n ; ii++)
        {
            aduu[ii].real(ADFType(uu[ii],1.0));
            for(int jj=0; jj<(ii+1); jj++)
            {
                aduu[jj].dual(ADFType(1.0,0.0));
                ADSType rez= this->QFunction(vparam,aduu);
                jac(ii,jj)=rez.dual().dual();
                jac(jj,ii)=rez.dual().dual();
                aduu[jj].dual(ADFType(0.0,0.0));
            }
            aduu[ii].real(ADFType(uu[ii],0.0));
        }
    }
#endif
}





double ADNonlinearFormIntegratorH::GetElementEnergy(const mfem::FiniteElement & el,
                                    mfem::ElementTransformation & Tr, 
                                    const mfem::Vector & elfun)
{
    return this->ElementEnergy(el,Tr,elfun);
}

void ADNonlinearFormIntegratorH::AssembleElementVector(const mfem::FiniteElement & el,
                                            mfem::ElementTransformation & Tr, 
                                            const mfem::Vector & elfun, mfem::Vector & elvect)
{
    
    int ndof = el.GetDof();
    elvect.SetSize(ndof);
    
    {
        ADFVector adelfun(elfun);
        //all dual numbers in adelfun are initialized to 0.0
        for(int ii = 0; ii < adelfun.Size(); ii++)
        {
            //set the dual for the ii^th element to 1.0
            adelfun[ii].dual(1.0);
            ADFType rez= this->ElementEnergy(el,Tr, adelfun);
            elvect[ii]=rez.dual();
            //return it back to zero
            adelfun[ii].dual(0.0);
        }
    }
    
}

void ADNonlinearFormIntegratorH::AssembleElementGrad(const mfem::FiniteElement & el,
                                          mfem::ElementTransformation & Tr, 
                                          const mfem::Vector & elfun, 
                                          mfem::DenseMatrix & elmat)
{
    
    int ndof = el.GetDof();
    elmat.SetSize(ndof);
    elmat=0.0;
    {
        ADSVector adelfun(ndof);
        for(int ii = 0; ii < ndof; ii++)
        {
            adelfun[ii].real(ADFType(elfun[ii],0.0));
            adelfun[ii].dual(ADFType(0.0,0.0));
        }
        
        for(int ii = 0; ii < adelfun.Size(); ii++)
        {
            adelfun[ii].real(ADFType(elfun[ii],1.0));
            for(int jj = 0; jj < (ii+1); jj++)
            {
                adelfun[jj].dual(ADFType(1.0,0.0));
                ADSType rez= this->ElementEnergy(el,Tr, adelfun);
                elmat(ii,jj)=rez.dual().dual();
                elmat(jj,ii)=rez.dual().dual();
                adelfun[jj].dual(ADFType(0.0,0.0));
            }
            adelfun[ii].real(ADFType(elfun[ii],0.0));
        }
        
    }
    
}

    
} //end namespace mfem

