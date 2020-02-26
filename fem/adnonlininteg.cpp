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
    

double FADNonlinearFormIntegratorH::GetElementEnergy(const mfem::FiniteElement & el, 
                                    mfem::ElementTransformation & Tr, 
                                    const mfem::Vector & elfun)
{
    return this->ElementEnergy(el,Tr,elfun);
}

void FADNonlinearFormIntegratorH::AssembleElementVector(const mfem::FiniteElement & el,
                                            mfem::ElementTransformation & Tr, 
                                            const mfem::Vector & elfun, mfem::Vector & elvect)
{
    
    int ndof = el.GetDof();
    elvect.SetSize(ndof);
    
    {
        FADVector adelfun(elfun);
        //all dual numbers in adelfun are initialized to 0.0
        for(int ii = 0; ii < adelfun.Size(); ii++)
        {
            //set the dual for the ii^th element to 1.0
            adelfun[ii].dual(1.0);
            FADType rez= this->ElementEnergy(el,Tr, adelfun);
            elvect[ii]=rez.dual();
            //return it back to zero
            adelfun[ii].dual(0.0);
        }
    }
    
}

void FADNonlinearFormIntegratorH::AssembleElementGrad(const mfem::FiniteElement & el, 
                                          mfem::ElementTransformation & Tr, 
                                          const mfem::Vector & elfun, 
                                          mfem::DenseMatrix & elmat)
{
    
    int ndof = el.GetDof();
    elmat.SetSize(ndof);
    elmat=0.0;
    {
        SADVector adelfun(ndof);
        for(int ii = 0; ii < ndof; ii++)
        {
            adelfun[ii].real(FADType(elfun[ii],0.0));
            adelfun[ii].dual(FADType(0.0,0.0));
        }
        
        for(int ii = 0; ii < adelfun.Size(); ii++)
        {
            adelfun[ii].real(FADType(elfun[ii],1.0));
            for(int jj = 0; jj < (ii+1); jj++)
            {
                adelfun[jj].dual(FADType(1.0,0.0));
                SADType rez= this->ElementEnergy(el,Tr, adelfun);
                elmat(ii,jj)=rez.dual().dual();
                elmat(jj,ii)=rez.dual().dual();
                adelfun[jj].dual(FADType(0.0,0.0));
            }
            adelfun[ii].real(FADType(elfun[ii],0.0));    
        }
        
    }
    
}

    
} //end namespace mfem

