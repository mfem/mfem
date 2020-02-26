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

namespace mfem
{


class FADNonlinearFormIntegratorH: public NonlinearFormIntegrator
{
public:
    
    typedef mfem::ad::FDual<double>    FADType;
    typedef TADVector<FADType>         FADVector;
    typedef TADDenseMatrix<FADType>    FADenseMatrix;
    
    typedef mfem::ad::FDual<FADType>   SADType;
    typedef TADVector<SADType>         SADVector;
    typedef TADDenseMatrix<SADType>    SADDenseMatrix;
    
    FADNonlinearFormIntegratorH(){}
    
    virtual ~FADNonlinearFormIntegratorH(){}
    
    virtual SADType ElementEnergy(const mfem::FiniteElement & el, 
                                 mfem::ElementTransformation & Tr,
                                 const SADVector & elfun)=0;  
                                 
    virtual FADType ElementEnergy(const mfem::FiniteElement & el, 
                                 mfem::ElementTransformation & Tr,
                                 const FADVector & elfun)=0;
                                
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

