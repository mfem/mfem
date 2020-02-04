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
#include "tadvectro.hpp"
#include "taddensmat.hpp"

namespace mfem
{

#if define(MFEM_USE_ADEPT)||define(MFEM_USE_CODIPACK)
/** The abstract base class ADNonlinearFormIntegrator is
    a generalization of the NonlinearFormIntegrator class suitable
    for algorithmic differentiation.   
    All derived classes must implement ADAssembleElementVector(...); 
    and ADGetElementEnergy(...); */
class ADNonlinearFormIntegrator: public NonlinearFormIntegrator
{
protected:
#ifdef  MFEM_USE_ADEPT
     
#elseif MFEM_USE_CODIPACK

#endif    
public:
   ADNonlinearFormIntegrator();
   virtual ~ADNonlinearFormIntegrator();
 
   /// Methods called by the AD routines
   virtual void ADGetElementEnergy(const mfem::FiniteElement & el,
                                   mfem::ElementTransformation & Tr,
                                   const mfem::TADVector<adouble> & elfun);
   
   virtual void ADAssembleElementVector(const mfem::FiniteElement & el,
                                    mfem::ElementTransformation & Tr,
                                    const mfem::TADVector<adouble> & elfun, 
                                    mfem::TADVector<adouble> &elvec);
   
   
   
   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect) override;
   
   
   virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                    mfem::ElementTransformation & Tr,
                                    const mfem::Vector & elfun, 
                                    mfem::DenseMatrix & elmat) override;
   
   virtual double GetElementEnergy(const mfem::FiniteElement & el,
                                   mfem::ElementTransformation & Tr,
                                   const mfem::Vector & elfun) override;
   
};
    
    
#endif    

}
