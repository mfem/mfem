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

#ifndef MFEM_BACKENDS_KERNELS_BILINEAR_FORM_HPP
#define MFEM_BACKENDS_KERNELS_BILINEAR_FORM_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

/// TODO: doxygen
class BilinearForm : public mfem::PBilinearForm
{
protected:
   kBilinearForm *kbform;
public:
   /// TODO: doxygen
   BilinearForm(const Engine &e,
                mfem::BilinearForm &bf)
      : mfem::PBilinearForm(e, bf),
        kbform(NULL) { }

   /// Virtual destructor
   virtual ~BilinearForm() { }

   /// Assemble the PBilinearForm.
   /** This method is called from the method mfem::BilinearForm::Assemble() of
       the associated mfem::BilinearForm, #bform.
       @returns True, if the host assembly should NOT be performed. */
   virtual bool Assemble();

   virtual void FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                 mfem::OperatorHandle &A);

   virtual void FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                 mfem::Vector &x, mfem::Vector &b,
                                 mfem::OperatorHandle &A,
                                 mfem::Vector &X, mfem::Vector &B,
                                 int copy_interior);

   virtual void RecoverFEMSolution(const mfem::Vector &X, const mfem::Vector &b,
                                   mfem::Vector &x);

protected:
   // Called from Assemble() if kbform is NULL to initialize kbform.
   void InitKBilinearForm();
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_BILINEAR_FORM_HPP
