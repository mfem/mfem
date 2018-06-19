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

#ifndef MFEM_BACKENDS_OMP_BILINEARFORM_HPP
#define MFEM_BACKENDS_OMP_BILINEARFORM_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "fespace.hpp"
#include "../../fem/bilininteg.hpp"

namespace mfem
{

namespace omp
{

/// TODO: doxygen
class BilinearForm : public mfem::PBilinearForm
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // mfem::BilinearForm *bform;

   mfem::Array<mfem::TensorBilinearFormIntegrator*> tbfi;
   bool has_assembled;

   DFiniteElementSpace trial_fes, test_fes;

   void TransferIntegrators();

public:
   /// TODO: doxygen
   BilinearForm(const Engine &e, mfem::BilinearForm &bf)
      : mfem::PBilinearForm(e, bf),
	tbfi(),
	has_assembled(false),
	trial_fes(bf.FESpace()->Get_PFESpace()),
	test_fes(bf.FESpace()->Get_PFESpace()) { }

   /// Return the engine as an OpenMP engine
   const Engine &OmpEngine() { return static_cast<const Engine&>(*engine); }

   /// Virtual destructor
   virtual ~BilinearForm() { }

   /// Assemble the PBilinearForm.
   /** This method is called from the method BilinearForm::Assemble() of the
       associated BilinearForm #bform.
       @returns True, if the host assembly should be skipped. */
   virtual bool Assemble();

   /// TODO: doxygen
   virtual void FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                 OperatorHandle &A);

   /// TODO: doxygen
   virtual void FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                 mfem::Vector &x, mfem::Vector &b,
                                 OperatorHandle &A, mfem::Vector &X, mfem::Vector &B,
                                 int copy_interior);

   /// TODO: doxygen
   virtual void RecoverFEMSolution(const mfem::Vector &X, const mfem::Vector &b,
                                   mfem::Vector &x);
};

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_BILINEAR_FORM_HPP
