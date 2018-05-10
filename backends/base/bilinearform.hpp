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

#ifndef MFEM_BACKENDS_BASE_BILINEARFORM_HPP
#define MFEM_BACKENDS_BASE_BILINEARFORM_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "engine.hpp"

namespace mfem
{

class Vector;
class OperatorHandle;
class BilinearForm;

/// TODO: doxygen
class PBilinearForm : public RefCounted
{
protected:
   /// Engine with shared ownership
   SharedPtr<const Engine> engine;
   /// Not owned.
   BilinearForm *bform;

public:
   /// TODO: doxygen
   PBilinearForm(const Engine &e, BilinearForm &bf)
      : engine(&e), bform(&bf) { }

   /// Virtual destructor
   virtual ~PBilinearForm() { }

   /// Get the associated Engine
   const Engine &GetEngine() const { return *engine; }

   /// Assemble the PBilinearForm.
   /** This method is called from the method BilinearForm::Assemble() of the
       associated BilinearForm #bform.
       @returns True, if the host assembly should be skipped. */
   virtual bool Assemble() = 0;

   /// TODO: doxygen
   virtual void FormSystemMatrix(const Array<int> &ess_tdof_list,
                                 OperatorHandle &A) = 0;

   /// TODO: doxygen
   virtual void FormLinearSystem(const Array<int> &ess_tdof_list,
                                 Vector &x, Vector &b,
                                 OperatorHandle &A, Vector &X, Vector &B,
                                 int copy_interior) = 0;

   /// TODO: doxygen
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b,
                                   Vector &x) = 0;
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_BILINEARFORM_HPP
