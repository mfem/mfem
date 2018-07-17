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

#ifndef MFEM_BACKENDS_PA_INTEGRATOR_HPP
#define MFEM_BACKENDS_PA_INTEGRATOR_HPP

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "vector.hpp"

namespace mfem
{

namespace pa
{

class TensorBilinearFormIntegrator
{
public:
   virtual ~TensorBilinearFormIntegrator() { }

   virtual void ReassembleOperator() = 0;

   virtual void ComputeElementMatrices(DenseTensor &element_matrices)
   { mfem_error("TensorBilinaerFormIntegrator::ComputeElementMatrices is not overloaded"); }

   virtual void MultAdd(const Vector<double>& x, Vector<double>& y) const = 0;

   virtual void Mult(const Vector<double>& x, Vector<double>& y) const
   { y.Fill<double>(0.0); MultAdd(x, y); }
};


} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_INTEGRATOR_HPP
