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

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_TENSOR
#define MFEM_COLD_PLASMA_DIELECTRIC_TENSOR

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

class DielectricTensor: public MatrixCoefficient
{
public:
   DielectricTensor(ParGridFunction & B,
                    BlockVector & T,
                    BlockVector & density,
                    ParFiniteElementSpace & H1FESpace,
                    ParFiniteElementSpace & L2FESpace,
                    int nspecies,
                    double omega,
                    bool realPart = true);
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
   // virtual void Dval(DenseMatrix &K, ElementTransformation &T,
   //                   const IntegrationPoint &ip);
   virtual ~DielectricTensor() {}

private:
   ParGridFunction * B_;
   BlockVector * temperature_;
   BlockVector * density_;
   ParFiniteElementSpace * H1FESpace_;
   ParFiniteElementSpace * L2FESpace_;
   int nspecies_;
   double omega_;
   bool realPart_;

   ParGridFunction density_gf_;
   ParGridFunction temperature_gf_;

   Vector density_vals_;
   Vector temperature_vals_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_TENSOR
