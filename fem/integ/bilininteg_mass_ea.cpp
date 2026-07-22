// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "bilininteg_mass_kernels.hpp"

namespace mfem
{
namespace
{

template <bool UPPER>
void AssembleEATriangularImpl(const int dim, const int ne,
                              const int dofs1D, const int quad1D,
                              const Array<real_t> &B,
                              const Vector &pa_data,
                              Vector &data,
                              const bool add)
{
   using internal::EAMassAssembleTriangular1D;
   using internal::EAMassAssembleTriangular2D;
   using internal::EAMassAssembleTriangular3D;

   if (dim == 1)
   {
      auto kernel = EAMassAssembleTriangular1D<UPPER,0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: kernel = EAMassAssembleTriangular1D<UPPER,2,2>; break;
         case 0x33: kernel = EAMassAssembleTriangular1D<UPPER,3,3>; break;
         case 0x44: kernel = EAMassAssembleTriangular1D<UPPER,4,4>; break;
         case 0x55: kernel = EAMassAssembleTriangular1D<UPPER,5,5>; break;
         case 0x66: kernel = EAMassAssembleTriangular1D<UPPER,6,6>; break;
         case 0x77: kernel = EAMassAssembleTriangular1D<UPPER,7,7>; break;
         case 0x88: kernel = EAMassAssembleTriangular1D<UPPER,8,8>; break;
         case 0x99: kernel = EAMassAssembleTriangular1D<UPPER,9,9>; break;
      }
      return kernel(ne, B, pa_data, data, add, dofs1D, quad1D);
   }
   else if (dim == 2)
   {
      auto kernel = EAMassAssembleTriangular2D<UPPER,0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: kernel = EAMassAssembleTriangular2D<UPPER,2,2>; break;
         case 0x33: kernel = EAMassAssembleTriangular2D<UPPER,3,3>; break;
         case 0x44: kernel = EAMassAssembleTriangular2D<UPPER,4,4>; break;
         case 0x55: kernel = EAMassAssembleTriangular2D<UPPER,5,5>; break;
         case 0x66: kernel = EAMassAssembleTriangular2D<UPPER,6,6>; break;
         case 0x77: kernel = EAMassAssembleTriangular2D<UPPER,7,7>; break;
         case 0x88: kernel = EAMassAssembleTriangular2D<UPPER,8,8>; break;
         case 0x99: kernel = EAMassAssembleTriangular2D<UPPER,9,9>; break;
      }
      return kernel(ne, B, pa_data, data, add, dofs1D, quad1D);
   }
   else if (dim == 3)
   {
      auto kernel = EAMassAssembleTriangular3D<UPPER,0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: kernel = EAMassAssembleTriangular3D<UPPER,2,3>; break;
         case 0x34: kernel = EAMassAssembleTriangular3D<UPPER,3,4>; break;
         case 0x45: kernel = EAMassAssembleTriangular3D<UPPER,4,5>; break;
         case 0x56: kernel = EAMassAssembleTriangular3D<UPPER,5,6>; break;
         case 0x67: kernel = EAMassAssembleTriangular3D<UPPER,6,7>; break;
         case 0x78: kernel = EAMassAssembleTriangular3D<UPPER,7,8>; break;
         case 0x89: kernel = EAMassAssembleTriangular3D<UPPER,8,9>; break;
      }
      return kernel(ne, B, pa_data, data, add, dofs1D, quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

}

void MassIntegrator::AssembleEA_(Vector &ea_data,
                                 const bool add)
{
   using internal::EAMassAssemble1D;
   using internal::EAMassAssemble2D;
   using internal::EAMassAssemble3D;

   const Array<real_t> &B = maps->B;
   if (dim == 1)
   {
      auto kernel = EAMassAssemble1D<0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: kernel = EAMassAssemble1D<2,2>; break;
         case 0x33: kernel = EAMassAssemble1D<3,3>; break;
         case 0x44: kernel = EAMassAssemble1D<4,4>; break;
         case 0x55: kernel = EAMassAssemble1D<5,5>; break;
         case 0x66: kernel = EAMassAssemble1D<6,6>; break;
         case 0x77: kernel = EAMassAssemble1D<7,7>; break;
         case 0x88: kernel = EAMassAssemble1D<8,8>; break;
         case 0x99: kernel = EAMassAssemble1D<9,9>; break;
      }
      return kernel(ne,B,pa_data,ea_data,add,dofs1D,quad1D);
   }
   else if (dim == 2)
   {
      auto kernel = EAMassAssemble2D<0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: kernel = EAMassAssemble2D<2,2>; break;
         case 0x33: kernel = EAMassAssemble2D<3,3>; break;
         case 0x44: kernel = EAMassAssemble2D<4,4>; break;
         case 0x55: kernel = EAMassAssemble2D<5,5>; break;
         case 0x66: kernel = EAMassAssemble2D<6,6>; break;
         case 0x77: kernel = EAMassAssemble2D<7,7>; break;
         case 0x88: kernel = EAMassAssemble2D<8,8>; break;
         case 0x99: kernel = EAMassAssemble2D<9,9>; break;
      }
      return kernel(ne,B,pa_data,ea_data,add,dofs1D,quad1D);
   }
   else if (dim == 3)
   {
      auto kernel = EAMassAssemble3D<0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: kernel = EAMassAssemble3D<2,3>; break;
         case 0x34: kernel = EAMassAssemble3D<3,4>; break;
         case 0x45: kernel = EAMassAssemble3D<4,5>; break;
         case 0x56: kernel = EAMassAssemble3D<5,6>; break;
         case 0x67: kernel = EAMassAssemble3D<6,7>; break;
         case 0x78: kernel = EAMassAssemble3D<7,8>; break;
         case 0x89: kernel = EAMassAssemble3D<8,9>; break;
      }
      return kernel(ne,B,pa_data,ea_data,add,dofs1D,quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

void MassIntegrator::AssembleEATriangular_(
   TriPackMatrix<TriangularPart::LOWER> &ea_data,
                                           const bool add)
{
   Vector &data = ea_data.Data();
   const Array<real_t> &B = maps->B;
   return AssembleEATriangularImpl<false>(dim, ne, dofs1D, quad1D,
                                          B, pa_data, data, add);
}

void MassIntegrator::AssembleEATriangular_(
   TriPackMatrix<TriangularPart::UPPER> &ea_data,
   const bool add)
{
   Vector &data = ea_data.Data();
   const Array<real_t> &B = maps->B;
   return AssembleEATriangularImpl<true>(dim, ne, dofs1D, quad1D,
                                         B, pa_data, data, add);
}

void MassIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                Vector &ea_data,
                                const bool add)
{
   AssemblePA(fes);
   if (ne > 0) { AssembleEA_(ea_data, add); }
}

void MassIntegrator::AssembleEATriangular(const FiniteElementSpace &fes,
                                          TriPackMatrix<TriangularPart::LOWER> &ea_data,
                                          const bool add)
{
   AssemblePA(fes);
   if (ne == 0) { return; }

   int elem_dofs = 1;
   for (int d = 0; d < dim; ++d)
   {
      elem_dofs *= dofs1D;
   }

   if (add)
   {
      MFEM_VERIFY(ea_data.GetNumRows() == elem_dofs,
                  "Invalid triangular EA element size.");
      MFEM_VERIFY(ea_data.GetNumMatrices() == ne,
                  "Invalid triangular EA element count.");
   }
   else
   {
      ea_data.SetSize(elem_dofs, ne);
      ea_data.UseDevice(true);
   }

   AssembleEATriangular_(ea_data, add);
}

void MassIntegrator::AssembleEATriangular(const FiniteElementSpace &fes,
                                          TriPackMatrix<TriangularPart::UPPER> &ea_data,
                                          const bool add)
{
   AssemblePA(fes);
   if (ne == 0) { return; }

   int elem_dofs = 1;
   for (int d = 0; d < dim; ++d)
   {
      elem_dofs *= dofs1D;
   }

   if (add)
   {
      MFEM_VERIFY(ea_data.GetNumRows() == elem_dofs,
                  "Invalid triangular EA element size.");
      MFEM_VERIFY(ea_data.GetNumMatrices() == ne,
                  "Invalid triangular EA element count.");
   }
   else
   {
      ea_data.SetSize(elem_dofs, ne);
      ea_data.UseDevice(true);
   }

   AssembleEATriangular_(ea_data, add);
}

void MassIntegrator::AssembleEABoundary(const FiniteElementSpace &fes,
                                        Vector &ea_data,
                                        const bool add)
{
   AssemblePABoundary(fes);
   if (ne > 0) { AssembleEA_(ea_data, add); }
}

}
