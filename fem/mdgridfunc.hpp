// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MDGRIDFUNC
#define MFEM_MDGRIDFUNC

#include "../config/config.hpp"

#include "fem/gridfunc.hpp"
#include "general/mdspan.hpp"

namespace mfem
{

template<int N, class Layout = MDLayoutLeft<N>>
class MDGridFunction : public MDSpan<GridFunction, N, Layout>
{
   using base_t = MDSpan<GridFunction, N, Layout>;
   using base_t::Nd;
   using base_t::Sd;
   using GridFunction::data;

public:

   /**
    * @brief MDGridFunction default constructor (recursion)
    */
   MDGridFunction(): base_t() { }

   /**
    * @brief MDGridFunction recursion constructor
    * @param[in] fes Finite element space to use
    * @param[in] args Rest of dimension indices
    */
   template <typename... Ts>
   MDGridFunction(FiniteElementSpace *fes, Ts... args): MDGridFunction(args...)
   {
      SetSpace(fes);
      MFEM_VERIFY(fes->GetVDim() == 1,
                  "Only FiniteElementSpace with vdim of 1 are supported");
      base_t::Setup(fes->GetNDofs(), args...);
   }

   /**
    * @brief MDGridFunction recursion constructor
    * @param[in] dim Dimension indice
    * @param[in] args Rest of dimension indices or finite element space to use
    */
   template <typename... Ts>
   MDGridFunction(int dim, Ts... args): MDGridFunction(args...)
   {
      base_t::Setup(dim, args...);
   }

   /// Move constructor not supported
   MDGridFunction(MDGridFunction&&) = delete;

   /// Copy constructor not supported
   MDGridFunction(const MDGridFunction&) = delete;

   /// Move assignment not supported
   MDGridFunction& operator=(MDGridFunction&&) = delete;

   /// Copy assignment not supported
   MDGridFunction& operator=(const MDGridFunction&) = delete;

   /**
    * @brief Returns the specific GridFunction from dimension indices
    * @param[out] gf Returned GridFunction
    * @param[in] args Rest of dimension indices
    */
   template <int n = 1, typename... Ts>
   void GetScalarGridFunction(GridFunction &gf, Ts... args) const
   {
      FiniteElementSpace *fes = GridFunction::fes;
      MFEM_VERIFY(fes->GetNDofs() == Nd[n-1], "Error in dofs size!");
      gf.SetSpace(fes);
      for (int s = 0; s < Nd[n-1]; s++)
      {
         gf[s] = data[get_vdofs_offset +
                      MDOffset<n,N,int,Ts...>::offset(Sd, s, args...)];
      }
      get_vdofs_offset = 0; // re-init for next calls
   }

   /**
    * @brief Returns the specific GridFunction from dimension indices
    * @param[in] dim Dimension indice
    * @param args Rest of dimension indices or GridFunction to be returned
    */
   template <int n = 1, typename... Ts>
   void GetScalarGridFunction(int dim, Ts&&... args) const
   {
      get_vdofs_offset += dim * Sd[n-1];
      MDGridFunction::GetScalarGridFunction<n+1>(std::forward<Ts>(args)...);
   }

   /**
    * @brief Sets the given GridFunction at the specific dimension indices
    * @param[in] gf GridFunction to set
    * @param[in] args Rest of dimension indices
    */
   template <int n = 1, typename... Ts>
   void SetScalarGridFunction(const GridFunction &gf, Ts... args)
   {
      MFEM_VERIFY(GridFunction::fes->GetNDofs() == Nd[n-1], "Error in dofs size!");
      for (int s = 0; s < Nd[n-1]; s++)
      {
         data[get_vdofs_offset +
              MDOffset<n,N,int,Ts...>::offset(Sd, s, args...)] = gf[s];
      }
      get_vdofs_offset = 0; // re-init for next calls
   }

   /**
    * @brief Sets the given GridFunction at the specific dimension indices
    * @param[in] dim Dimension indice
    * @param args Rest of dimension indices or given GridFunction to be used
    */
   template <int n = 1, typename... Ts>
   void SetScalarGridFunction(int dim, Ts... args)
   {
      get_vdofs_offset += dim * Sd[n-1];
      MDGridFunction::SetScalarGridFunction<n+1>(args...);
   }

   using GridFunction::Read;
   using GridFunction::Write;
   using GridFunction::ReadWrite;
   using GridFunction::HostRead;
   using GridFunction::HostWrite;
   using GridFunction::HostReadWrite;

   using GridFunction::GetData;
   using GridFunction::SetData;
   using GridFunction::SetSpace;

   using Vector::operator=;

private:
   mutable int get_vdofs_offset = 0;
};

} // namespace mfem

#endif // MFEM_MDGRIDFUNC
