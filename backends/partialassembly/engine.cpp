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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "backend.hpp"
#include "bilinearform.hpp"
#include "../../general/array.hpp"

namespace mfem
{

namespace pa
{

template <Location Device>
void PAEngine<Device>::Init(const std::string &engine_spec)
{
   //
   // Initialize inherited fields
   //
   memory_resources[0] = NULL;
   workers_weights[0] = 1.0;
   workers_mem_res[0] = 0;
}

template <Location Device>
PAEngine<Device>::PAEngine()
   : mfem::Engine(NULL, 1, 1)
{
   Init("");
}

template <Location Device>
PAEngine<Device>::PAEngine(const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1)
{
   Init(engine_spec);
}

#ifdef MFEM_USE_MPI
template <Location Device>
PAEngine<Device>::PAEngine(MPI_Comm _comm, const std::string &engine_spec)
   : mfem::Engine(NULL, 1, 1)
{
   comm = _comm;
   Init(engine_spec);
}
#endif

template <Location Device>
DLayout PAEngine<Device>::MakeLayout(std::size_t size) const
{
   return DLayout(new LayoutType<Device>(*this, size));
}

template <Location Device>
DLayout PAEngine<Device>::MakeLayout(const mfem::Array<std::size_t> &offsets) const
{
   MFEM_ASSERT(offsets.Size() == 2,
               "multiple workers are not supported yet");
   return DLayout(new LayoutType<Device>(*this, offsets.Last()));
}

template <Location Device>
DArray PAEngine<Device>::MakeArray(PLayout &layout, std::size_t item_size) const
{
   MFEM_ASSERT(dynamic_cast<LayoutType<Device> *>(&layout) != NULL,
               "invalid input layout");
   LayoutType<Device> *lt = static_cast<LayoutType<Device> *>(&layout);
   return DArray(new ArrayType<Device>(*lt, item_size));
}

template <Location Device>
DVector PAEngine<Device>::MakeVector(PLayout &layout, int type_id) const
{
   MFEM_ASSERT(dynamic_cast<LayoutType<Device> *>(&layout) != NULL,
               "invalid input layout");
   LayoutType<Device> *lt = static_cast<LayoutType<Device> *>(&layout);
   switch (type_id)
   {
   case ScalarId<double>::value:
      return DVector(new VectorType<Device,double>(*lt));
   // case ScalarId<std::complex<double>>::value:
   //    return DVector(new VectorType<Device,std::complex<double>>(*lt));
   // case ScalarId<int>::value:
   //    return DVector(new Vector<int>(*lt));
   default:
      mfem_error("Invalid type_id");
   }
}

template <Location Device>
DFiniteElementSpace PAEngine<Device>::MakeFESpace(mfem::FiniteElementSpace &fespace) const
{
   return DFiniteElementSpace(new PAFiniteElementSpace<Device>(*this, fespace));
}

template <Location Device>
DBilinearForm PAEngine<Device>::MakeBilinearForm(mfem::BilinearForm &bf) const
{
   return DBilinearForm(new mfem::pa::BilinearForm<Device>(*this, bf));
}

template <Location Device>
void PAEngine<Device>::AssembleLinearForm(LinearForm &l_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
}

template <Location Device>
mfem::Operator *PAEngine<Device>::MakeOperator(const MixedBilinearForm &mbl_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
   return NULL;
}

template <Location Device>
mfem::Operator *PAEngine<Device>::MakeOperator(const NonlinearForm &nl_form) const
{
   /// FIXME - What will the actual parameters be?
   MFEM_ABORT("FIXME");
   return NULL;
}

template class PAEngine<Host>;
#ifdef __NVCC__
template class PAEngine<CudaDevice>;
#endif

mfem::Engine* createEngine(const std::string& engine_spec){
   Engine* engine = NULL;
   if (engine_spec=="Host")
   {
      engine = new PAEngine<Host>;
   }
   return engine;
}


} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)
