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

#ifndef MFEM_BACKENDS_OMP_ENGINE_HPP
#define MFEM_BACKENDS_OMP_ENGINE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "../base/engine.hpp"

namespace mfem
{

namespace omp
{

enum ExecutionTarget { Host, Device };

enum MultType { Acrotensor };

class Engine : public mfem::Engine
{
protected:
   //
   // Inherited fields
   //
   // mfem::Backend *backend;
#ifdef MFEM_USE_MPI
   // MPI_Comm comm;
#endif
   // int num_mem_res;
   // int num_workers;
   // MemoryResource **memory_resources;
   // double *workers_weights;
   // int *workers_mem_res;

   enum ExecutionTarget exec_target;
   bool unified_memory;
   int device_number;
   MultType mult_type;

   void Init(const std::string &engine_spec);

public:
   Engine(const std::string &engine_spec);

#ifdef MFEM_USE_MPI
   Engine(MPI_Comm comm, const std::string &engine_spec);
#endif

   virtual ~Engine() { }

   /**
       @name OMP specific interface, used by other objects in the OMP backend
    */
   ///@{

   MultType MultType() const { return mult_type; }

   ExecutionTarget ExecTarget() const { return exec_target; }

   inline bool UnifiedMemory() const { return unified_memory; }

   void* Malloc(std::size_t bytes) const
   {
      return memory_resources[0]->Allocate(bytes, 16);
   }

   void Dealloc(void *ptr, std::size_t bytes = 0) const
   {
      memory_resources[0]->Deallocate(ptr, bytes);
   }

   ///@}
   // End: OMP specific interface

   /**
       @name Virtual interface: finite element data structures and algorithms
    */
   ///@{

   virtual DLayout MakeLayout(std::size_t size) const;
   virtual DLayout MakeLayout(const mfem::Array<std::size_t> &offsets) const;

   virtual DArray MakeArray(PLayout &layout, std::size_t item_size) const;

   virtual DVector MakeVector(PLayout &layout,
                              int type_id = ScalarId<double>::value) const;

   virtual DFiniteElementSpace MakeFESpace(mfem::FiniteElementSpace &
                                           fespace) const;

   virtual DBilinearForm MakeBilinearForm(mfem::BilinearForm &bf) const;

   /// FIXME - What will the actual parameters be?
   virtual void AssembleLinearForm(LinearForm &l_form) const;

   /// FIXME - What will the actual parameters be?
   virtual mfem::Operator *MakeOperator(const MixedBilinearForm &mbl_form) const;

   /// FIXME - What will the actual parameters be?
   virtual mfem::Operator *MakeOperator(const NonlinearForm &nl_form) const;

   ///@}
   // End: Virtual interface
};

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_ENGINE_HPP
