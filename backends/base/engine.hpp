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

#ifndef MFEM_BACKENDS_BASE_ENGINE_HPP
#define MFEM_BACKENDS_BASE_ENGINE_HPP

#include "../../config/config.hpp"
#ifdef MFEM_USE_BACKENDS

#include "../../general/scalars.hpp"
#include "memory_resource.hpp"
#include "smart_pointers.hpp"
#include "utils.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

// Forward declarations.
class Backend;
template <typename T> class Array;
class Operator;
class FiniteElementSpace;
class LinearForm;
class BilinearForm;
class MixedBilinearForm;
class NonlinearForm;


/// In parallel, each MPI rank will usually create a single engine.
class Engine : public RefCounted
{
protected:
   Backend *backend; ///< Backend that created the engine. Not owned.

#ifdef MFEM_USE_MPI
   MPI_Comm comm; ///< Associated MPI communicator (may be MPI_COMM_NULL).
#endif

   /// Number of memory resources used by the Engine.
   int num_mem_res;
   /// Number of workers used by the Engine.
   int num_workers;

   /// Memory resources used by the engine - array of pointers.
   /** Both the array and the entries are owned. */
   MemoryResource **memory_resources;

   /// Relative computational speed of the workers. Owned.
   double *workers_weights;

   /// For each worker, which memory resource it uses.
   int *workers_mem_res;

public:
   /// TODO: doxygen
   Engine(Backend *b, int n_mem, int n_workers)
      : backend(b),
#ifdef MFEM_USE_MPI
        comm(MPI_COMM_NULL),
#endif
        num_mem_res(n_mem),
        num_workers(n_workers),
        memory_resources(new MemoryResource*[num_mem_res]()),
        workers_weights(new double[num_workers]()),
        workers_mem_res(new int[num_workers]())
   { /* Note: all arrays are value-initialized with zeros. */ }

   /// TODO: doxygen
   virtual ~Engine()
   {
      delete [] workers_mem_res;
      delete [] workers_weights;
      for (int i = 0; i < num_mem_res; i++)
      {
         delete memory_resources[i];
      }
      delete [] memory_resources;
   }


   /**
       @name Machine resources interface
    */
   ///@{
#ifdef MFEM_USE_MPI
   MPI_Comm GetComm() const { return comm; }
#endif

   /// TODO
   int GetNumMemRes() const { return num_mem_res; }

   /// TODO
   MemoryResource &GetMemRes(int idx) const { return *memory_resources[idx]; }

   /// TODO
   int GetNumWorkers() const { return num_workers; }

   /// TODO
   const double *GetWorkersWeights() const { return workers_weights; }

   /// TODO
   const int *GetWorkersMemRes() const { return workers_mem_res; }

   ///@}
   // End: Machine resources interface

   /// TODO
   template <typename derived_t>
   derived_t &As() { return *util::As<derived_t>(this); }

   /// TODO
   template <typename derived_t>
   const derived_t &As() const { return *util::As<const derived_t>(this); }


   // TODO: Error handling ... handle errors at the Engine level, at the class
   //       level, or at the method level?


   /**
       @name Virtual interface: finite element data structures and algorithms
    */
   ///@{

   // TODO: Asynchronous execution in this class ...

   /// Allocate and return a new layout for the given @a size.
   /** The layout decomposition is determined automatically by the Engine using
       a deterministic algorithm: calls to this method with the same @a size
       will produce the same result, as long as the Engine remains unmodified
       between the calls.

       The returned object is allocated with operator new and must be
       deallocated by the caller.

       TODO: Returns NULL if memory allocation fails?
   */
   virtual DLayout MakeLayout(std::size_t size) const = 0;

   /// Allocate and return a new layout for the given worker decomposition.
   /** The returned object is allocated with operator new and must be
       deallocated by the caller.

       TODO: Returns NULL if memory allocation fails?

       The @a offsets should satisfy: offsets.Size() == number of workers + 1,
       offsets[0] == 0, and offsets[i] <= offsets[i+1], for i: 0 <= i < number
       of workers. */
   virtual DLayout MakeLayout(const Array<std::size_t> &offsets) const = 0;

   // Note: There may be other ways to construct layouts in the future, e.g.
   //       block-vector layouts, or multi-vector layouts.

   /// TODO
   virtual DArray MakeArray(PLayout &layout, std::size_t item_size) const = 0;

   /// Allocate and return a new vector using the given @a layout.
   /** The returned object is a smart pointer that will automatically deallocate
       the vector.

       TODO: Produce an error if memory allocation fails?

       Only layouts returned by this Engine are guaranteed to be supported.
       Using a type that is not supported will produce an error. */
   virtual DVector MakeVector(PLayout &layout,
                              int type_id = ScalarId<double>::value) const = 0;

   /// TODO: doxygen
   virtual DFiniteElementSpace MakeFESpace(FiniteElementSpace &fes) const;

   /// TODO: doxygen
   virtual DBilinearForm MakeBilinearForm(BilinearForm &bf) const = 0;


   // Question: How do we construct coefficients?


   /// FIXME - What will the actual parameters be?
   virtual void AssembleLinearForm(LinearForm &l_form) const = 0;

   /// FIXME - What will the actual parameters be?
   virtual Operator *MakeOperator(const MixedBilinearForm &mbl_form) const = 0;

   /// FIXME - What will the actual parameters be?
   virtual Operator *MakeOperator(const NonlinearForm &nl_form) const = 0;

   ///@}
   // End: Virtual interface
};

} // namespace mfem

#endif // MFEM_USE_BACKENDS

#endif // MFEM_BACKENDS_BASE_ENGINE_HPP
