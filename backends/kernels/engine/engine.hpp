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

#ifndef MFEM_BACKENDS_KERNELS_ENGINE_HPP
#define MFEM_BACKENDS_KERNELS_ENGINE_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class Engine : public mfem::Engine
{
protected:
   kernels::device *dev=NULL;
#ifdef MFEM_USE_MPI
   const MPI_Comm comm = MPI_COMM_NULL;
   const MPI_Session *mpi;
   const int world_rank = 0;
   const int world_size = 1;

#endif

   void Init(const std::string &engine_spec);

public:
   Engine(const std::string &engine_spec);

#ifdef MFEM_USE_MPI
   Engine(MPI_Comm, const std::string&);
   Engine(const MPI_Session*, const std::string&);
#endif

   virtual ~Engine() { }

   /**
       @name KERNELS specific interface, used by other objects in the KERNELS backend
    */
   ///@{
   kernels::device GetDevice(int idx = 0) const { return *dev; }

   ///@}
   // End: KERNELS specific interface

   /**
       @name Virtual interface: finite element data structures and algorithms
    */
   ///@{

   virtual DLayout MakeLayout(std::size_t) const;

   virtual DLayout MakeLayout(const mfem::Array<std::size_t>&) const;

   virtual DArray MakeArray(PLayout&, std::size_t) const;

   virtual DVector MakeVector(PLayout&,
                              int type_id = ScalarId<double>::value) const;

#ifdef MFEM_USE_MPI
   virtual DFiniteElementSpace MakeFESpace(mfem::ParFiniteElementSpace &) const;
#endif

   virtual DFiniteElementSpace MakeFESpace(mfem::FiniteElementSpace&) const;

   virtual DBilinearForm MakeBilinearForm(mfem::BilinearForm&) const;

   /// FIXME - What will the actual parameters be?
   virtual void AssembleLinearForm(LinearForm&) const;

   /// FIXME - What will the actual parameters be?
   virtual mfem::Operator *MakeOperator(const MixedBilinearForm&) const;

   /// FIXME - What will the actual parameters be?
   virtual mfem::Operator *MakeOperator(const NonlinearForm&) const;

   ///@}
   // End: Virtual interface
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_ENGINE_HPP
